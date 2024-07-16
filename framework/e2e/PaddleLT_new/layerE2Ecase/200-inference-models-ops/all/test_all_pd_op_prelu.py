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
class PrimitiveOp_1ec43a1f647a5d2dc0df55a48b9fa06b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 512, 1024], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d9745790ff39ddb205b9ff8f4620c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ec43a1f647a5d2dc0df55a48b9fa06b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.44482421875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_124bebe20410c3f1eaed791018184aec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b108ab0c51a3a4486052f63ffe3e6a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_124bebe20410c3f1eaed791018184aec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.306396484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f25b25f1d5ca62fe227316d9bf95ef4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_124bebe20410c3f1eaed791018184aec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.40185546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a962c3f203ba279f861e0f902a24edd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d215f54ed98a40f7febc798d47473135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.12213134765625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b40d2e3d9dc7f60071a3fb564136548(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.005645751953125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dfa5acd43129f5928d8bd60dfcfdd102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.402587890625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9f1f3e5aab8cff6d94ab2100e7c0542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.408935546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e75e30454c1f602b4b4c9858fd316744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0095977783203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0694ff99845fea83b20da0cd4cade0b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0147552490234375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5998483b7e5a2623a6076e10b501282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.495849609375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e49333fa34e6f96a35ea1e1daf59ce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2081298828125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b510d73faaa260bfe52c4fd1da55547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26806640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2f500da45d3f7da17161d264b90cf54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0124664306640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9826eeb10908a627e443bf031b7cf4bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0994873046875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_032892c3331515b58d8c8104379cdfa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.45361328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_154eeb504ec3a91279e8563163959495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.114990234375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dd7a2f3b80e320d4c641c5b88d6d4eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03997802734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53f55da2706a93dd26e56ff026e1436c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26025390625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d446bec5da64bff02cc401888e1a2daf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2218017578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f650d40050db69bf847fccc3376caac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.39697265625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34512a44d04983f1c155c047790e1443(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.148681640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_110fdf1accdef04f84eb4771cfda19d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 128, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71bb325f8fa043ac447cea97d4609ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_110fdf1accdef04f84eb4771cfda19d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.44091796875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fae0ab953a93ed7075c4609c379e3b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_110fdf1accdef04f84eb4771cfda19d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.007366180419921875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 128, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68a7f9b2c70b1c6c13b74d2704415278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.377685546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_887ce724b0a379aab0a63079ba5ddb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.186767578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_47b513ca6248a9617946515ab55bcdef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a995ca79d36e6df65f2445b3d721e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14697265625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f4f50378d04939114cd28260efed40e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.27880859375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53495263105646be2d809c388bb969b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2022705078125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ea2470bd132a1fa85983e7cb0f0cd93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1373291015625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cc5614b392732d9536a62a3b7a1571e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4443359375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b69bb1a68581910e4c5e82a1f1ed34ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.485595703125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d580c98c5b97c35a992f03aa48d27108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.301025390625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf05da124bb637dc657b907d150caf30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.005859375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3ef881275798180be423ff963012e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.36279296875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa4dff46c7f788b6f0e8b2888a2812cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.07696533203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51ca5917e5c050d871c0fae6f6b5deda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.253173828125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07918b0961e58dcd95338513d487527e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0196380615234375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ef240c642defac8f62dd19d97ae8443(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2042236328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b83b90df6f93900e4c2100d7261b14d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.103271484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39ac2145f6793a3066e0b57606b3eb8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.290283203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bde09ad753b8c86f26782f666e114a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.11907958984375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_daa3e68f45997da69c4a680d0e4e71e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35498046875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_416356384498f5d88d54b4b276378cce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.05767822265625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f0dcc34939cfa10dccbfb19d67dc24b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4013671875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f613461f0eaa26e70b1cb7fd09be5dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.31201171875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5780e4fd1bcb0f1b265cb814bd68e60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.134521484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_961d41e8b2337fe8679a4abaffc119d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.143310546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_786e98e66954a951ca20e9b3bbae2b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1461181640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b16559ce2ac4246467e774c1f27bef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.399658203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fc38517966b29b9810e7bbd416d099c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.436279296875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4affcb4d8416a1d94be0a214ae82a0b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28369140625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11f7a359c619dd8cc317ae045b7a3e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.04632568359375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_740ca350c46e5ff66ea1d12e3dc81976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.445556640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5edb43843f42edc9f7eac56a81c4d879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.21435546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a0839bc60d6cb61bfdfa3bb2a18d630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41552734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4eca21fa1fa6fc330b77075b1f0c924e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.29931640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71cd6e4280444054cda99dbe45dc84ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.00923919677734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b37cb34c8c88c154f67f222ab9fb64f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43408203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f166fa91ad6650be7b9d6cbe92de1d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.32177734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2400d241ebb25fec8fff369b28ae949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.23486328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2683da66e7fac1e78599b4664ba7679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.255859375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_219bf736063b646a3b9c408a694e3e65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1796875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e122ce716ecc9448ab8a80b35af71292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09832763671875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5968b4bd7ca63ffa703cbb5f648b8d80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1087646484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7de49f0d272241ba4bb949bce4e7fe7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1143798828125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdd2301f31265683629585b2f9296ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3193359375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08fb093272dc753d26737ee44cdab2fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.425048828125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_998c1a7518ab7afcfad45a34bf12fedf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1597900390625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdcd4c08afd816de221e724d0056a8bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2052001953125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9792111ec7e9321ec72c65acd4ebe5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.379638671875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9a3ac5b023182bb553c07eae09fddca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.469482421875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f99ccb1fe0fce2ce1d048563c4530ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.087158203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54296cd7b6fa8c7e31033afbd86c03cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.18408203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bd4ed465a0fcd6606e451ba2665952f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.17041015625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_958aaec1c914f6644b18f6f2a539e06f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2255859375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac61d50b82c5e28f975b600e155c05a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.151611328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e1db93402cb9fbd17e6453512d67fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.374267578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_729ff9354ac5b24e4bba420be4039009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4892578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97e334c8c9c540743905b9ddad8359be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.415771484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1eccb6b19a1f81797bdc7d1d2ae7788f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.398681640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec70c04eda2b28512b5d433fc68100ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43896484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b72d943bf30a35733d149a21e86b0e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.213134765625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fb26c029489315d370112e70c53d588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0933837890625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a950915da796f0cf0de127b4561983a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.08306884765625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_716e16a5782ddfe5dd36252fdcc1543c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.10638427734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b771f7cd7e6c1e2729d21ab9b1be4e8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87a82174132ad9f7544fc6c964e8f140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1767578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f99473ef2bfa0ecdd3af26e9b902787(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.416259765625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fab7c7166b551e61cafd4d86c35775b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.252685546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbba3cb65f3cb1a0420d1516f196084f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.432861328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39271dc7133e9bae4f8922e8b5462527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2198486328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d325ed45f937145562aa10e329750b76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.12103271484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ce1b3231f2d3e07d1aff4ac443aada8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.32958984375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_eb949b3bd53e65cf2c8e248046d85a00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 512, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_379d04e99ad5a40607aedd3f699e4934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb949b3bd53e65cf2c8e248046d85a00
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.314358115196228], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_492f2df027d2ee830132caa0f27520aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59f9651aa5dea2b31bc8277bab65518a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_492f2df027d2ee830132caa0f27520aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.052421826869249344], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0dd29b7d27b0d771bbcce645fd87c115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_492f2df027d2ee830132caa0f27520aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09084463119506836], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1979ef080e7abea654a076844d5b1bae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e0444b0306c3868bbfd740ceed9763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3915839195251465], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dcdd12467afed588956cce0df43af7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3220139741897583], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9ad112dec18b1c9346710e12dda0a936(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f99f7293cf321796c93f7b7494a7618d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04689391702413559], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4792a12efc52cc4a025d2dedb1f1cb09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08532910794019699], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7089842bba973118304f8a183ec628b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4937308132648468], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f16d7ec6e1642cffdf1ddae150314c2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.010392474941909313], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1318a29c662c2377a96f8c71c32b7938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.019680321216583252], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e8367756f98a29c9b368b1140a19b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4588574469089508], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67e482b4bc168e017e1732f00113ec58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3339824080467224], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8da1d575d1c70942bb67371b40436082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29064294695854187], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee900144502d2a6d0e442aba1cecc590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06224584951996803], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a62d5b80e4a8377559acb440964109d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17978592216968536], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23525ce7d6017b80dc5d9295b2f68676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33237406611442566], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80639e2ace26bc16a6b8e55fe260aac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16752490401268005], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a509dca8bdfbd31566102c878cc9531e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29538246989250183], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f370aec843274c183d15741c78f875a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2335609495639801], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6acb017be08527cb737d79efe48845d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1376357227563858], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f30306123fdb5649607c59dd26f3a1bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23633278906345367], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_50e12a4b69ba863d0af1e8f8be578ef5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90ee4ceb9db0050d99491289afbd5aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e12a4b69ba863d0af1e8f8be578ef5
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.004106772132217884], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45ca34d189410f267dcf44602fc2e6ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e12a4b69ba863d0af1e8f8be578ef5
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35764452815055847], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9441b45d7aabdc85763997387e3655f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ad73baa8c641ef2ddbd50f85189cf6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31079283356666565], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b8c35d62b0e016ba16a4a14f3e8c655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18068981170654297], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66b9f3c31efd0dc5314bb04680657e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10860823094844818], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a78930cecf4cb874d5ed16bfeb52c1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49767911434173584], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_051108ab6aede530468e04a09e626391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28878816962242126], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bf29d2c5a81e4057db4bde6b4ee75b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0720832422375679], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_451db8c0c48f92d641181753faa8674b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33083537220954895], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc3c084916cea0de9c47995bc2b21054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3862634599208832], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b8230cebbf1811ff36f43d397afc957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4179062247276306], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3356f465fb44885c5b459a191f05845f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35635676980018616], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4e4c17eec23f2fd60b7706e02ba7408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.005568895023316145], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e61aa9c88e3ae8595663516ff3da2c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08479639887809753], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d91bd273f20af079e765e35d4311f0e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.012169374153017998], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_656002a53bc1b35705eb9d02ff62c510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10607193410396576], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc8dc2c03e2c10a5a0950ee6f0ba7df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29513606429100037], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_609f86dcd35fa3565a56ca5fb64b4ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32401877641677856], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d47406191b6eeb25cdf43bf1adf83ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21968157589435577], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d6ad0d7f8592ff2ed6a0584a8e9a05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23008735477924347], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c26565985cec8bcb47c91186cd93d133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19388431310653687], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc78937f66ad91146b9c3ec669facd7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.330475389957428], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80802f9be3edeb6beb23573cbe5d43cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04432373866438866], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1db792bd27edc566c308def3a4a9586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09809526056051254], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1e9f7e4e66c19f66eaea1e64b1ab25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2747535705566406], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f26cfb645c73c95b8998054405f1ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3237344026565552], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ecfc852d46685839549d3b9b552ce1ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.051573317497968674], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c99b46d26def887e8945d03979e0a70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13795672357082367], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d69391633c23def39aae16d51798524(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4679030478000641], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_249d5bf7f69e74178421290dccfcf8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32427895069122314], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bbac14f2babdd8453743fa10ca06ecf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28791937232017517], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f848a37d562d2d68b453e9ce796fc68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.477694034576416], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbfbb538791b5a5706b65d91c1dceca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09003639221191406], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dc870307ab0ba84a49f7188cf004202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4373798966407776], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_526e167d88f2c5bea05f32b39f4d6e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.026438254863023758], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3e1ab0d7886b1ce0a03e9c4b6251ba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49493318796157837], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_602b38781e6568ed7309b00c037ebe9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08877985924482346], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ee160d69ec4d10191a7b5eec9bdba47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12529967725276947], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77819a46588b56c35b25215463fc529b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26045963168144226], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7513350ec92761ff6b2c2c9523caccd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04828999191522598], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbffe4de45a528e598af930f201a03a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34927839040756226], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10ed66dbb961672581c8cd78ce10645c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40646424889564514], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2314b5c81cfe0ceecea573731d4f841d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23910847306251526], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9f70f576e96b13df2a0eff02c9a2c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11615440249443054], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b29d2f34c1c8cd34e11f7405f9d220da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.47520142793655396], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f8fa1219a74b3112510dd70b73422a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22972087562084198], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_451b51043d63717efbd1d7f030cac1b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09742721915245056], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_863c20eb1ab9c1cd670f880b3f97d7b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03373267129063606], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88352d2a1c7a0ac437aa305dd5adca23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3002816438674927], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe324fa50067cd6bd341a179878b9ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32523679733276367], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c3b29267b84d7e3aee08ce84ac516dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14292868971824646], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a77a84bde20013950be731f1ed1335e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.009021447971463203], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43bb2fd9d4ef7d71d40b88b04ea05660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1764586865901947], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5223f12fb41fc102734ae319bec66d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34536728262901306], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6046029e24a6b200e7471d31cdd93d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08105312287807465], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f59f7a61278846c836f202ce9cf69d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3045939803123474], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e9623730a0887ae3890377cccc748ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22427093982696533], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad862633a78a0d6cdf190f32d89aa00c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10214207321405411], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e99807e66a985e93e55203a313da9db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34631118178367615], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87b896c98d0ac65cbf26f74bac4c5444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2616497278213501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c638beb7bb5e4dfb9132ad7032bb32b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13935445249080658], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_908481c022041569db3c0c15c92c83ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24144785106182098], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad57a5809455f74b34c41b64456417a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.43686866760253906], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c28c0338652f63c992c6e5616990a737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14641542732715607], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fe383c9f14694d7ff723b62ce0973b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.002417148556560278], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5230548af3cb437cd2dd4f18dce3fe6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.41789016127586365], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5cc648ab2826a97678ce4ab1a6bd8ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24127952754497528], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d1b84b0f3f23d02670e71f46b349786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.01861017569899559], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcadaa3fc32eb3fe58e6c3d132f2e932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18633492290973663], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51c82499154500bf9bc5721635473b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4888635277748108], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_076c73a4afb2886e2f1b250108b8265f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13422684371471405], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89234e2fcd013eab3907cfc4337c6bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.01220704149454832], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_512a27636321d96698d83ca04d3b6bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.44482421875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a003a4c181ee99399ed3dcdb3fe8391d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.306396484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2fe9e0ea315ca81edb9702a5a42aab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.40185546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0032c9c75528213c66c960f9abe42f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.12213134765625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49a082c90690ad984856f477c4385a5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.005645751953125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_983a5f88897217d73d2691fdb546e39f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.402587890625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_351040e72cbeeda93aa8f7c70b50bc00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.408935546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bfefa11ff60e32f8862b1b84e94bdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0095977783203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_590f416c12896eed85eb74fbbf8a0b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0147552490234375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd015f8bf01e023d2fed0ae8d270cf3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.495849609375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77c6690f5d6323042a6a51573eab3baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2081298828125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dcf88e41536b9e2a0c7b4921758e60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26806640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bdb9d66d4dc1d5f7cc667d305e95030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0124664306640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cb0e531f3db17b800744e8524a4de6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0994873046875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_550e78a2c7b06620892c56337eea8d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.45361328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce88d8335d04b51a10e864ddb9f32720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.114990234375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50e6edb3f8ef03ebed229f126c9216f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03997802734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea518d188b6b135af4d0fd6bd90a5b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26025390625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3929455247645c74387a58a82bc43a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2218017578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_442e52186c45769ddee8a2c5a4281c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.39697265625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49c60551a4527bcc107ae18eb3c53651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.148681640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c97f19509f71039cfefec42385fbfaa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.44091796875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d026cb454b5e2e2399949989f2e57657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.007366180419921875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d85a0f6da3ce8dd858c52466a6f7532f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.377685546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce600010c2c02cac8d9e8192fe95c308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.186767578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e814ad23539b2216b045345059324303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14697265625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63b72f95536157e0495defba2ea51d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.27880859375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be7bffe2d9e93ec937f8f67a2b5aa089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2022705078125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_733ed54525fdffcadf02756e1fa0fdfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1373291015625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_583def7b2775d869e87e2082a4cdd7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4443359375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac790f7a835f0cdb5b4e41d4b9ea463b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.485595703125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a57ce9cdcca25e22024d164bab6e51e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.301025390625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f6243900181cfce6b9f9d2e444ea8e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.005859375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d767ca4a6ae37f8b60ad6b4355fe581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.36279296875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5edc55c42cafed0cafeb10e911a8f57d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.07696533203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdb1c48de88acd74a9d2b97cfbbe9c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.253173828125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30344460db1b7a4fcfa7e4df0b30bc59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0196380615234375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2c2793f3082f7edcdd398a8e0cf795e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2042236328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0fb3fb92f085e35890398fd27d8bf83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.103271484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ceeb7f79fca5402086d4480ffa1157ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.290283203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6df094b6fbaa555f40718303a38360f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.11907958984375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7329338d83a03aede1447eb0c713cf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35498046875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc9ad912653a96b5a64b741ffdc7b283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.05767822265625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63bdc3403d40920008a7c90560bac8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4013671875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a56a577fa2ca410cace65070b5164321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.31201171875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d471bca749e092ad1b79a7a80d4efbc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.134521484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_396593fe277e6ec6fac7946b2a13458d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.143310546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_602a7efee9a7c6c2e5cb041804a398a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1461181640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e5832d903eecfb8869406931f7e4b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.399658203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73fb5a8ee9669a15f38351dfb83b688e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.436279296875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_279c618f1d897b5e3663ba2ecedc34a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28369140625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c560bd372cfa23482987b1bcf9d70a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.04632568359375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f168886cb3e9ea6a9b81af923cb102dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.445556640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75cbb7fe10e7f652d3eca7588bbc28df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.21435546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_757a23c8fc0980cfc060ffc39023878a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41552734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13e49e4c243e862888160dfa22e02399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.29931640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5172aa943b47d7c3ab1808b7032c6168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.00923919677734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e0de5d1bea150feee3dcc47bb1879ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43408203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_003f7ec4dc3a1f9875606350c9dccbd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.32177734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4257fb160688c19961133e8bfbff7129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.23486328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6c18a5cd40f660fb0711b34b56a87c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.255859375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10c653224e8233e1d0dd34c01b24277e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1796875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_340a0c1bdea14b5c2cf5bb0c4e7084a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09832763671875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49b1c1418367fc0119c4a3db11d2b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1087646484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ae54028241d268ed3d3dc31a0871421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1143798828125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4113e879a9663e4e970e044a2c6b31db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3193359375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efc3a8899ee2583fcd90a5961136a6d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.425048828125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69ea7420b7767daa3f7996d94c7fb448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1597900390625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bcfc9f365c49cba32ba9849b9552d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2052001953125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c23c396de3f997a246cb56033879c12d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.379638671875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a40c456d304dc709dc407b35acc790d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.469482421875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67d79a41c427b7d041636fcc3f706727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.087158203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dafcdec6b95d5523b3f14c416ab2f0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.18408203125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4afbccabb08abd432fa2af74a93651ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.17041015625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1381778ca477c31f2f8962418d35435b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2255859375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17f8eb85b142d5ea23d584e6e3587b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.151611328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90ceb72f979087753ee6cfde3d291f4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.374267578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37f884e7593d839e5efa7f723677e607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4892578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c11b0b767d9135912477fb36fce610b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.415771484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63b0c429f711528c2979ea3739ef2b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.398681640625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adf09e42a19062173b72a68fa42cb543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43896484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_731b18347505067de2b64516946f1d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.213134765625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_879e841b0036cc97365c1aefa2eb4c97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0933837890625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dd327c4f7d07608da714de6f7b9eb4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.08306884765625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b6369f9a613fdff1bc625c61b0a3b09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.10638427734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_919af387aa927f8ea79e75c4a4bf682c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6b8b2f5ebad4635e9f7feac8194787e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1767578125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9288519c74062d586eb1823b14a88936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.416259765625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eca5ac53a38e47885ad19ee1cbd9b1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.252685546875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_331662faa3639e07789e16294ca364ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.432861328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46ec831b934ce1cb19faa4a339479d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2198486328125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e2ccb0af2c74f6728db7f3f8e4eba15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.12103271484375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9571b2107aa3e56db53f825c706bd2b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.32958984375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24eba0a0784de5be98a200cae54553fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.314358115196228], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b759ca6b867465ba9d5818255f1395c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.052421826869249344], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd95d37ac76046d6fbc525c21c0f8b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09084463119506836], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbbb52ed9e4a27225c142430c7628550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3915839195251465], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89241de6f955f4bcd6ad08990fb0e42e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3220139741897583], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_550ba47dd944bbd759ebb497899d6cf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04689391702413559], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed5b5eb3c47138040c7faa5d942cd1c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08532910794019699], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c67d23fbe6131f7b123791f60c6e0aba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4937308132648468], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b83e6702ddb9be348433c12cd2d60028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.010392474941909313], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efc8136dc93f34d6c5659c0f015d8ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.019680321216583252], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d2694191cdef0e03c2faf5700bf6547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4588574469089508], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38e3fc377512977777d03bc8eee03bb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3339824080467224], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d9465bbb2b4b3c63ea843ed3005a0fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29064294695854187], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f518e7b55d8dea3832973b6e8b61585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06224584951996803], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed93e80a35acbd04296d788314a495cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17978592216968536], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcfaeb1bc9427dc4c9e8250d04ff96f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33237406611442566], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d132e3dc404b52540f5599e2986466e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16752490401268005], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_077aecf99bc366ad238c0cd468a97ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29538246989250183], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d718265ba76aa820caef5ad6c325d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2335609495639801], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_740ba360c0d934d4c08781caf1b2798b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1376357227563858], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9656cb1dafd311676686f51a439ad75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23633278906345367], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad7a3e741e5924df3c5bb5d31a89cdd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.004106772132217884], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbf05cb2dd7c31cbca1b329f11d4a426(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35764452815055847], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_501f2dfd642c74650f8d333e1c112904(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31079283356666565], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60b8421631670e259695605abb390a3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18068981170654297], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b112df6b034719a3fcc51e5552c3b36f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10860823094844818], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_521a0c1db2c80a5eb5cda811cc068b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49767911434173584], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58bbfccc17349b03d0d91ece313657d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28878816962242126], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_218e66fbdde4b1f5b414f1dbf6fb193c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0720832422375679], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2d2078277049280858f012cb60f44c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33083537220954895], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_173cb53b5d13310bcbe356cd1f9b98e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3862634599208832], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf833ac99ea636f79cedcaaf41c416ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4179062247276306], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e88ac9b8a14edd5cb15487ddf211941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35635676980018616], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e55c2cfed54d6ea036aeab7f4a15153b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.005568895023316145], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bfcefdf51f72c9ced1e29cead2885822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08479639887809753], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a50c2b97a860e3396f154e52604e6cfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.012169374153017998], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32f57854bf41ebc3cd1f0d6817e58876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10607193410396576], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74937d5ef9fb3ed0f315f0bb0786ec34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29513606429100037], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3e1411a6aa6487c9bc3245d1a8c3a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32401877641677856], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df85cbf2d2de0062cf8703b7536097aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21968157589435577], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2d3882eca451188337805a0640aeeda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23008735477924347], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2efeb1ceee32033c2e7127abf2e4c96f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19388431310653687], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c03bae832d5204b65c58bd37fab4b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.330475389957428], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e210b3d72830fe2c69bed03bc3f3e0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04432373866438866], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5455b645b1f7f4d4148bb2b244e28e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09809526056051254], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f3fc566751d7484d607b86e76f45b72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2747535705566406], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35b0344a00c78a050825b9d7653343c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3237344026565552], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c2d7145fa919556be0a70a6b7d3ab2d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.051573317497968674], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_facae96430acf7dc55624fcfbf352af8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13795672357082367], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd77aa3201688efe1711b861ee5e01db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4679030478000641], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa05c42e366f92f3ec0f5d09ed2b1df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32427895069122314], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec680037dec0263ef4ddf52a788b1be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28791937232017517], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14c49aec27df4f3248b4c4e590531631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.477694034576416], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f07317aa15d3f5ba123c467ec237c29f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09003639221191406], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_451344d4dd9169895a03f56713f75dd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4373798966407776], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9eaa5f2329ea6abb0eadb3683461866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.026438254863023758], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86fc890d6db0f53ecc00e8aa6414d059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49493318796157837], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff70a8c91959c7abb7e559277fdf9ec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08877985924482346], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f1489d58880bf387e73902a7db747000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12529967725276947], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_513f345ff26067d81e34fb3906c55048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26045963168144226], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8c0e27c1da091c1429d99273e9ecca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04828999191522598], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b997a6bba7f460390f7ca01619d7b80f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34927839040756226], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1973cf42d30bd8515932a41a733cdb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40646424889564514], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_894e4300067bd800b6fe51dd550251bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23910847306251526], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f65e7974306eaa03d4c997e303ea797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11615440249443054], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c52300cbb31c49034a2a060eb02c3312(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.47520142793655396], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efc568fb7d60cda45beef216cfa52c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22972087562084198], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_261535c5fb8d3dda3e1ea888360568ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09742721915245056], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5056784b6f63ca2d8a7367acbe6548ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03373267129063606], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4bd166e5ba183d340ff52ebe29e75ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3002816438674927], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd9d8c9aa215f08cfbf5582500301385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32523679733276367], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4555199e560afa9217c1922bf95f8bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14292868971824646], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdaf5cc31cdf39e312a5397db1e69c0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.009021447971463203], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f16ffd73792cf6b53b3a5dd3a1dfedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1764586865901947], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c160c0fd784c50fafd3c8011d7ae4053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34536728262901306], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a87e96a7834745cbc5f6f6cc18e0b8f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08105312287807465], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb72bae62f3ced3ca47b2c9d8408c6d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3045939803123474], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e3f0b73c2c0de44d0dc71bf8396fef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22427093982696533], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d60c5ce1bb4ce1ac09b039d9eca817b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10214207321405411], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17e4acb1bda339eeed8ca7db3fd2029e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34631118178367615], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c110344a96ecb43e5a3108455d9ed98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2616497278213501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b49263a34f85bcc9a585eab52edfd73e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13935445249080658], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd3dc6600cff945daec58e9684f1d971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24144785106182098], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f94d9c37149f46bf7ccdc8ed82bd063c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.43686866760253906], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea5e8e696c2e4b8dc0abc62d92a253c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14641542732715607], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cecd41deeb7de0196c872ac4ed69fd41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.002417148556560278], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5998d2a8480ad87e6aab7e4f1a75abca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.41789016127586365], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6a29bdf66b046378ae25be276c09eed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24127952754497528], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a75a6a5b083da19480b2c78180368e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.01861017569899559], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce7d1600cade5862053df9192130f3dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18633492290973663], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44fc31f2feda1e281c7b5e69386dd7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4888635277748108], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8f034792b84afae2668dd40a3282e9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13422684371471405], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e19896e77a05237ec8235310e52b08ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.01220704149454832], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()



if __name__ == '__main__':
    unittest.main()