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
class PrimitiveOp_775fb8a249b0606af5a28c3a049a9bc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3e2eaf3eec1d69cc59a20190f0e67dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775fb8a249b0606af5a28c3a049a9bc4
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 400], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c53e289d679c88c463e9b11a2f8bb0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd98d54309d6218b20c62b453e74ec37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 56, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0b3ee05a4ae3ca57553c34a4f520e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 120, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de65859707c9287f3dfb63d5053069ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7adecbce7b099c956aa482ba9a8b6ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2d2c7dc5674efaef0ecdb91ab443134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5386e2a548b8dec74c32e8882174e97f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_048ffd776d86eefc4860bb00eab1d7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 32, 100], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d60dbdc77be42ce1a30d2e1ef70b6110(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [13, 0]
        input_2 = [14, 1]
        return paddle._C_ops.slice(input_0, [1, 2], input_1, input_2, [1, 1], [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 54, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c4c5c746a2a44bb055c016ea7946671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d60dbdc77be42ce1a30d2e1ef70b6110
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_37436a9e63651329266ee1b479ee448a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [13]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 54, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1a038e3c679898f2be4817f80536dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37436a9e63651329266ee1b479ee448a
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a36f10f492a81bfdb75d83ba74fe1632(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [14]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 54, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24e087f710c4742af2210b24daeb8723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a36f10f492a81bfdb75d83ba74fe1632
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f8cd58df6f73063428343dce78587aaa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_913bf9e7fbedb5361e9695e0fc8c3b9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 20, 2], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb56df5bd7cb96931e312e1a6d73671a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3200, 2], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6db75e0fb0f73f0fb395c5ac504e1b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 26, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68d493f54ef0f4fc95a6def5ab6ce8a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 26, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_360e3007632ce6cc01c5a28924604ede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 320, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba1bcb9a342bcf46fc0776da3119a7ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 320], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_82b4cbab6085fd68d3dbd2f07a4d3a04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 5, 196, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bd8d65ec2039fd65188b1f1738459bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82b4cbab6085fd68d3dbd2f07a4d3a04
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_cacbff482fc316de98c7843bbb35a6c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 5, 196, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_829171777c8acb14333cacd2e36e4ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cacbff482fc316de98c7843bbb35a6c9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ec68e1b3f4945b24255c28894e7ace74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 5, 196, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c3b2b7e3c6a83b5fa1e1ef39f145545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec68e1b3f4945b24255c28894e7ace74
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f1ace9370948e452103988d5db23030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_123616624bc79d4b099137f7a5994203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7f7b066c9f22871e6a853bd2ddb7fcd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58a0c107c7a87ad6f33e436d0695cf72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f7b066c9f22871e6a853bd2ddb7fcd0
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1d6eab60de82690348083c5d1e9fa1fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e961f546f38d17d9c659bd2403b5c998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d6eab60de82690348083c5d1e9fa1fa
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2fddbeb9968f68a32d9426722f443844(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec4528bee231775dca459c657d23dc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fddbeb9968f68a32d9426722f443844
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_cf19822150555ea3e4a2725534f67a5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-2]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28ba28108cee8c6093a420c793c7cb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf19822150555ea3e4a2725534f67a5e
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 18, 128, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52de5747556d04c36041f123d42ee7c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf19822150555ea3e4a2725534f67a5e
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 36, 64, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac4e1bd3714c7cb71488d14404d61214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf19822150555ea3e4a2725534f67a5e
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 32, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_363fa46e17df2298a58be5af5a92c352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 18, 128, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_170a305ad9a68e28b92b6c65ad3b64db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9367cc16a6bd1a4bca55e3cd3974f306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 16, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ebe54cb0e6fda6add5781e5e226bc6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 576], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d34116d57a92995c537f72ef449359c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d7f1144f1229a82ffdb34b938afa13c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d34116d57a92995c537f72ef449359c0
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[501], dtype='int64'), 'int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e5abaacb62d3466402db6e4a9daa8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 160, 160], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_44da07ddb58f97b73be00e5bcac7f5b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_261bd02bd9bd4c851c6831e499a29529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44da07ddb58f97b73be00e5bcac7f5b6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 160, 160], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c1bf3169064e27cdedce1d18bf03a4c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec82305844231e3e53136ef7bc88f506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1bf3169064e27cdedce1d18bf03a4c8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 160, 160], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50fb33bc566cd83e503af7df6c4d35e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 224, 224], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7014ce6bf8b3607a8af4dcedad873f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 96], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0050feb27fec77da463974e4d659c514(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 64, 3, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6cd81c008803c5d478bd7f1675f9ead9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0050feb27fec77da463974e4d659c514
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_15cb3205106553266e501720b2c78ba5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 64, 3, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cea8d157ef642bbd0ed87f70d48fa100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15cb3205106553266e501720b2c78ba5
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e7cea3bbd4246bdabcfa8433419dd1ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 64, 3, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b459fba835d018d41bcf0f7c10f2f9fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7cea3bbd4246bdabcfa8433419dd1ae
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5ed65a152d8968845ee64cafb58735f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 3, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3842bf8485c5f66c42891bb7c8a9f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ed65a152d8968845ee64cafb58735f3
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d2518c73734023bb628e6b3e39213f70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 3, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acd0f645ad593121a25ba37825ebe79a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2518c73734023bb628e6b3e39213f70
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd712f0ca9d1c385c5a69b17393791c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8d3335d43644aad7655d51923ed7a6e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 16, 6, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_924ff5786e403bcfa7a8e647c5ce4326(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3335d43644aad7655d51923ed7a6e9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_08f5c7928b48bf5db15a86c2cca66eeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 16, 6, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d07185ee0f18df0827604018af9012a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08f5c7928b48bf5db15a86c2cca66eeb
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_457ee6026f7a0b701ed3d55279aad0c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 16, 6, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6844ba2c7c5bf7b0866e77b290c97f57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_457ee6026f7a0b701ed3d55279aad0c2
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a6f3848d8bd45cc78371a67f992e8b34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 6, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afde699f0c74be8654a116cf8c2a3d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6f3848d8bd45cc78371a67f992e8b34
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_df16d2ac111e6657678ca6893d95cd5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 6, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c8a86893ae8a6783b468fdd993fc18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df16d2ac111e6657678ca6893d95cd5a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c24ce7b963cfe09cf7c9e5e51e2b8d18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f5d73427b6396842e0a891923c83a2c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, 12, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_003f24aaeefb2449af1d63d23e2d7aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5d73427b6396842e0a891923c83a2c8
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8318a93cce5d2b931f4aed3d8b72b903(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, 12, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f04f05bd17bdff45768519115ec7d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8318a93cce5d2b931f4aed3d8b72b903
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8364cae6bfc830cb4e8c173125819bcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, 12, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a7e33facc3ff92efde9746c8d129d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8364cae6bfc830cb4e8c173125819bcf
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_23a0b5d5edb3d6198c7d6b2cc4a56736(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 12, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_443e7445743ed599d0c2fb2f40b78eb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a0b5d5edb3d6198c7d6b2cc4a56736
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_dacdc319b02d65aff4a0a013b6897db5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 12, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87c1e43b16db558d96ffd38c0cb636bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dacdc319b02d65aff4a0a013b6897db5
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_078b2e4cb3cea0f6a32c635b31f50cce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 768], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3ceb7f3e4621a584ccd7f0bcd536c6fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 1, 24, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e9ab1c8d75dfd6921988d712ef7f8a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ceb7f3e4621a584ccd7f0bcd536c6fe
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4cbf00b3204824d5462cd03ce101eedd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 1, 24, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a8f95fc86638ee283597d86930a88e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cbf00b3204824d5462cd03ce101eedd
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d2d5ae69f704631f76f67771bdfc293f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 1, 24, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_330d3e70e7abcedea5d3c9332e531236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d5ae69f704631f76f67771bdfc293f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7e42be44a8259363e4e0f66dc2a23650(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 24, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_096650eb444e57bff6de1b3fe83a4fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e42be44a8259363e4e0f66dc2a23650
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ce19dcf5626bf536e9db1040fba4b6a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 24, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8b9c24ad1fd8c270943d0c927fee04d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce19dcf5626bf536e9db1040fba4b6a8
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29ff8ad1aad30c2dbd9b1c4ab2fffd7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 144], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3ee0e8567e219a7acb2b12f03391282a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 256, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64f7c0e869c2628dcfc07b2b1455f339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ee0e8567e219a7acb2b12f03391282a
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f6a86abbef2483b14be5d640c7bb563e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 256, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed014f8878b975f91201c7db8207e2e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6a86abbef2483b14be5d640c7bb563e
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2d4560a20fe5dfd90bf7b7bb3da46230(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 256, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d9687684b922c826739036bbc88cc5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d4560a20fe5dfd90bf7b7bb3da46230
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ffb86b585ec39ae2a00b775b4b1ed88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 64, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_508a31db5180ab91a2563c4c0bf7b01b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 64, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e1546e7c3daa3baf9235c8ccd3e5245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_508a31db5180ab91a2563c4c0bf7b01b
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3deb7f9e1e4c385a06b12972ad225677(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 64, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b87fc13224959651fd870ea8b842318d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3deb7f9e1e4c385a06b12972ad225677
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3bcbf1ff617224b620d8419ea6e7d80c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 64, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc5c7233cf5fe1f8d7fcfb3961707cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bcbf1ff617224b620d8419ea6e7d80c
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0781a597ffbec72f225f41d932a5e4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 16, 240], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8543aae511cc9211168bbedf1f7796f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 16, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3538843971a2e387912b9880f466a740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8543aae511cc9211168bbedf1f7796f9
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_07d906d78a1c59fbc8ef9b107dbbf82b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 16, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8911f46b2da3620e1eb3b95c9cf6ebba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07d906d78a1c59fbc8ef9b107dbbf82b
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_efc457de497fef1101d1ea7823cbd03d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 16, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_741756c01fc938b43f8a0d65e6006866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efc457de497fef1101d1ea7823cbd03d
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f267ed033d27702ea811125ce20b784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 640, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98bb835a7f58e09cfaebf5856acfde19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_cbe51b9b247d7c4aff0f02ee1bf5bec5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, 20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de143fe7e545279720cfe7b4059500d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbe51b9b247d7c4aff0f02ee1bf5bec5
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d962a8e9cf986bc172043e5c78687cd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        input_2 = [258]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, 20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a03d0ceeb840591e341b96d3682ddedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d962a8e9cf986bc172043e5c78687cd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf007e2eb9ecc5f3e8c8c78e39223be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 255, 20, 20], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3d4daf30d54503a2b17227e4e807e138(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 85, 400], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d1e7679d15529c92a19acb8e9a6fab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d4daf30d54503a2b17227e4e807e138
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_36ad149ed584e77ed7a2748506e99659(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 85, 400], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0caf1e07acd98e2215a6a8db7153a4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36ad149ed584e77ed7a2748506e99659
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_77d5f0467c2da28836eceae76c4ae42f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [85]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 85, 400], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5d68989529920ee956cfbea17126014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77d5f0467c2da28836eceae76c4ae42f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f793bfd6888489dd3d82b44d9944b58c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, 40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc6f3f95e8780bab587692324f393e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f793bfd6888489dd3d82b44d9944b58c
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2d5370c9f5e581b4139ce04ee98841db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        input_2 = [258]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, 40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c570ef11a70980011124b7692930459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d5370c9f5e581b4139ce04ee98841db
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b674ad101af6d68a083e271fb06d8085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 255, 40, 40], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_94fd763f11883dcec3d3b059bd92a622(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 85, 1600], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1525e37993cd97735f5f2678966443f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94fd763f11883dcec3d3b059bd92a622
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 1600], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1a0ba721fcc7f335e88b5afe991d09f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 85, 1600], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50e95ec644579e07759e021d9d9a6cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a0ba721fcc7f335e88b5afe991d09f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 1600], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d2c0c4dcafa93b307bf86df61b877a06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [85]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 85, 1600], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ed97d1529937417a93327b553e29e34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c0c4dcafa93b307bf86df61b877a06
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 1600], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_eb7007ca2f2512a25dacaa023f6fc21e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, 80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_027aacdf587ccb9a1b1b972c4e06e585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb7007ca2f2512a25dacaa023f6fc21e
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ea8e7d9dbcac124e98533ea478cfc1c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        input_2 = [258]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, 80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18e05fed99ed68474878a13691cee672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea8e7d9dbcac124e98533ea478cfc1c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b362446d236e62ef4c7f58366c07d09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 255, 80, 80], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4916a40f2e984ea73a38722860912e7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 85, 6400], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aee5aab306fe2012790698b2752df747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4916a40f2e984ea73a38722860912e7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 6400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_78034c7908d8d78f53ca94ab1e15a9d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 85, 6400], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_738df27a5d8a450ada22b9a4a3cf1aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78034c7908d8d78f53ca94ab1e15a9d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 6400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_25ba85733df712e92999d47495e60ccd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [85]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 85, 6400], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b142e0cb08922df10042ca38cc84d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25ba85733df712e92999d47495e60ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 6400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b91ba139523613f0c601ea20b51ddfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 32], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9ef8ccbf4b09a4b4781c51225ea8e804(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 1, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b23499d692317824069a9a622175e489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ef8ccbf4b09a4b4781c51225ea8e804
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_fb0697f6ce1fb2baa44bb4794209fbd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 1, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb0c771764f56f9ea2b541a306577237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb0697f6ce1fb2baa44bb4794209fbd7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db8a65e6d8242a346dcaed478e2e85d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38512f2cd618a3eee8abc3519bda8192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 64], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_50c64ce6d08d6b6c12a615ae3228232b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 2, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_246adf0ae3e2dc778d3093307c9ea459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50c64ce6d08d6b6c12a615ae3228232b
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c44557e1e52c9aa5ba665445200b1121(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 2, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_300a5be84eacfd52340302f01eb71676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c44557e1e52c9aa5ba665445200b1121
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e0ea259e80bf5c4df63ebf5b159376f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_593949c9a8f1fa20d4f4a7d940f38f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 160], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_422d7dd295f73e1d410557aeba54175a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 5, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9484c8762310a4e0b34b988fb7aa3f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_422d7dd295f73e1d410557aeba54175a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6feded31498b1c9e355d9b1e9450a452(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 5, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bcd1ad3cc0038cbca8be9b92ff9d02d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6feded31498b1c9e355d9b1e9450a452
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b4e85b6f061f1e2d6019a401f6ec774(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 640], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88d3c883386a206fbfe0769175b03d2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_92583eff7c20740643806ca4e9f2fb88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44b798d3f6c8a83f6da90756c2ca76ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92583eff7c20740643806ca4e9f2fb88
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ac34e4fabeada8c2de4834d01167037f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 49, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94b610aad251ca87379c6127ecefb561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac34e4fabeada8c2de4834d01167037f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf27b8d00756c844b7e536b1fad1a90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 1024], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c028068729b6de3ff35cb8927fc0e582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c77b8408149e9ae511d691844b722927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 4, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78d41a20069dd5740d3e3850d70f3a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 16, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efb35b3c1efd6392f811f134df827de8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 4, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28c8d31a4f315f16f772e873b39fa33f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 8, 8], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ec331fc06d447d64a90f2a8d81d797e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 4, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59721abe7d20b994b76842dd37bc37fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b36f4a846eb7c9ccc3621d183112c58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33519e7ea335fc797e9b5334a84ce360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e37474e7cdb1e2e9849733909dde3c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d99faf3462fc442e93bc3c23915ba3c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60cff85f8e6d17993972fe6842a645ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_622c9d9e3be84b4e0ef4cb98e27ea45a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f88e800c00a7f79789c63aec94b1e513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e02dcec6c067d28a056d24e3187f9dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 1280], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adb23bff530a34811a2344c8e0a2ad88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 1280], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c9ee00f2b09e94b3c7008f0a0f703c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ee4952b3d540c07a2f49d6a02ea7c9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de8eca1f74bc34792d729dbde3d2cf5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_addbf14aad73661f7e0a5556809e53ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f89e4644e287cb26e9b882914e5b2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 768], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_323ade1f8f939f8c2c60a512aff05463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 768], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dfa7cbb211a5b1b73f9164e6abd5f83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00abcdf184a3f283b26254441cae08d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac72a7af3d3debe7e6b2acf065522ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 2], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ce31e5495757540503a65b54525f774a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40166dc949d86a167711b7f3f3c41c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce31e5495757540503a65b54525f774a
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_13b0352da9262351beced2cad80c837f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af6ae1d3b30a135c45ff18a9109932ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13b0352da9262351beced2cad80c837f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3fd22e2bd4b5933e0f760cc1ec5ac278(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d578c180869d022f45c8b85a5b905d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fd22e2bd4b5933e0f760cc1ec5ac278
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0a89a1a2a0c6fe5120da37afabd57820(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e10727c09957b0d25d635bbef8ea101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a89a1a2a0c6fe5120da37afabd57820
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9d50d3740d2a613e6d09d05487fe23bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2ed9f8e0963aacef5dc2ae25027ce58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d50d3740d2a613e6d09d05487fe23bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_51e313303c8a2a5f7f6eb53044d1428d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a4c888d51456566c39defd7afba1b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51e313303c8a2a5f7f6eb53044d1428d
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4e668f94798b1353fd0919c1a4e51171(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72c62694ed99608ef442cf8887ed0ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e668f94798b1353fd0919c1a4e51171
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_eeb28752a89f46e9b0fc14c805e1fb66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddf98fc47230b498cf7ff3dd8c41be26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeb28752a89f46e9b0fc14c805e1fb66
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6fb0f06fb4914d1380101a2958d1ac56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_822bcd2b095dadcc3d43f037d422da81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fb0f06fb4914d1380101a2958d1ac56
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e47383d76fb6f80d9d6e679352371c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9216, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96dc74f6a8b57da07391f6fe9ff72563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 96, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6f01fa672cca76a5799b805c289f82b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 144, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_193c3eb3617268e600fd7deb5a1a711c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_801081848f120efd60897c2340e5a208(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_193c3eb3617268e600fd7deb5a1a711c
    def get_inputs(self):
        return [
            paddle.uniform([3, 64, 4, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5e9376c6c9467b4bf41a605ee007df2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f933306835229b69fc2aa3c3906c70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e9376c6c9467b4bf41a605ee007df2b
    def get_inputs(self):
        return [
            paddle.uniform([3, 64, 4, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6114dff0c656e59faf0e6f8504271e54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d83d754ffd68290248f1c81bfad89a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6114dff0c656e59faf0e6f8504271e54
    def get_inputs(self):
        return [
            paddle.uniform([3, 64, 4, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ba52a663ea08b616d69eed68dd922f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2304, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49e6331960194cdf3692284303d5fdde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 48, 48, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fb6e5ed9742b15cb8dddf17c4088934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 144, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_163402520c8148d48f5d814d13651bf8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0cb00588298d5bc37a52b4319df794a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_163402520c8148d48f5d814d13651bf8
    def get_inputs(self):
        return [
            paddle.uniform([3, 16, 8, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_04656120f356e3943217924d09e6f97c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e42daa7044f4beaed03f21cea1406753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04656120f356e3943217924d09e6f97c
    def get_inputs(self):
        return [
            paddle.uniform([3, 16, 8, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c483bdd3a122e9ab623c3cdc7d371fca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5cb0ba8bdfa876d17d2b85ec9387db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c483bdd3a122e9ab623c3cdc7d371fca
    def get_inputs(self):
        return [
            paddle.uniform([3, 16, 8, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7fb0e5c42f9c458432b41b89d976be5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6281a7756cedb0c3167cce9057a7ede8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 24, 24, 512], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3eb70997978d9457248f3f34e779651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 144, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ed70cfaf6195b7b7c087f1c02f01b386(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 16, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60cc14cde827bb1b8f27af003b649880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed70cfaf6195b7b7c087f1c02f01b386
    def get_inputs(self):
        return [
            paddle.uniform([3, 4, 16, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_55454c0c163305b9960f1f1317131771(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 16, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f69f0ec92a7d44ef5fc1a4f5756c49f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55454c0c163305b9960f1f1317131771
    def get_inputs(self):
        return [
            paddle.uniform([3, 4, 16, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5b8fe6821a08e3dc5e977b8f72a224e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 16, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cb21c8d66b020bb7aac43ba5059ce12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b8fe6821a08e3dc5e977b8f72a224e5
    def get_inputs(self):
        return [
            paddle.uniform([3, 4, 16, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2055eaf1f024599b992b75408562a347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 1024], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e42d391ca4f2c5fde221a01cf28d9758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 12, 12, 1024], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e89e5743d0d6a07d08079599c3958ac1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 32, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0215e73101f6abf9fb5e66f2f070547a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e89e5743d0d6a07d08079599c3958ac1
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 32, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_912f2929d058a9292d4384cac9234203(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 32, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2784db4528745bae74e75f82b8f41e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_912f2929d058a9292d4384cac9234203
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 32, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_aa2d788ec7ce9bd1b34eedfb8db5627c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 32, 144, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c841d2f6e5a931058fdd17f6a60777ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2d788ec7ce9bd1b34eedfb8db5627c
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 32, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2063549e3f15b19092a21143d51f7813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([196, 16, 24], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9c1cf499fd49225f0e0bf0e1c6d012e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 4, 16, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79741dce239f08bb1b64164778876d90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c1cf499fd49225f0e0bf0e1c6d012e7
    def get_inputs(self):
        return [
            paddle.uniform([2, 196, 4, 16, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7076df3344d15ee868eff63e307766f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 4, 16, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33e6189e21e37720fe1a4121a0e59182(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7076df3344d15ee868eff63e307766f6
    def get_inputs(self):
        return [
            paddle.uniform([2, 196, 4, 16, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dae9ddf7e78787cd7f330bd818890b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 197, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_43428a05f71c4dedbc842ade4d2c4872(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [197]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9dc0c95768ef5b455482ac314f950d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43428a05f71c4dedbc842ade4d2c4872
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_12f380a0004dd5e5daa8d0a355e3290e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 6, 197, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c5e3252760320f6c445ea4db9d60778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12f380a0004dd5e5daa8d0a355e3290e
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a69343c3d3e6b5eee57b10d59760a367(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 6, 197, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c663ab38bc9acb6ed343401d4572699d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a69343c3d3e6b5eee57b10d59760a367
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_202c1a8a602cc4a07e2ac2385d052533(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29b297c8d39bd11df4347664e2cf7476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202c1a8a602cc4a07e2ac2385d052533
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4521c6c8ac8c151e2fe8ed38a1f7a619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 512], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93908a8cdbf329c46d31258b5fd321c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0b572690b08b86faec53b8b6bca4cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 32, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_617854c79adafa6ca8d4ac496606b1e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3b4849f83b36762ccb25580b9c2d8f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80d5359f547888a7f4f63806896fc493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84ee53a7410f33380d6739ca4b454112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf19822150555ea3e4a2725534f67a5e
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 48, 128, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a054b7c3beecfd00e632d89f7523ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 512], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c617d2f66695723159795eb760b237a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac5acdecef9f2ef37c4df3ed6dfffbb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c617d2f66695723159795eb760b237a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0d167f8894039cfc30f56acdd1d74598(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f87ed030514ce01ccb18db40a3f4fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d167f8894039cfc30f56acdd1d74598
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7dd17e5938a5b597144e6849c7fea665(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d4daf43ed1a02b71660518038f2d56a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dd17e5938a5b597144e6849c7fea665
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3110a969fa56e3d33b79e513c3d91cc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1479e89401d720ef4dd0cdcec88a6bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3110a969fa56e3d33b79e513c3d91cc0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_39eeb3cf4e8b8ff35d3db7d5db737156(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82b5afc779f0b8c4a543eee9acf7a056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39eeb3cf4e8b8ff35d3db7d5db737156
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9470b10cdb699529cd9778bc24f69d31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c6d67ea8a2e610dfcd43b7f3a6828dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9470b10cdb699529cd9778bc24f69d31
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e523e69fd5a166cb38c106e3f22e011d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b051c642448e857346c19027d9e39f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e523e69fd5a166cb38c106e3f22e011d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6e75b2fa17e25f1030975e8b96ce1e52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_775c5076031e2ec9d7e30d5b54bf78b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e75b2fa17e25f1030975e8b96ce1e52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_b6c6e8f4f605b422316aae9cc82e8517(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22f6df3ee1a81270d1dd31d7bd91a448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6c6e8f4f605b422316aae9cc82e8517
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_cea4cf9f5acd2676a0f04d8730826e87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60fe14c2a1cb3dda5d8634696f2feb44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cea4cf9f5acd2676a0f04d8730826e87
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7241834747d14cd718a3c452bcf67d59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a044523170b10a471da61a55a3313e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7241834747d14cd718a3c452bcf67d59
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0e46cfc68e45794d6bae585808aa806d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 85], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6397c36470ed250c37aa9dac4f0c9fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e46cfc68e45794d6bae585808aa806d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6c9e13a368e186c166244ae60a5c456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df8ddfd8eb08179726a22c7baa5a95af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cfc347f83b3fbe92df420a441afe1a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99637a33ce305056b3a7970cef31d9f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 32, 24], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e76f2e4c711891c7ca64760ff0a7f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 32, 24], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_020c8fe39a18274a4eccd4deab9666c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 16, 12], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94551617f02916c444f5865949f3e95a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 8, 6], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0e847f3eae903129b21bc6077163bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 320, 4, 3], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a6c1b533b0108fa2f17cbd55a6c5fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 17, 32, 24], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27bf599f825b98fb1499c7507e0f536b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 26], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_27edc23a2f06bbee947f0dae7b6b58b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db563889cfbec70d2a2860955f16c282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27edc23a2f06bbee947f0dae7b6b58b0
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2, 350, 25, 1], dtype='int32').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3feee1c5badb2e56b57f7c63a81cba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 350, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5254c56767d97b808d39c5f83bab385b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 350, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0a825c5d4150d376ca75ef5b9653dba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 175, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc138717651fd9d20b6e389f890995fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 175, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02c2aa6920cb28f374a58460f74155ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 88, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cf0caec15cb78937bd8c93be8cec7c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 30, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf5b452a5ef91501ae2de6bcd1a8cd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 480], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45483462e5a4264653cd138fa1dcf1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65f78b1dd1c09789f1154dcfabe34c56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 18, 112, 112], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c6141706ac72647cc5770eac798124e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 197, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82278f1f8688f01532930c39e01428c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6141706ac72647cc5770eac798124e4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_04d17ea74d0309764629d714b84a7b1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 197, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84f50a6e1ef56141f50005c485a15139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04d17ea74d0309764629d714b84a7b1b
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_95c229e82c2786443e02e6ba73b329bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 197, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3831ca4e14b6e31041aeb30ffe69e6dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95c229e82c2786443e02e6ba73b329bd
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f2e5991bf4b92d189dd8fbfe091f2b02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [197]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02bcd8b2cbbc34ba538b0f39ab6729eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2e5991bf4b92d189dd8fbfe091f2b02
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8fda967d63377b7e34a5be6b45d5e0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 8, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6947a1cf7ccfab7d910d3b1009c4317d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c2c5da34d8aeff2136deb4114912648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6947a1cf7ccfab7d910d3b1009c4317d
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 8, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac2c115107a2c857bb035d0c19cad422(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44da07ddb58f97b73be00e5bcac7f5b6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 8, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_584585636b9e1be6d6b02af055e822cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1bf3169064e27cdedce1d18bf03a4c8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 8, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6024d713a4681b8f84522395d3a87eb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7bc97c99fcf68e4786a1c49299aae68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([25, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_77002ab49868e4b374fd3ad1c09117bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [25]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5000, 1, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0865b119e79d680b1a2d3cc38b9d6ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77002ab49868e4b374fd3ad1c09117bd
    def get_inputs(self):
        return [
            paddle.uniform([5000, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(25, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_522bdd0a93da07325342705ee2cac039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5d9652e526d35ba498ede0819662bcda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75a1641ce0350affc96c50c3e7b03a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d9652e526d35ba498ede0819662bcda
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_21f532830bca09f5d50d829be31300bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ccd2fb11add8cff7c769cf50ec94fd87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21f532830bca09f5d50d829be31300bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_cbe00447cb14adcf6fa0d62eeea5bdba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9beeac972973eeeacc6bf8437450ee7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbe00447cb14adcf6fa0d62eeea5bdba
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_059151094a8e46245fee4459b0f2887a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3302f65085a310db70b2e9f34534f77b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5000, 1, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89f564eb93a0fed26c445b7eef51010f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3302f65085a310db70b2e9f34534f77b
    def get_inputs(self):
        return [
            paddle.uniform([5000, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_834f3456e78a7de2acf9f1f7f137198d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ab0f2e01555d5724c2cbfaf3d73edfd
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb052f873538b71c1b7fe52ccad46e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d9652e526d35ba498ede0819662bcda
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 1, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_386613da3b8cf43e4f31989823a5809e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21f532830bca09f5d50d829be31300bb
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 1, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32ba9a298404b5705ab30cab449d660f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbe00447cb14adcf6fa0d62eeea5bdba
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 1, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ee4ef5d16a6e726c8d7fb20fc67a7dbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_865061874dc97e91e6761d938e2321af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee4ef5d16a6e726c8d7fb20fc67a7dbd
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5a73c2e7374180d730fd4660858b3f3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1d2ff6a683dafd2c0cfa3a1715c7b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a73c2e7374180d730fd4660858b3f3f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a74ade86876589bbe480c3feee0f6075(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_733f5a880bdbba685f1e34795bf3eed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a74ade86876589bbe480c3feee0f6075
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb01b7feac15194d3d45b9b9ddd75628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 240, 240], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f35dfdbf80f86a46f10a4c8274650386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44da07ddb58f97b73be00e5bcac7f5b6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 240, 240], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a6d6da9a950cfbc1d47bdd45e646aaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1bf3169064e27cdedce1d18bf03a4c8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 240, 240], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_661ffb7c6d34997010aa4efbe3326e26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1536, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5fb0637a86c27be6238345678dc4ebe7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd04cbb38b96a5b54979329281dce39f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb0637a86c27be6238345678dc4ebe7
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_fb1ca0f8beec1919dcc22b0d4381c1b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 4, 16, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e397348057618cda8027a96370bb62a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb1ca0f8beec1919dcc22b0d4381c1b1
    def get_inputs(self):
        return [
            paddle.uniform([2, 196, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4e4c4e8555f951021e71714411323bcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 4, 16, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09fe6cd0d1f11a4f99d89578dd8099d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e4c4e8555f951021e71714411323bcf
    def get_inputs(self):
        return [
            paddle.uniform([2, 196, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8e8ad86f346ed68fb22c80aaacee741b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [197]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2898e942e0acc77e131f69a74dd07b0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e8ad86f346ed68fb22c80aaacee741b
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_83af44c9c45a333bf7904754a71a7044(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 6, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_12658ccebaeb77c3f42ba4b6b0274bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83af44c9c45a333bf7904754a71a7044
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_708950413486227e3c7ae8c3008cf5f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 6, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c72e4f9634c9ab73a20207680bbc43e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_708950413486227e3c7ae8c3008cf5f4
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_602dce5aea5190370b8902404d2e4f3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57211d77d1cc937d91e7166fcdfc40ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_602dce5aea5190370b8902404d2e4f3e
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3c712e7dd60a630f8e93e0eaca20487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 56, 56], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffe8ab8ce693a03c627a01b43343a56a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c6715de28b0b87cd10a2bcb480211da3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_692ac8fbb17c0dcdfef30a4e6307e258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6715de28b0b87cd10a2bcb480211da3
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1ec7249192d78219b6f183d0cc8c223b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb12f42797b0ceee690096d75c0a89ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ec7249192d78219b6f183d0cc8c223b
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2f100cbe6c48636ba77396b788470da7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1375caedcd4cd5301ef2d019c00d5b3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f100cbe6c48636ba77396b788470da7
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2fbd24dad8fdb252d3695fb13e512c0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acaf17e3bb466f9fa346b3d154496435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fbd24dad8fdb252d3695fb13e512c0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_83fa9484a5dfdf1d51a2d8cb2845cd03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [27]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a75dae69354c4b464cbf13d8c52313b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83fa9484a5dfdf1d51a2d8cb2845cd03
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9579fe457484aa72295b69abe511181f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [50]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 61, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_695300aeeac8e861f8c41eee170cc58f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9579fe457484aa72295b69abe511181f
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_17e6fc4273366a47fd7f97301d0e449f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [72]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ecc11be7da4aa70ffce2ebe9538eb4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17e6fc4273366a47fd7f97301d0e449f
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ea02556fb412f7ad4bfa196b7f4f4a92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [84]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aa76e4fb40cc76dff338cb9720531fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea02556fb412f7ad4bfa196b7f4f4a92
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8b10187343cf7ac4bee7e04b128f9856(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [95]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61e7e6cee8bb41ffea618c05aef759b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b10187343cf7ac4bee7e04b128f9856
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9924b36caacd878a3dc9041399cf6def(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [106]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26dc42c771e35b69f494e25b7f2117c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9924b36caacd878a3dc9041399cf6def
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e52736a188b1bab2b1e53adb82c98205(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [117]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_184aa79a221cdba73d1f45a610d4104e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e52736a188b1bab2b1e53adb82c98205
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1402273de54aef25a3b860eed039e116(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [140]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81dbd008d95f17cf920ab5c7288f99c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1402273de54aef25a3b860eed039e116
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_89719b958e5a7b2a715cc3b7d143653b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [151]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c26d7c298abfe34d8824701fc59f628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89719b958e5a7b2a715cc3b7d143653b
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_70283e2b9fa73d2502051b9c4074fb5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [162]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38f51e9f3cc735b4b44ff9927d1ba6c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70283e2b9fa73d2502051b9c4074fb5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_84f27238e53ea54e1fc161ebf5f4621d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [174]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 185, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0d46644d1997306c24f8746657478b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84f27238e53ea54e1fc161ebf5f4621d
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49aa5e2192a58dae767e94cd540e5314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 224, 398], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f1b7d49d7f5973162e0821b5810247b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44da07ddb58f97b73be00e5bcac7f5b6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_484d15a71db4cf5db0367dd59fb30dfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1bf3169064e27cdedce1d18bf03a4c8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49677a1a207df24a59fae1d35dce47c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44da07ddb58f97b73be00e5bcac7f5b6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd4e9dbe20d448528c28815707fcddc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1bf3169064e27cdedce1d18bf03a4c8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b63673200d697bb7739e3472efeeaad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 18, 112, 199], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6f5b3c60dc71b9d9dd736d9d0ba2377b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [113, 113]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 113, 113], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25f20f2c4a839dc62826b810e79cd0fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f5b3c60dc71b9d9dd736d9d0ba2377b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 113, 113], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_bc45cf9315d90e030d44e2496dcd45ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [57, 57]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 57, 57], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1357bcffeb4c030b7e693526889a065f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc45cf9315d90e030d44e2496dcd45ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 57, 57], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_364d7c8b4e5f53614d0d1b430760c901(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [29, 29]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 29, 29], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_291fbb956d89239abd6aa81f72b4d984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364d7c8b4e5f53614d0d1b430760c901
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 29, 29], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_96246e485e85edb43029360878087464(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [15, 15]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 15, 15], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f1766d82348e0aafc29e67eb17827540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96246e485e85edb43029360878087464
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 15, 15], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a3bd45f7a592dec71a763cfb1fb5e61e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [8, 8]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 8, 8], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_560008f1dd600a3051309be7b4c0d93e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3bd45f7a592dec71a763cfb1fb5e61e
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 8, 8], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e54255553c83f43eb28fdd1e656b9581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 384, 384], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c24abbb3f962d6aef548f419575d0136(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 577, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9008af76d00ee171e9133f41005b23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c24abbb3f962d6aef548f419575d0136
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6e52fecb6f76e9161c9183e0941736cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 577, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_797b1a63eb45c51aebdb5d58576003ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e52fecb6f76e9161c9183e0941736cc
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_fc316732e03e44ab4f58d351bb7ed675(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 577, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3530f7d9673c7eec161777bb5e80625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc316732e03e44ab4f58d351bb7ed675
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_757eab2116eea2b1a4c3ed242d457396(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 577, 768], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b80ad1df7bbaa692de2d0242922dec39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_757eab2116eea2b1a4c3ed242d457396
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd00fb257489b32bf44ae1b4e60de695(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 24, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1cff56ae5ffd7d9fefec2bcf6bcdbf00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 48, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa41328439df2486127757c48194aff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_588044a02a567dd2feb9e4180ba2ee4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fda5503f5e92f97c2bd8311784f64b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 512], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_906548070e7da17a6817b7c528d571fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 2048], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67c7b4abc83f336b34f9a35f8eed580b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 48], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_144279a9c3b28a8ce440b10f04543918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 48, 56, 56], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cf835b21501a0982f0c5bccedc649f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([56, 56, 48], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3791127d6c9a5b850d71425023d109e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22715159e28985eadf263cbcb4859bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 96], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94aa5cd2812f1abe3543e7280cea2a81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9418b320fc1343fd516102c28000c280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 56, 96], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4760b50db83dfb70b0c77b9a1d85870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61b5c1b7e8cea1c2007ff4c7a51a0d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8d57412715391f005535d9351625a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f922df4ae4fa9858b965f516825ea95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 98, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd72f66a27d7d6a9e2fdea078d54fda8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a264984a3120c0b839620e7ca3de8c75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 64, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5e2dbad66c619daabcf8a97c9886670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a264984a3120c0b839620e7ca3de8c75
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6be4321c0da968eafae8127ac049e9e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 64, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_497f6a036abf68e0bbe3f04c0b73f289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6be4321c0da968eafae8127ac049e9e1
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_674e4c20dfe98f442762650a8abf3e9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 64, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cfceac9396e1982d94b129e25fe7f132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_674e4c20dfe98f442762650a8abf3e9f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d54231824429e963ffc00ce2b4b3ad35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0bcba616f5d9e690a06592e1d8b3d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d54231824429e963ffc00ce2b4b3ad35
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a45559b4103216a77445eee3c5be7acd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 3, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f51ad330628d35f4f7a016107f8de4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a45559b4103216a77445eee3c5be7acd
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_71b7de61c85f8f570c79357b811c2221(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 16, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e8cd60d4282e1ddefba30e2256acacf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71b7de61c85f8f570c79357b811c2221
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7240ce4375e3522e197a4864b2de4352(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 16, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6eb6068bc314bdf3b91dd1ee5f413eed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7240ce4375e3522e197a4864b2de4352
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6c0949d9eeaaba077bcfc6d2abeb3435(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 16, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ba7fce919d1d37b259c43a6d010fd1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c0949d9eeaaba077bcfc6d2abeb3435
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_630344bf8cddc14f9ab023cb9e6cea5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b38e4f7e0e41955b367d175021d1608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_630344bf8cddc14f9ab023cb9e6cea5a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d09f8785bc04daf714e1fafd518b1a0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 6, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_576f501a2d4a473890048edd636c73d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09f8785bc04daf714e1fafd518b1a0a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_30a53275fecccbc502a938efbcf4c18c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e08287d1d9b99fad92f7edf38e4f7110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30a53275fecccbc502a938efbcf4c18c
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_fb78840c786ac202bff875e93d4e4af7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90acda0d19a7a95f6d71f267e598d4a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb78840c786ac202bff875e93d4e4af7
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c0f442cf8f9f48c29b76100c59b3c79e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 4, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_989cb00a817d6edf09bd2e6610c8a2d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0f442cf8f9f48c29b76100c59b3c79e
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9afe3c82d7eec75dd99fcdf01419ffe3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ec49cf99a3d82dc5b6c3b4c9f89640d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9afe3c82d7eec75dd99fcdf01419ffe3
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d10496279d809f7a88aaeb1226cbb879(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 12, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee3776a6009c25b537aec248a64ea840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d10496279d809f7a88aaeb1226cbb879
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0485330ecfaff5d416e74907325eafa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 1, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6bafc3d79759ce2cba7a5e4c9604d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0485330ecfaff5d416e74907325eafa7
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ca246185efdfe7bbfe4ec936d766bbd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 1, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2208dd82452fd669340992ef2ebe17a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca246185efdfe7bbfe4ec936d766bbd0
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_df182f30dc21ea6930ff85c475dc3445(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 1, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67a17683e54d64d7d8d962cba314c238(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df182f30dc21ea6930ff85c475dc3445
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8be048ddb39634e01c129c5e76585910(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e5b91c931388d50715dfaa35ab4ecd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8be048ddb39634e01c129c5e76585910
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6501615a5459dc1a1e6529722d9bd326(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 24, 49, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48f7299b8d13e350e0107121eee6534f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6501615a5459dc1a1e6529722d9bd326
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2a3a6ba0ccd775029a536c23a2d583c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [13, 0]
        input_2 = [14, 1]
        return paddle._C_ops.slice(input_0, [1, 2], input_1, input_2, [1, 1], [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 54, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5055b2cf3953f3011bbb879a23236b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a3a6ba0ccd775029a536c23a2d583c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ce665bd824f2ff46a292d11a7d34df08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [13]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 54, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2aa09a66d605fc0b7daa74a445f261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce665bd824f2ff46a292d11a7d34df08
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_261a9ca3b1ac4d0743309ec32ab90f56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [14]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 54, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1afaa0e80d4b9dc25e584a6b76e0a6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_261a9ca3b1ac4d0743309ec32ab90f56
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbd0614b4bedf47c1c7e65004090c8d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 31, 31], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d38b560778fc3ee50d30e08f195f50c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 61, 61], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43d77de83df3c53f7f9425cdd30802d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 122, 122], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f07ef09994d02ca87bde143c3d2558a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 6, 40], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_335eb035dfc69f5602cc9aca0e87c02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 6, 40], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_001ddb00549a57b3e3e633c70f288026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44da07ddb58f97b73be00e5bcac7f5b6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 100, 168], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f72f9b15311ea3e670372beed07425e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1bf3169064e27cdedce1d18bf03a4c8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 100, 168], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de2fc3d1971a2e4fc946af035598ea99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 128, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1ac4ccaeedfa58093c4851874a9393d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 577, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac62b0577fd6460d211d099408b20b4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ac4ccaeedfa58093c4851874a9393d5
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5ab24ec4822d54a34a361daf287611d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 577, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2392d2e7d6405e70e525623c6a9a455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ab24ec4822d54a34a361daf287611d3
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e2e1e19238eb12acc9bef50e4053a24b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 12, 577, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_316b847a4162e4c8df63081d5fdc4072(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2e1e19238eb12acc9bef50e4053a24b
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9646cd16a9caebaa27994c005cf55f7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 577, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b62cf6b132f8e53a32d6ced947d9072c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9646cd16a9caebaa27994c005cf55f7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7366d93efe25ea3a490ff7ba90ae15a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [113, 113]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 113, 113], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_565b20385bdb048f66b9c44c49c31818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7366d93efe25ea3a490ff7ba90ae15a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 113, 113], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_b8ae1a449376876bdd451338e58e3ad1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [57, 57]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 57, 57], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df5eadc2a34e8c5a31f2ac6f8469f25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8ae1a449376876bdd451338e58e3ad1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 57, 57], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5cf3117d36a3d1712fa5bd34b00194fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [29, 29]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 29, 29], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dcc1c82191551e3043b7d42f7f0d677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cf3117d36a3d1712fa5bd34b00194fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 29, 29], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a9beb3158e0f9e902cae6dd4d1a9eb9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [15, 15]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 15, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3056041eb99ca1dab28ef5cc84a8537f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9beb3158e0f9e902cae6dd4d1a9eb9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 15, 15], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_54e1893e0993d216ed9aacd841c99dca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [8, 8]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f447ecd371a608f92858ce5563ded4fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54e1893e0993d216ed9aacd841c99dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5af8529c1b34df57018201a0dc2087ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 256, 36], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34f3c02b955bf21fb2c7a21555a1f3d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5af8529c1b34df57018201a0dc2087ef
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_53503584aefaf1223ad2ed3a66da9d48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 256, 36], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eeb3193d312b04601e70cb152cb4e145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53503584aefaf1223ad2ed3a66da9d48
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_142fd978cb964e4db39bd4ae9ce31d90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 256, 36], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_464938cf52941d28fca09b359c35461e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_142fd978cb964e4db39bd4ae9ce31d90
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_fb3ce031b56feec17bc319f140be08ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 64, 48], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d91f0091f5a0fda6f96a381c95fbfb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb3ce031b56feec17bc319f140be08ad
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_704390eba243232351d9cb10bbb8a882(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 64, 48], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82bdcea93caf300e5c0848ffec25b7ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_704390eba243232351d9cb10bbb8a882
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_83958a1404e544883d722aa567952e29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 64, 48], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9929cf36b085a711dfb7c401a1ac3105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83958a1404e544883d722aa567952e29
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_89d6b483b01f03d4a6801a148e801f01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 16, 60], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8a42452c41679a3a62911e779110e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d6b483b01f03d4a6801a148e801f01
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_09c38c739d909ccc12146c2916924607(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 16, 60], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d6c91ea1599b15c9e4f0532334f2409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09c38c739d909ccc12146c2916924607
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_cac5097b1b149e40f2a6e61040825d77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 3, 16, 60], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_143cc4dd212eb14fe7449ac8a43b5d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cac5097b1b149e40f2a6e61040825d77
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e2a3695da0f0e346b3703074567fa73b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 5, 196, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64a6a3a45a0b387a5ed03e37bf3f8285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a3695da0f0e346b3703074567fa73b
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0c7152628321e5eddd06244227908e34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 5, 196, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2c75afff65b5cf7eb3c575a2433afec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c7152628321e5eddd06244227908e34
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4ae8d31420aef1aea7a74a16cf915fd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 5, 196, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a053d41810f6c60f8f5417a5cceb42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae8d31420aef1aea7a74a16cf915fd4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1fbfaef11766dfa2b865a3dbd4ea2431(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_062e2f9116ca2560b1b1a32aac86b192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fbfaef11766dfa2b865a3dbd4ea2431
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_fe705986b13c647d6e4cb857778982cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bce1080c22a5d86175dd32d94ca5cbf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe705986b13c647d6e4cb857778982cf
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2e7bb89c5a0f992b2dc50660adfca40f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09e0f09b376a2a4a0b64fc686c721dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e7bb89c5a0f992b2dc50660adfca40f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fc4cd5ef2084fccaa0723fe4f961102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44da07ddb58f97b73be00e5bcac7f5b6
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abc4c08e12957988480e8c9767b134c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1bf3169064e27cdedce1d18bf03a4c8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d94a9bb59c4dfc8631ffa8bac9420ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ffa27da7350407f0b707bc58fd0bb79
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b82b9d2cb14c8bc5c9187c3fba5e1b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([256, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_87ac7c1d69227a0a985c85115c267427(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [256]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 1, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_701ccdfd39f7fd530197cc9a571ed622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87ac7c1d69227a0a985c85115c267427
    def get_inputs(self):
        return [
            paddle.uniform([256, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e318c061e9c601b0bccbf50d83348e3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 256, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f85e0e90d0a741bc722dfb4b861d3e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e318c061e9c601b0bccbf50d83348e3e
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3dacc890d7c5161c6ddd2040588ad650(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 256, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d644e8332a5966c1ddb3d4a7622e8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dacc890d7c5161c6ddd2040588ad650
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_b7cfaecf7cb60d800117eca22ed61dc5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 256, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a34ebe03a17cd5f5e3d870003e5a389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7cfaecf7cb60d800117eca22ed61dc5
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a9902a3c23063d72b72b32142204c11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 8, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_378a51351cc1a5ea2d5d8ae75b1b0aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([26, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4c20e9d90b024faaf8f3ed771c283e4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [26]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[26, 1, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0142e538368f0bc9057ab339cbb813e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c20e9d90b024faaf8f3ed771c283e4d
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(26, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f3b14fb8a5614a63683314b5c37ad51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775fb8a249b0606af5a28c3a049a9bc4
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_346aeb5d110aecf995d0e4c46c1460e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 26, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c68129460e561f814c52ec62e64c3f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_346aeb5d110aecf995d0e4c46c1460e4
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 26, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f1b447feb7bc8f6ce78cab00d974991a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 26, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9cf8dd46f738805748b89d4a9e96b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b447feb7bc8f6ce78cab00d974991a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 26, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_661f6c2faab9880279b87bfc0ec1daf4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [25]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5000, 1, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f35d665624eef288e1b6ece84c921a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_661f6c2faab9880279b87bfc0ec1daf4
    def get_inputs(self):
        return [
            paddle.uniform([5000, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(25, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_416af448dc4e70bacfaf21a321ae7032(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a05f70ceb03ebe16c4e7607f9562bbb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_416af448dc4e70bacfaf21a321ae7032
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_86617f57705b9b503bccb52130840f4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fe9794d15671773da913f2ccf261eb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86617f57705b9b503bccb52130840f4a
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_af757126ac671bd6204274b628350c4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, None, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c0e4a2e574ca935c81047ae92b87e5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af757126ac671bd6204274b628350c4a
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1377ca6d7f67ba33b553f44176ebe6a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5000, 1, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73f391d0d191f0283b25c8d7c9b35c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1377ca6d7f67ba33b553f44176ebe6a1
    def get_inputs(self):
        return [
            paddle.uniform([5000, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1c7e283382a8ba9a1995c8c57f23ec07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [197]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c19d0e59a70994fc761bf6dae651f882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7e283382a8ba9a1995c8c57f23ec07
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6e5e2c05813d320147f01c08a72aff92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [27]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5884ede6e92ed86f8e408c572d07ac8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e5e2c05813d320147f01c08a72aff92
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_44eee085a51291299001cea8b3c7744d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [50]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 61, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e58df79e08d4e654bc487df9cb071d9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44eee085a51291299001cea8b3c7744d
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1808ac9f77dc488e10d49a2ec9b54dac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [72]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e104f5b82d80f62c0c3b8cab05a09215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1808ac9f77dc488e10d49a2ec9b54dac
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_b4b39183962479d147e03e2fb47b2e52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [84]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53ba6fe545cc75f223c54970637e91b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4b39183962479d147e03e2fb47b2e52
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f8b366c9ebee59445f38bcff52a31d54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [95]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9623a7ae3a6b68f60d004ff6b1ea9b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8b366c9ebee59445f38bcff52a31d54
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7fbab4db8811931fe2860af4f87a9b15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [106]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d939151cc74995d0778e139314fcbcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fbab4db8811931fe2860af4f87a9b15
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2bfb2d122952252d63c646bb8b43b2fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [117]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_432c0a01a5b0c4e1da98e29285ca6ce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bfb2d122952252d63c646bb8b43b2fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3aa241922e8b93a18220b520b0cd2d2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [140]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77ecba3c94426125fc908abd034ae11e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3aa241922e8b93a18220b520b0cd2d2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2726132ac367c321d40ac1a6b4ae0544(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [151]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a24905bd8c81ed884b63fed1ae0a4b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2726132ac367c321d40ac1a6b4ae0544
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_089f178c18cb61455e69e812af30173c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [162]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51ea1da7a4ba839f5d621fb483d478eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089f178c18cb61455e69e812af30173c
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_61db19d402b0498f5af701a0752e6c36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [174]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 185, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffb7f45904534dbb5301ab764a6fbfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61db19d402b0498f5af701a0752e6c36
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e9d19912ae5b490ba71b30539dfdcc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 64], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_232156506d2e13e37ec0ca80764b9378(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 1, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3beeb2ca11d450bfa8136c76371b0c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_232156506d2e13e37ec0ca80764b9378
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6f48abd52fe909b58c4c7795e679c664(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 1, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa4a1fcb4d8b3340b11e5c23880b1088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f48abd52fe909b58c4c7795e679c664
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3bdb395b0bb0fbfe91782716812e9d8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8cd58df6f73063428343dce78587aaa
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0f7167186c16297a927c06934c2cc22a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 2, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69634d72d236158cad59672b2d3cbdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f7167186c16297a927c06934c2cc22a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8c11c165edfe96f033aa9bdc7904898f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 2, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43e3a158a745071109c5eda488a3a2ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c11c165edfe96f033aa9bdc7904898f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f024a10d635a9eec37e1c511c84677e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 5, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e227d74047a3e3813682a647331cec6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f024a10d635a9eec37e1c511c84677e6
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_de4eaf0d31b6bfec2c7514c5bc95e888(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 5, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1f25062c7f195b33462d5ff804e84ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de4eaf0d31b6bfec2c7514c5bc95e888
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_fdfe3546d490bad778da55c2ea551a6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae30136a02ca6a871d547d87c36e2526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdfe3546d490bad778da55c2ea551a6a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9700659449c07c954767b6a8f785f97e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 49, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_599aaf32898313debbb913dad3b4c5d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9700659449c07c954767b6a8f785f97e
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d164a80fd305a70b15e90ac1e31959b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 1, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a567dd361ea45fee860acecf15e14b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d164a80fd305a70b15e90ac1e31959b1
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4c85ae8b9069a2c528f13beb564dcf5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 1, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7a9148f90324cc149d17489017ea1f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c85ae8b9069a2c528f13beb564dcf5f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3494ae2aa67d1f0b57299b2229d15012(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 2, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bef1e8c31bbd0720f2dbd437318db24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3494ae2aa67d1f0b57299b2229d15012
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7d17f1b1e4c139cdabdc21d4ac548d0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 2, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f8a66d3ff73f1b2957ac4365da6b637f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d17f1b1e4c139cdabdc21d4ac548d0d
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_39cf00e246c4542387fc620b9758412a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 5, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_12178a39597f59f15edcdd5f02294131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39cf00e246c4542387fc620b9758412a
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_396ce3338352b476bebb1dd3aaaf487c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 5, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42eb97db480f0b976842be56ac96a1e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396ce3338352b476bebb1dd3aaaf487c
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_71a33c1ed12ea93ef7347a274cfe0d56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c54877a0f9e2ecb6583620bd739f2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71a33c1ed12ea93ef7347a274cfe0d56
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0e0968ec230f947423b650772beaf331(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 49, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_294e51bf2162cdb074ace0873e1950f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e0968ec230f947423b650772beaf331
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_353b82d686f3ba3a11c9079d1eeaeb41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4, 112, 112], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a9f5f7e90c9db868febe18b769a767a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 56, 56], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40223d49c18738373d15343c3811fde0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8, 56, 56], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bd80dc209644b049762af39bf07f23b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 12, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc9113f52d0e5a01ce15e3e3e1a55ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b261d2461644563c744af21fa697ab83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56739a06f33418a113dfc44328d0260d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34c0b3287f5029998a8b7440bde980c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c76fef8cf757b6237714263f62d15be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_924494aa1708aa5ee7f4a83c8c97dde0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4022b54fe97459f84349526024dc1a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac44d5395eb3c05295648d81c449004e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3778b80414717dd0cc5e8fce4e62147
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_346fd51076df3a174c441f1bc563b92e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 32, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_389f3b8070b8f7e54899b48e40a5fb17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_346fd51076df3a174c441f1bc563b92e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_22e0a4da39410730db2a9a4415df9f23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 32, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ff0a2e4b084c6a980d70bc5d4f74d61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e0a4da39410730db2a9a4415df9f23
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1e5f1b67272e0e1411fecf6d071574bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [-1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 3, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6db0615c638006094b58017f5a0edec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5f1b67272e0e1411fecf6d071574bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d9035f737345f8a3f4cfa3869cce68c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 3, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b1a236c5dc4fcd8c0ae24b6b4a331f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9035f737345f8a3f4cfa3869cce68c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a4cb2da41a5a0f7a28ddcb24fe85865b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80, 2], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4108ac8e5721c5898152481228849b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4cb2da41a5a0f7a28ddcb24fe85865b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_b1a881ded8d4ea24c1c8dd2ab91626d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80, 2], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6028ff556c1ed2a31d52c06ecd9ff10c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1a881ded8d4ea24c1c8dd2ab91626d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_826a7b8148d1740123944771a087dde2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160, 2], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a54de13ca9653cf2b612a388815a5625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826a7b8148d1740123944771a087dde2
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_652bb7f03515b8cc2df9bb189ee9596f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160, 2], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a3586673c9423cfbeca016e7b41645a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_652bb7f03515b8cc2df9bb189ee9596f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f6a805c13802993c330378864a7f3b09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320, 2], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d8f1e6be7899964f1316e5214c6386b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6a805c13802993c330378864a7f3b09
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3a3dd37182f1eb7b6da0769b6565765e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320, 2], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96f78d96981e24a13703902fc69e6e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a3dd37182f1eb7b6da0769b6565765e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1dbadffb450c4d79d367ebc4d4b7ec2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d80dfadf17a1f2f7258946e4a395b51f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1dbadffb450c4d79d367ebc4d4b7ec2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_367a95b91f7ab3f08c0d617afe456a31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e831e3e0f7c5603cf8dde2a5bf868640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_367a95b91f7ab3f08c0d617afe456a31
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c44f671a8c79c0aa811fc8c1bb1cda87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2, 180, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9061c4627c5222604c2b72896f94c4f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c44f671a8c79c0aa811fc8c1bb1cda87
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_33a878478ed03ebffaac411513e08fda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320, 2], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_985ae1d0430286b3960da3dc12dae26b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33a878478ed03ebffaac411513e08fda
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_98669e3cb7332c1bfddb49d4d04406f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320, 2], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c2aa835601d09a765224aee333e76ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98669e3cb7332c1bfddb49d4d04406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d04aa69492846e8bac515ae57a9e8f5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d8f21995e17901188093a2252011ca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d04aa69492846e8bac515ae57a9e8f5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_799db2202dc6bd2d4d4def5b9d50a9f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 3, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ebfd26f42594ee3c23ac9cb8a7887b3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_799db2202dc6bd2d4d4def5b9d50a9f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ba496014f65baf6cf9d58e34c5928687(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 3, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9eac22ffef7fa289ccb71dd7a31f0f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba496014f65baf6cf9d58e34c5928687
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a50ed8203e0bea9493e11771bc1f3451(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [256]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 1, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70d5164d1f23b88291e798a6440d4267(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a50ed8203e0bea9493e11771bc1f3451
    def get_inputs(self):
        return [
            paddle.uniform([256, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_b637d55c7b407c449543862f08b1ccf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 256, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c839645d34b1b44c6b4e88bcf9fbc3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b637d55c7b407c449543862f08b1ccf6
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_73449d5962ce4d7ecdf7cdcdcf566521(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 256, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a24193aa91cc2a35152de25255d91f11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73449d5962ce4d7ecdf7cdcdcf566521
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ae0ce2c66caba8b446d05f1a4ac9b873(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, None, 8, 256, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f50ad8df16696c7950d03803c61a03a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae0ce2c66caba8b446d05f1a4ac9b873
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_05a8e636714a78227f3944e2005cdcd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [26]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[26, 1, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bec3b9ae1fbe908eb01e2230f8adcdbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05a8e636714a78227f3944e2005cdcd5
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(26, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_841e23620e6336b95bd18ad1c1ce7967(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 26, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94c424311e421ab3bdb3fc5cc8d09afd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_841e23620e6336b95bd18ad1c1ce7967
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 26, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_b409baa720fbaf6f68c093a6f9f66e94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, None, 8, 26, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74a7d7b6dcfe90b383d870ace4feef06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b409baa720fbaf6f68c093a6f9f66e94
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 26, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c2237a5a39a7df6d778e3fba83e06a97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4861631c1fb2a14188684e1af739fa71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2237a5a39a7df6d778e3fba83e06a97
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c5e7905dae050c2e2c3ff7b93128a68a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af2607213b2c9b0ddad167eabcab7040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5e7905dae050c2e2c3ff7b93128a68a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_75302a56f3b2563e30f14212153a2c27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df92fb0511678746d2858367a5851c9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75302a56f3b2563e30f14212153a2c27
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5f7f3452c088c851f59b8b25496c1b14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c8f2d950743d178574ac4ea9b447b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f7f3452c088c851f59b8b25496c1b14
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a56787db2d3e5e70402f9786b5d59d8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c3894cdd80f64115ffbbd5af8881532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a56787db2d3e5e70402f9786b5d59d8c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2b82c96363091e99163f95c3a9f903ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d75a2eea1ce6011c0e30614ce8bbd8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b82c96363091e99163f95c3a9f903ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ad7284e54e3de4a8c59a4c5dfa833e05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63b98b3af48c376a628aa9dd47598833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad7284e54e3de4a8c59a4c5dfa833e05
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1358de6590a98e8bc6a9d0a601fe865e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cfd2b00f4d503f6603e88cb65bdd06d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1358de6590a98e8bc6a9d0a601fe865e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5165519a2971fe656e14e27f4c12f535(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e4d32093edc650ca077e302c2ee162d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5165519a2971fe656e14e27f4c12f535
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0bd76e1f32af468b4116c43d8f14d888(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f98af65535ab1b23abac6ab4c03f99a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bd76e1f32af468b4116c43d8f14d888
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9a7869e56681fd88a1b4c24e7bd38c8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3b30d6b56bb24df8fba6e866bd09439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a7869e56681fd88a1b4c24e7bd38c8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_72d35867faaf40db259e437994e0bdc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 85], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c5522339d3ce0b41d58ef41ab23f930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72d35867faaf40db259e437994e0bdc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3d65d7fd91d999b80bbe03a7e458d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 400], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54dc1c9a0ef52efc232edf2625c881a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf6a0e78f024297ae99c90ffbfbb8a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 56, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14057405459d090236eae33e13257e5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 120, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14250da2887ef5116c1381d436e401c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf0e79882de6d412a0a473d496bc22e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c45ffbdf3ef02a883fba410a31d4b12b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36e2f2df3076dfa033164aca1491c893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1024, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df9277efc844087b263e45abdcd446b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 32, 100], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f44250e8329495015f71bb31102486ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [13, 0]
        input_2 = [14, 1]
        return paddle._C_ops.slice(input_0, [1, 2], input_1, input_2, [1, 1], [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36ada0ef48d9ae39ea368705a5e1c65f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f44250e8329495015f71bb31102486ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8617b75212223083dc66d17eda1b94ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [13]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ed3c182f8df0021c97117aebf391ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8617b75212223083dc66d17eda1b94ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_344afcaf3319ef760bd750c19106ac47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [14]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ca9fcf4790c6bf7f3ccf330293edfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_344afcaf3319ef760bd750c19106ac47
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a62237e086bb371160de0771c9d0f82c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 20, 2], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36792b7300849732f71a2b18b98fd7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3200, 2], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e275ea6f1e6351146c96284c7dde11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 26, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d76f33c9c05b068d4013dc23dc1e795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 26, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_790ef552fe7ba0bf22f5799f5499a98f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 320, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b825c757c379646d8971b3b698eaca74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 320], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abfe4c8deb983a6ab7b042e6ea927955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0542f979859ef985a430a5fc43ad9886(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e1d2ae13ae68bd8b743b35d43abc3df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdbf76eb7bdf2fdcc668d7dc8c4224fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0173b24587eaec60d8d5f48c9151fd6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89813cb812fce7d4993102344a40ceb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1051539472815c470c16c873354ac17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0db69120109d5758a5913edcbc57bf8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ad2282127be2075409f6873d3eb1f14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7c4d3b5e317e84c1fa525edf97143a9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-2]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7faf1abd051f590985ab0660eba34975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c4d3b5e317e84c1fa525edf97143a9f
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 18, 128, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d27f92e4738ebe43feced12dd1879a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c4d3b5e317e84c1fa525edf97143a9f
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 36, 64, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_173cd8511abdf84308b2c5e36273207c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c4d3b5e317e84c1fa525edf97143a9f
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 32, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16902e47eeda50ef9a11af74364e3362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 18, 128, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5d770b17239d7de10dd0cb4b5f6ced0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 1024], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37a3472b4882a439b3ffb23c840f962b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 16, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a550d2bafa74b60c7563fd46644fb738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 576], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58f1977c718f68fd411a4e937c3ac126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 160, 160], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a1345dd90a080436c5ace85de956e907(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c2c33816acc04f2734388e32ca1e29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1345dd90a080436c5ace85de956e907
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 160, 160], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_688eb4c63fda4e286f5f233a2b1131ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3aeecda45bc1e48eb054e2ce2d3bae9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_688eb4c63fda4e286f5f233a2b1131ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 160, 160], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14846d08ff63d575f63badf641a08956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 224, 224], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c13a47ac5c78dfa56a30b2907a94ba98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 96], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_881fbca4c83b327b0fe3a4c7f18663e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25cc80640ff6c0fce2e1778ab96a0e81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881fbca4c83b327b0fe3a4c7f18663e4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ec8f4457e556bf97149624ff015ed154(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93f7ddeae19f0130fb121a4f292388d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec8f4457e556bf97149624ff015ed154
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_76f36d620d9903c677ca7f7045b6ed46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2be48faece067bf0884f49eea6944fac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f36d620d9903c677ca7f7045b6ed46
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec90ff29bd60ec40a3258673f6e76e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86c24f4b98446a2a479344b93c03be64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 3, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09a3663402ccddc08108b8ef7488380a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d2741ce7653306cfd53546edd6914d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881fbca4c83b327b0fe3a4c7f18663e4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a608bce7e04434af33c834ee641b41e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec8f4457e556bf97149624ff015ed154
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_793b7a42311c1085591795a883c3db1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f36d620d9903c677ca7f7045b6ed46
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_637a25e5ec26470eb44a15bfac2a4dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6fea1b009af041acd44eb3e87fe5bf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e296f19da12518e780b6ca6f1cabf43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b1101ac2f82409c71b7b9a282da6f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881fbca4c83b327b0fe3a4c7f18663e4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86d306a91198bdfded15b90594cedd3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec8f4457e556bf97149624ff015ed154
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_604c49f31a25b22314b36c587cdd47aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f36d620d9903c677ca7f7045b6ed46
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_142d4fc299ec3902328274922c5ca66f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3a047ba0bf16889ce94fcc13ff51563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 12, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7036eb8a6e88f72b7659ad4f212a52bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 768], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce78dd86379eccecbd0337c0b2b144a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881fbca4c83b327b0fe3a4c7f18663e4
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a337c2e7f7f16b61ed88bf71f672acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec8f4457e556bf97149624ff015ed154
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f0de1c566cf8876fc515a77c174e161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f36d620d9903c677ca7f7045b6ed46
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ce1bc6b026b3b2c5084de181435f021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f7ac68a2a514fd65facbb6b92edcf20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 24, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_061b72a8abb1d7470bc8f91685faebc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 256, 144], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_18770a98ac1d0b6900b6cd3944da578f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b694913682b1fc493bf096d961f4ff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18770a98ac1d0b6900b6cd3944da578f
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4d88c05d58f5a8f4dd00fcde020ab334(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0700d2a4c9cea571b8e335ed173e118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d88c05d58f5a8f4dd00fcde020ab334
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e213ced38b9535dca09d3b48f062db25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_111736497c34f07748da6b4f61148718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e213ced38b9535dca09d3b48f062db25
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbe4995a0c1bf204f0555b66cbb84741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 64, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9689f7e00aa08d1078dbfcc35c74f614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18770a98ac1d0b6900b6cd3944da578f
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1277b4f1b1a4bb7c065e2a0f8d1d2ee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d88c05d58f5a8f4dd00fcde020ab334
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_326fb3d4fb3d053fac7f1f0f39f8455f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e213ced38b9535dca09d3b48f062db25
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8429a041b92403600886b52dcda9613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 16, 240], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7464c190ff961299deabbcf5ecb5b378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18770a98ac1d0b6900b6cd3944da578f
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bb7cbaca4e20dafbc236a7c5dc2a008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d88c05d58f5a8f4dd00fcde020ab334
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73a830b7a5d81ae64ea88f91ca36543c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e213ced38b9535dca09d3b48f062db25
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d369607b9fb2fe26c5a3d69cb18c8241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 640, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_456451f449c8f3bdf7531fc6e8faeb7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_49e3e99d9e887abc8e9767d2fe1145e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac0a09a02cf0467b8b22e3c208346676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49e3e99d9e887abc8e9767d2fe1145e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c9786b91d399b7a6fa0575c7a5f236a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        input_2 = [258]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec37b8b7ac52108def5a2330ea709e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9786b91d399b7a6fa0575c7a5f236a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_984710b7fbbdaf98873946006442f2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 255, 20, 20], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5091a421f09d57b8350e698747edf643(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f56890d59fe07567a6729e2432a7b921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5091a421f09d57b8350e698747edf643
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_87f5a83682a9b779f37a9daca08c9b21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d365913f0d576f91171fe21952a746e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87f5a83682a9b779f37a9daca08c9b21
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_41e42d17f58e4f19db4c78c264a68c79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [85]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fecd2e925adef83557235a20113138df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41e42d17f58e4f19db4c78c264a68c79
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_559b8650a5f3dc312f03c9a72cc96c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49e3e99d9e887abc8e9767d2fe1145e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c45b9bf66fc725fda8855a569ad0fc82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9786b91d399b7a6fa0575c7a5f236a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bc7463229ec22738d1a5e3d77abaf63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 255, 40, 40], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9049c25d323d85206924c734f8bf268a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5091a421f09d57b8350e698747edf643
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 1600], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32828f74d0b434c32d65a9d5d3971407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87f5a83682a9b779f37a9daca08c9b21
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 1600], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4c35c04ccd9479feb610c4b62715478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41e42d17f58e4f19db4c78c264a68c79
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 1600], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bb075b4f9ce7a921fee461e2082b28f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49e3e99d9e887abc8e9767d2fe1145e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e00f65b7a46f40300de229dedccf39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9786b91d399b7a6fa0575c7a5f236a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b756f36443a5a271cf6ce296f60c34a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 255, 80, 80], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b17c0534235ecffac3710405b8ee80f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5091a421f09d57b8350e698747edf643
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 6400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_802bca4111b7aa4e46982ab92fe8d314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87f5a83682a9b779f37a9daca08c9b21
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 6400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_818c0fa4460b8c1edc9ab7b6cbf485ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41e42d17f58e4f19db4c78c264a68c79
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 85, 6400], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e5b196f03a84c1c269b0b1a224e979f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 32], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e17843b812def310b61859d15421ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4f721bc10a7f14f9186f596e76248c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5e7cc040c0373da42b90218a40d77e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eca53b96630867be1be0c8d649e6f892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 64], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2cf5ef3903f8fde5895d93ecc6c7ef7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88f423fe0260e1ed04ff53cd13f3a2d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fa1044e2a9a06102684fac5f7a6dcc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_766a35fc52c76d584346aa6a97546522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 160], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb39ecec0820d82666b34543327b838a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb8fbedaf64f57e0c08c1c865469f4bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7584d22f42a241a617f7ba505ee7cbdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 640], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05a245d20bba39b25d415094dab81cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d81f432ada7abe62b9f8a964c4195153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f587126eb4c9248797f04db95cca2e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f15adf5006f17f1b96840d7028f73b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 1024], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_979c7ee497fb059456f2fb0f8a15f054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89c7d19e69be0d18056f3485b7ce0777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 4, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f6d84f314ff0b9ad51103fe9ca69844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 16, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_827c2615a34e56f3c6291ac7504cc5b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 4, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_106b62c8250cc0c1ca6ddd82e67e604e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 8, 8], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6ae377b37c30094ee7aaac93dd231b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 4, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_12fe7c1df8512c4c9a5eac9d3c8b9f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e69d5bc20574a4fcd6022d2eed4d7e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87ede644991bf1e246ee95a34d5c298f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb461a136e4958580e458764e23d309b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42efaf8f64e4fc7206c078bef30cbb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4da1b33e13aaf36edd1011b0aa9d0bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27d42dcc0749f5ff0d1f70b42b4b113c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db082ec3490acace06be4f583ac10da7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_352aee287c01b6ecdbc0b0005b91ab1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 1280], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a3b55924cc2b6e8f91f9ce5fe60c2c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 49, 1280], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3e59f1a8c424907c689828d7a11b3d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f3cc48f8bd1fab46785ce48c613062a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8307f4fa25759974d4e80f7d937e484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9de237a133cb3cdec9115f7b23dfaf33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43fc498eddbd110d1a0b11b6b03ab8be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 768], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56c2f050d5453d5839cbb922dc44076c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 768], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4142db571147ea34aea63114b8c39d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb7a2eb97534377ca17e804ab0a775f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b33c9a619e9f9511f5aeb04e4365aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 2], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_048b5cbc40c45914e86fc04faf7186e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a886496817e05784471712ff1fea1bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cce2326f857880d7fca227dc3bc6e5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1eb9f25b28acf420c8e092f5c8ae4014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10cdded873f007ed35e63658e0779ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea174b52a788e89898d965b774b74d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09618ca04020120b351a33e4f69fd016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_704d1f2af7632877321ad225b9f8e192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df7afcc16a955d4922325490b2104ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a84cc754efe233828d473a7cb036a3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 9216, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10390cd948923e6db93a46d642280185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 96, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e9862aa871d890449b4e46feb4702fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([64, 144, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcd0f85c4a1286a2af78907d125bb788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 64, 4, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b353ff519062edc69c5e1e4516602365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 64, 4, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_332a0614082a5959d303a23dc420f751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 64, 4, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9858627e4019b47d37172669e1a25bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2304, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3a19ecd5298b2513984351d83a2da52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 48, 48, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd7558a1e46e793b50d33a5b6c3ff990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([16, 144, 256], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1aa1372cbedc736e9480d4d5cc1b22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 16, 8, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c141f6c765e7deb95edf60ad2202ae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 16, 8, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5585cf28559a5e556c9c938531d39ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 16, 8, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38c17362322b4858b8f991a29be1b087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 576, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff6fec56f3e806c14bc4a0accdba2afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 24, 24, 512], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e9b691c18238d884629a6ed555e196d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 144, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38654e5b4cf31907491d99b22567d183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 4, 16, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5b25027e841f35f99977a1f4e705d99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 4, 16, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44af8aceac8c344076a98b7b493df7a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 4, 16, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_487a631e4a416f11880489979d26c35d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 1024], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91f801d9a0de18a3e709f9bdd1b8e88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 12, 12, 1024], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d5cd7ccffcc2c8d120ae31d5d63b98b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 32, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ed9b88cd8b47f643aa2615eed508174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 32, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0dcc63d258fd1791b3e17c8ee79881fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 32, 144, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efc338af793ea01c68adfa7ca9d4ef61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([196, 16, 24], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7252c7c6a22729c6813361e04cd755bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 196, 4, 16, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a160a867594669955fcfdf1b39f5c866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 196, 4, 16, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7ad02235c1dab2371480d553d73a382(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 197, 384], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9acf96831338a0e15952e7336065fccf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [197]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff6f9c59e512fedf136213c0cbb79e4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9acf96831338a0e15952e7336065fccf
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f270814d93b14db53a6c2492dd787a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92698bf2330d814ce4efe479741b4ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_004acdee4f19cf56f622cae84cced311(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33a93b894239a7d29655cfafbb069279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_004acdee4f19cf56f622cae84cced311
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_039e2fe09eaaefeee87ebfad736dbf06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 512], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fc2f1bbe3d8318050d14fa8f43c1fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 32, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80ac0ba842d53c107af862650e053b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 32, 16], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23b0bf947238e681e8e5f82085dfa741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78ec7370a5bf50e982877447175f5004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94fa2ff79822fb9c3794ec51520ac067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_848a9d6ce2c48279daa9152256afd027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c4d3b5e317e84c1fa525edf97143a9f
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 48, 128, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0a89b2bbb42fe7a75a4ad5feb18e8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 512, 512], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c4dbca031eb9bfbafcedb75d269be31f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb4cd227652b20aaf3ef91371034405e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4dbca031eb9bfbafcedb75d269be31f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0105e04b844edfb9142ed47ed1e71b89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f1930be19112eb013fd92e06857047cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0105e04b844edfb9142ed47ed1e71b89
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e19301f5b2e895c81149ed7182c5d3ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0832f6a6919c24014c5eafad55acb81a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e19301f5b2e895c81149ed7182c5d3ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c20b61ef3c4ebf69531eaad2c2fc98a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd070410f3f3ef4d1eb4036182c5dc32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c20b61ef3c4ebf69531eaad2c2fc98a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_471dc8d3ba37d132477d58373a6dc66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4dbca031eb9bfbafcedb75d269be31f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_faf595737efe0c098fb80892a282b5dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0105e04b844edfb9142ed47ed1e71b89
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef95432082b9b115bb14309f8c564c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e19301f5b2e895c81149ed7182c5d3ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34b871c02d41d60aa9f384e24a897924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c20b61ef3c4ebf69531eaad2c2fc98a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2c5c7076c78496c1ca8ccc37d2eb451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4dbca031eb9bfbafcedb75d269be31f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07136ebef3f6f05a8bfcfb93d6f8e75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0105e04b844edfb9142ed47ed1e71b89
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a30ad30aa20458f28f6a8e44f208e15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e19301f5b2e895c81149ed7182c5d3ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d47cfa8154ce76e6ca15af27f120674d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c20b61ef3c4ebf69531eaad2c2fc98a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b25bf2944e71d2648a0872e4449de4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 1024], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0d96640748d38c1f9d04fad68bc01b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fc0702fea676d18303044eb0d3bee5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8b5253bc3c1bd18a0db1d4ab09579b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 32, 24], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d39cdb4fb76a0730702c141e60ab39f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 40, 32, 24], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d54b5d031bec9a3e2b8462e077049478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 80, 16, 12], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b5d25da0e728399beb0fd3e0b4fde37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 160, 8, 6], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bceb73eb244bb78f2337c70a1a8d7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 320, 4, 3], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7189f35467e6fefe63ab5cafc0d447e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 17, 32, 24], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba8c921286a06fbaa41284bb29eb00f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 26], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e97bd5556b0c03a55e0157bb06c038f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2, 350, 25, 1], dtype='int32').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ae588c4b1b4e900e9b63da96fd1265e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 350, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fcfa8798f9a5e71b4c0e74dc6df97b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 350, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7499b65fe63df3c3b8ae38fb39eda46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 175, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_474e0388b76d070e00a9b29ef22346f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 175, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f00cae6c76d5965177161a33aa580caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 88, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4e89bb3cadcbd244292f01bcb8d8f97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 30, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a94ca9bf487608cc28c0b8ea44e0947d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 480], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb9e902976ef1da460d4723b51358fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad5ffa4126db11291fd829618cb38242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 18, 112, 112], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbff18c508e5491befa50b251592647d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c367d84ffbef7c821e88e8d8c5c0f1fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7e33e5f1c1c6b6a77621be8a7dd22a26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9353655a5bc35afa972c875b9b763ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e33e5f1c1c6b6a77621be8a7dd22a26
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19432619efc06d5413bc0c959b3ad86f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9acf96831338a0e15952e7336065fccf
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ebc429c463a4a384f9a08a7326b21aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 8, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec8c3016120c5944b20a9c11c6af919a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 8, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1853876b71437cc42c46bca9411b2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1345dd90a080436c5ace85de956e907
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 8, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64928cbb74e82dd608fc07e3911fa376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_688eb4c63fda4e286f5f233a2b1131ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 8, 64], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddc54e52158f51a8f39029bae90f4277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0863b345b7305da1d5f836ad72c339df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([25, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e91614069d30a0a84b9f84cca5902280(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [25]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e15b831e0fbe0945ff79f9a09efac104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e91614069d30a0a84b9f84cca5902280
    def get_inputs(self):
        return [
            paddle.uniform([5000, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(25, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ed0efb1ab1904b354c0ad59b8903209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 25, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9a1968a352e422f5ad1334256d93760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd078d8df109263a50b06a25f2bc5fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0d9928b53b3540e1bd29b863ca1ba5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c72ee156393c3f8041d2ba3a374a315e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_47497b955999a90efadf39903cdfb398(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5417bf0c51ead862e8ac9fb25bf38a16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47497b955999a90efadf39903cdfb398
    def get_inputs(self):
        return [
            paddle.uniform([5000, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdaecada2d4c1199ab7c6ba9ef2581f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a57f23a04ae1d3f529051d2c1e8d06a
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e980b9e849df61c6c1781bc25baad14a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 1, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9910f8306b155c92a04ed666f4c9109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 1, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e92de8cac4ad5bb7a788ad849c29578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 1, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d458e8c6f25712177e50f8f46b59f8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_112c6f8dada9c81f2b7d96a3a842284d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 25, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7617124d6495b45c588d13ef633450b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6aee0d485f8f5535dd83ebd82eaed150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7617124d6495b45c588d13ef633450b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aed01b1d3184b69e20764c5eb34ead5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 240, 240], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a0bbca95428b0c4df96a7d0fa26f4b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1345dd90a080436c5ace85de956e907
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 240, 240], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_395bee32497b64ba85f5e566f2014f74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_688eb4c63fda4e286f5f233a2b1131ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 240, 240], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c702e8e8bddd19b0c1dc71e24664323e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1536, 1, 1], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d759a6715395f23ad7b521b0adeb999(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_004acdee4f19cf56f622cae84cced311
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_491ae50c8c14b20fae8a5d34c1b81451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 196, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7de3388c637d20a0c53ee304d64ba8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 196, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e094cf590d47e57f65cd0b5808697e7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [197]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e0ba03c4b165bb32cdec46724615a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e094cf590d47e57f65cd0b5808697e7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79f2de4ca12e83168162ac26d87b2862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b07ac70fe7ade6ab3309b34678508d13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e4683499ce4ba345302baeef8b23aaa5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_655768c87c54886c2ef20b84005c0ced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4683499ce4ba345302baeef8b23aaa5
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91b69c0fd04f59a5ba4893f2d50df28b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 56, 56], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a496a102748667c432760442310cc24e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a43cf27c1dfffa9dd30248771ba8f538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1660cf038afb490214813ffe049989d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e47837d61a34ed89bdbc7d7b544d143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 197, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bfd9ec7de764cb06091268475c57f56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4683499ce4ba345302baeef8b23aaa5
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8e8f8467acbf89f283be18ba31abe34a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [27]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c727f1a34830e2ec0d50886014bf7f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e8f8467acbf89f283be18ba31abe34a
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6a23bccb1361ecc456046a0c48fbd015(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [50]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0bbdbda27c949dd6c5a9cdcccc875658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a23bccb1361ecc456046a0c48fbd015
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e0f2a969a3261c726e762c3ff478fa4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [72]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e53fbc96181d47b5ee0e796b8cebf73e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f2a969a3261c726e762c3ff478fa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_21b3f45c207ce442132ceb943dbeecec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [84]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e49d2dce8f255a0df742a4431112af70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21b3f45c207ce442132ceb943dbeecec
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_86fb6c6543526d61a5975299fd800c35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [95]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e09df5a50997ed9cb74b66d74222de45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86fb6c6543526d61a5975299fd800c35
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1b7b8ab8a511bf0c9cfd63310c1304e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [106]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ac300bea6f936fb83a291ef23622dc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b7b8ab8a511bf0c9cfd63310c1304e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_157212349405d7bc06843e1cc14d0ca3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [117]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2defcda8f3bc4a467681bee1e7ae0c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_157212349405d7bc06843e1cc14d0ca3
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2248d7e78cddb9d51afdd89699dd7f1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [140]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb66cb93bcf8a2727b2b5ffa436d1fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248d7e78cddb9d51afdd89699dd7f1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_567068c37ec96ae8466a1a4b23137fcb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [151]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_839b27ad9ba9be24e5e192a62900278f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_567068c37ec96ae8466a1a4b23137fcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_64eaff48d557f5dde7c799a2c3e93f1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [162]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08068dc4d9553a58d3aad591715c5786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64eaff48d557f5dde7c799a2c3e93f1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0a0e1ec3b5499be51aac0f448a6131c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [174]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e58e2d0e79857c486d77f06a01f11247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a0e1ec3b5499be51aac0f448a6131c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56ffe7e1fc521ef0594debd208fbfc15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 224, 398], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f8bf501e9a0668a721d358c25a151433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1345dd90a080436c5ace85de956e907
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d1211c937d4e42a386b79aefb1f8526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_688eb4c63fda4e286f5f233a2b1131ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 72, 28, 50], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e7ae393a3a3f90e272807741ceedabc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1345dd90a080436c5ace85de956e907
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7ec29a250d4f5691ed5d27573267f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_688eb4c63fda4e286f5f233a2b1131ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 144, 14, 25], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59331dd086671018b617ce6d8234fbcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 18, 112, 199], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a873d3e6dc2cb146da857c4d7b4e6a4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [113, 113]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ffa93e0c51a7a1f4cb3501186b5d8bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a873d3e6dc2cb146da857c4d7b4e6a4b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 113, 113], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_440a4a4baa3986fd401a485c8b9f23d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [57, 57]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5f7e175c64c703a13b1816f1062dbaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_440a4a4baa3986fd401a485c8b9f23d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 57, 57], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_75c52e8e9bec9b95995c0eab604383c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [29, 29]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58e6572c03a565467de43678220c7c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c52e8e9bec9b95995c0eab604383c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 29, 29], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c1ab6248b005f45b4687d7b571d4590b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [15, 15]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_537fa51f695bebfd5d0adb80961bc448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1ab6248b005f45b4687d7b571d4590b
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 15, 15], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1541edd2189bf361d021bfebf62059ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [8, 8]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46c41d1fb274ebe282a42ad28c3a77b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1541edd2189bf361d021bfebf62059ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 8, 8], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35cadafde8d7c3643915012d4d032688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 384, 384], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06e1be2c61a3049fca26eadca1b76d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_636a620a611a7e6713be9bd12ea7cf86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6622294649b2763012209f3685ed44c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e33e5f1c1c6b6a77621be8a7dd22a26
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b72bccb9e53ed97847e91df20e8a375b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_004acdee4f19cf56f622cae84cced311
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94a19c8266aeac26fff967ee7199026a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 24, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89a7124a78162a1dfd4176a9e7a1ca1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 48, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1422f40523876910b73a5c15b9fcb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53dcf2c68707f40f8ee97ff6feb9e253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 128, 256], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdf5957d8631b9733a169ce62edf1af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 512], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ff4c786f0c06fe70f364c8306fbbf66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3, 1024, 2048], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f8e422e01ebdf811fc48b1cbf66fd19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 48], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2857e7fb550cfb1be4370ff24fac68d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 48, 56, 56], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e78793fd17d48bebded228684f40a177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([56, 56, 48], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1513fa158b840a8e30d9681fddb2e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e20dff7a277b4b9350e4884246a2c990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 96], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4bb5793962f22f4c670fb62fc79db4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c8b1d6dfd8035c8e5b962f6c991dc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 56, 96], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c13f1ce6587b379dca5781a775fd6eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93eea89767f6e02e6c8f265fa44edad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 196, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec4a7bfa523575cbdd8489697e32de54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32d4399f9ee34d88ad3fd9c8a3024b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 98, 192], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c4fc159dab5b77e2dca12940d809849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 768, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_163d69f77e3abbfb28333026c4c928e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4582cdef41eb9d1b8e8e8136cb3026b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_163d69f77e3abbfb28333026c4c928e1
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_28adcaa09532273059c0a159a8a19bc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fcb99d6fd09c98267fdf78cef7bd223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28adcaa09532273059c0a159a8a19bc2
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_72e32f9a8399aee6cf049dbd1a8d36ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a09b11fb4d81c66503542dcad1f25333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72e32f9a8399aee6cf049dbd1a8d36ac
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d3980b2dceb5f6bc46453ca65ba25be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0eaff447c8b05df4fbf76931efab18df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd50238597a10e6b6c9f6bb839b2009f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_163d69f77e3abbfb28333026c4c928e1
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b11db5e7183b234bf2cdc61e8c4d4359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28adcaa09532273059c0a159a8a19bc2
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4eb1402c593fafde3e7ccd0d41c77b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72e32f9a8399aee6cf049dbd1a8d36ac
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8258493bb8c82a7ff27b2d2aa6903dc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4b99c4d61afc33a34fb34193da5bf58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9701ad78bb808fade06abe0884cca0be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_163d69f77e3abbfb28333026c4c928e1
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec5fa93150d5283aa8342d75650aa6c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28adcaa09532273059c0a159a8a19bc2
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b0b8d13703c1f6fdb1ae731f58c3877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72e32f9a8399aee6cf049dbd1a8d36ac
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2931a649d6ec7e39528c83b2aecd1469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d46db91cb09ea76504d50cc8038d313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38204ce947038dc3dac45e07b5f09ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_163d69f77e3abbfb28333026c4c928e1
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce2bb4a99d0389f745ff319d8b173fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28adcaa09532273059c0a159a8a19bc2
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2f6ef6f0f2a4afd04f1031bffbec7b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72e32f9a8399aee6cf049dbd1a8d36ac
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff98e89333379636342b27aa2f671956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7c7e361d334c8efeafad88a4c5603c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_64a3fdad63e3da4543e30c75797c9240(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [13, 0]
        input_2 = [14, 1]
        return paddle._C_ops.slice(input_0, [1, 2], input_1, input_2, [1, 1], [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a00b22d7eaabf673d353ea245c8e7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64a3fdad63e3da4543e30c75797c9240
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6828d2f8f568746367337dadec59b593(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [13]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_452bba1ee3174c464a37af039f3ce026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6828d2f8f568746367337dadec59b593
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_aef0912cc7836b125ce0486a8f649324(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [14]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e69eba3fa46af2379a7be8ac5f2899e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aef0912cc7836b125ce0486a8f649324
    def get_inputs(self):
        return [
            paddle.uniform([1, 54, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_306154b404f2cbcfd26428f407df54e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 31, 31], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6a2cc22e5f5e521e0457dd13b5207cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 61, 61], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6fc16d869d2eb4cadd174cd9f83e9cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 96, 122, 122], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df3ad8cda08621f5f1cd74741ee3c89b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 6, 40], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f57f0322b75e3b9e8584cad56ee17a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 6, 40], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4af445f361d20c57b5a12761f2f24fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1345dd90a080436c5ace85de956e907
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 100, 168], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30e8263e4d11f37190fd900c37f70ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_688eb4c63fda4e286f5f233a2b1131ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 100, 168], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de1b48ee9c9784b64a317f19b9930ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 128, 128], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4348eb636d833ce62ec38792adabc591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_864f17bb7541bf6e9a8b7441b5ca0982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_15619e5c9b1ea6bd7c6f16127134a319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72269a9009b4773b07f0155b2acafb89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4683499ce4ba345302baeef8b23aaa5
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5ff6591488f4348e56fa69bab217444e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [113, 113]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19f829fb60db4dd7e8cb24d70316c2a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ff6591488f4348e56fa69bab217444e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 113, 113], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9b33920e27c4ef1f9de0958e282009ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [57, 57]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cbd92f1c7acab07578200712164cde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b33920e27c4ef1f9de0958e282009ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 57, 57], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a38840526e8c6a709c6e3f0c1f0ffd56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [29, 29]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7262b8df0cbf7bd39c0daaa36332debc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38840526e8c6a709c6e3f0c1f0ffd56
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 29, 29], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_55054443d72392f7f0b4a88e24ec6aca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [15, 15]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f41393968609ebb079cca84f5da44c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55054443d72392f7f0b4a88e24ec6aca
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 15, 15], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_cfdef21f520cada8ff58ceca4aad832d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 1]
        input_2 = [8, 8]
        return paddle._C_ops.slice(input_0, [2, 3], input_1, input_2, [1, 1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aff64161905f96a9d0818f3546f05ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfdef21f520cada8ff58ceca4aad832d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e27cd5ca9a191d59271dd67409af1c4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_535865a6479a4be64df6e51cd48da888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e27cd5ca9a191d59271dd67409af1c4b
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_da2301b14275dbcae40a3aa2bde3c7fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7e204d94f75c8395158f4b587167150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da2301b14275dbcae40a3aa2bde3c7fc
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7e0ec0f06208101c6b8b8fb03c6dde54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [3]
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49ad6622091df016f8e968e25e6d8ac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e0ec0f06208101c6b8b8fb03c6dde54
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 256, 36], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa4a26360b83ce2e7c7f6a43c1914d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e27cd5ca9a191d59271dd67409af1c4b
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_746a7bfd5042f50a4eedea3a4475e0bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da2301b14275dbcae40a3aa2bde3c7fc
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1abe319942681bd6d91d5d9c32dcf37a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e0ec0f06208101c6b8b8fb03c6dde54
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 64, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9aecda6f08dd0b0c20a3d4d2f86f8605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e27cd5ca9a191d59271dd67409af1c4b
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bc4c4feb0a130e0c5db69e8d4aa4950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da2301b14275dbcae40a3aa2bde3c7fc
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c26d22ba49385669bc07d1de9750b0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e0ec0f06208101c6b8b8fb03c6dde54
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 3, 16, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a39bde3e134959bff9160512c568764c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22f71d57945f52459ebab207a4e0e75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82153099fd1d07cdd63b30adaa624f86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e33e5f1c1c6b6a77621be8a7dd22a26
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 5, 196, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec95c10303104041ad7efe50dba3a68c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d46fca7ed9460af68e3ff31af4c2e9fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5156b68911b2ad033def8dc1e721db65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e33e5f1c1c6b6a77621be8a7dd22a26
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30d0c4ab8360c975df3430d2ebef2f9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1345dd90a080436c5ace85de956e907
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2fc23274b210b9a828ebe4c0c8f71a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_688eb4c63fda4e286f5f233a2b1131ec
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa223b918e675062f05865ed37da21b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df00e3779a4d958dcd3379cca3abf6f8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 32, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23487415c528274671baae6889e67e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([256, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_825792a24c845fb9a59e9f21e3781637(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [256]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_274efeb8df950c5d8fbfce313447a44a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_825792a24c845fb9a59e9f21e3781637
    def get_inputs(self):
        return [
            paddle.uniform([256, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9642ec873cd05fcc9b6df70c4666fa39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5101aba47a86c61eeae4bc34bda8335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3127d685a2076bbab730da074b015925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e33e5f1c1c6b6a77621be8a7dd22a26
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_088b615034c8ec4e38d4952ce061ca86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 8, 32], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11398537c8e2bc89db490bc0e5b8a8f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([26, 1, 512], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_b9232b375dc2013d9262f05e181e698b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [26]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c6ffbc89b1a0d0b2b3df168a03aa047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9232b375dc2013d9262f05e181e698b
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(26, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a71b42d9b671b4dac2fc5e393102060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4654e57a046aca148da6afa1b5e39b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 26, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bfb977c1e938f48badc599494aaa99f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 26, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_cab1f73d32fd6852476f44a8aacf56fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [25]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c22acc03d895d4f77bf471a67fb7bbb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cab1f73d32fd6852476f44a8aacf56fa
    def get_inputs(self):
        return [
            paddle.uniform([5000, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(25, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ad2684529a786c8f6f9324052c93711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_997a6fbb237f7523eaed2e4476c9350b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_12728304ba1e2cb8a853cbc7e2bd9d4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e33e5f1c1c6b6a77621be8a7dd22a26
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 25, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4c1a3db71ef95e8b3db23153fb0864ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48bb96e4856b83ab3f7ff980e8e353ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c1a3db71ef95e8b3db23153fb0864ca
    def get_inputs(self):
        return [
            paddle.uniform([5000, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1768e4cbcb9b9bcf2c3d65fb01769fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e094cf590d47e57f65cd0b5808697e7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_31bc42f3b2d90d320e3d1eaeef16cd77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [27]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b79ad80e32403433e4c25dc17fe3c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31bc42f3b2d90d320e3d1eaeef16cd77
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4281addaf229facf1901d2b1c906cea2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [50]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ad1d4b1af644c7cbf1de1c019ff1992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4281addaf229facf1901d2b1c906cea2
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e9d098e1fe5b9f2eaf737ccaf189acf0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [72]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c9cdb9e64e8c3bde708a2d117a68948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d098e1fe5b9f2eaf737ccaf189acf0
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5f2cc0a37bfcd24f7330d128e0b62f84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [84]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0b0abf6977f6698fee517cb213779c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f2cc0a37bfcd24f7330d128e0b62f84
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_bbe0a534af7232bcbe0f1cf0cdeec294(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [95]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fd59932aea6e534705dcf74825fdec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbe0a534af7232bcbe0f1cf0cdeec294
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2ecc729c38909f4adaf1380d4f24e82b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [106]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36cc913e4afba8fd614d3f722acd1bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ecc729c38909f4adaf1380d4f24e82b
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a67d1784e926b93a14fac9add89b9f02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [117]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6a24a2dbd822f933572922b51729def(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a67d1784e926b93a14fac9add89b9f02
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e7f6be212783811d478e43cf8113c63c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [140]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f38d5adc7f6fec8fc6ad8f7594ab2a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7f6be212783811d478e43cf8113c63c
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d3a268d5ede1e6b52ffd68055468e36d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [151]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74e69f89143faff6c8adac850a47a86d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3a268d5ede1e6b52ffd68055468e36d
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_87ebd48f40ad40984ff339a1fcf2e4a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [162]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1664efc42a8aa581889aeff97254f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87ebd48f40ad40984ff339a1fcf2e4a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_194c19a8499b76523eaf4a017fc9c465(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [174]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d8bbe461d6cf15df8af7fabf8da6254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_194c19a8499b76523eaf4a017fc9c465
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce2e22bb21a2767fb125ae94420a97bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 3136, 64], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_890ee811603c63d8b397bc1ed71d74ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da39e9544272c12ffcb44132ffb8eeee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2debe7e9ba71158e6685971bf8a70bcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 784, 128], dtype='int32').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52b9c2648033244eefd84951aaf096eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_819b53d9f5491777fdb8f7dcb138eefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b474668620f3fa25707246a353ca01e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_896132b64da0d15752f97558f7487259(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ccf989db77a847ad9a48217d70df9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d95517cb171c8de7a3c71f633ddc49f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9cf55a1874ca9e7e09fa8289aa837cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1513e0b1eae1f8fa936a79694d9aa76
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 64], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89adb31af544a224e34a658e4c608338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77e3b28e618fc4689296c4d77f29320f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 1, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c91e114e8ab76eaba5ef159d0503fe75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2fb2f0492ecba81e76b66a7afb4bbdda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 2, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a551c0a454070f5ed556e77d317f489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8cde22049a031ff08cb35eede22d704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 5, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b46a2adc1b5d81ca33b6c165429129b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a54f7f333840b0a33552baac8ecfcc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 49, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f5d3472c651da3f3c687be579a86491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 4, 112, 112], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_593142b6ea588dc9afcb5195c4e3d70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 56, 56], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a83b5223ee0f09bd559b0dbfa8f51ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 8, 56, 56], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f74d956d8e9ad2d46366ebfdc78c531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 12, 28, 28], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8b578b2bcef1413f06a12c86a39bace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 16, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d293752c81bab5b7c48b75b4754edf91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4414571e17858ae97ed7301f89355b8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 32, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e10218e70e2dbb71126d57bb198ce7ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 14, 14], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dfc5f71075ccdc0da7b5606898423dca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da67fc9e66429618480c45d62c83960c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9ecf52cfff7c012741f627e577793bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e23775fa7702067b81365e72316e9d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8e282d869808f3d1c28969c80ad1a7
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 384, 7, 7], dtype='int32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d851e8b74d9a54d911a66f5485e78c23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85fa2c42d19eef10c0d39245d74ac96f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d851e8b74d9a54d911a66f5485e78c23
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_07a295689200b3b6cb6da2bc679216ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af50a6f67fc072ce9bb1aadd246ac514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07a295689200b3b6cb6da2bc679216ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_88abb96a636e11883a1e3f3c8e006a80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [-1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d864ee93e932f1065b7827f78c8235e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88abb96a636e11883a1e3f3c8e006a80
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_cf690fe39c32b6575b3fdc7f47ec79e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb3185a0faa80242906575ed5c296c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf690fe39c32b6575b3fdc7f47ec79e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ecd4d7291a6f5191632571ed1c68e88c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f3d549f07d3e31a438cead9259665d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd4d7291a6f5191632571ed1c68e88c
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1d2abed77cb82e603ab419b144aa0765(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [3], input_1, input_2, [1], [3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_077f10a7dfe5f9b14cd7f97a541891bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d2abed77cb82e603ab419b144aa0765
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bdefcf7033d6dcd1b8717599b0d3db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd4d7291a6f5191632571ed1c68e88c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d332171297d92267e5437e4a7e320eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d2abed77cb82e603ab419b144aa0765
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5dcd186869cf279b53052eb4ba40d636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd4d7291a6f5191632571ed1c68e88c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdf45eed357b3b5e237557e45a5c4213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d2abed77cb82e603ab419b144aa0765
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6beeee3562069759f0cedb466d45e246(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75583060375ca6b8dd4fac9166916bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6beeee3562069759f0cedb466d45e246
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_281c3682966b63e4a7c1df9d11f701c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8a937161a88869271fbb6fa36658e4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_281c3682966b63e4a7c1df9d11f701c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_74ab2f8d679a0d55ee909b91c625b039(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [1]
        return paddle._C_ops.slice(input_0, [1], input_1, input_2, [1], [1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba149d3da4da932cab6887706e28057f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74ab2f8d679a0d55ee909b91c625b039
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40ae1977739a48bc74469c21e1333dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd4d7291a6f5191632571ed1c68e88c
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_855e07131c1c7cc7a682e1f99db95e34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d2abed77cb82e603ab419b144aa0765
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320, 2], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_862a1b32ddb8aea32f69366d61b1fd5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d851e8b74d9a54d911a66f5485e78c23
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_819a27eab8c044a44563228aeac8d253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d851e8b74d9a54d911a66f5485e78c23
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a180b8882155f8808879b9ccefbee90f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07a295689200b3b6cb6da2bc679216ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e4c0779e81a8fee16a57d2b6d3adcaef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [256]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7681e434c0e018914069f0546737fac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4c0779e81a8fee16a57d2b6d3adcaef
    def get_inputs(self):
        return [
            paddle.uniform([256, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14c3e4c07c87eb45d654601731f7d0e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_763816a57771e9d382dfa3599a020adf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d22b760afdaf6229f4db83fe10883206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0310595e1b96f6b2bf3eb7b9e3e80507
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 8, 256, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_870b4e0753ea5e23b4627db9a93409b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_2_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [26]
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [-1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_616b5bec41431883448c5a92b66ed33f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_870b4e0753ea5e23b4627db9a93409b5
    def get_inputs(self):
        return [
            paddle.uniform([26, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(26, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_865aad8388647771b79b45247c1c68ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745c7675e4471f74f7e46d8c1d0b5bf9
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 26, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22206b717ab89691f3ee68f98b93b5cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0542f979859ef985a430a5fc43ad9886
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 26, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e2d589da1f1c06811fb5a5199bea8a7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        input_2 = [2]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f8e8023b41c1cfceaba0b3083689ce64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2d589da1f1c06811fb5a5199bea8a7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_fc5bdc8448ae33538770712079d426ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        input_2 = [4]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73b15cf88a05270f6d6e70a56e662dba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc5bdc8448ae33538770712079d426ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_bd7a66bcf8d8fcea0d49264470d3c9ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [5]
        input_2 = [2147483647]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1e3d6ec9ef81d50d9279401878c3487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd7a66bcf8d8fcea0d49264470d3c9ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4f25972c4b5749039587eedc171870a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [4]
        input_2 = [5]
        return paddle._C_ops.slice(input_0, [4], input_1, input_2, [1], [4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a413654a6aca1971dfd7480cbef26c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f25972c4b5749039587eedc171870a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c36eda4a8bfc74acdacf9e10c6ad2833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2d589da1f1c06811fb5a5199bea8a7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98a8e5e8360f0faf131738698bddb9c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc5bdc8448ae33538770712079d426ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c22b94ca75c49fa2b94ed72cd1787c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd7a66bcf8d8fcea0d49264470d3c9ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a62bcc65ecfb47027ff4e06dcecc13cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f25972c4b5749039587eedc171870a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1121c602f6b998efab14f43ebbfbd109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2d589da1f1c06811fb5a5199bea8a7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5df6c759e7d3bbcdaed83e0c602fe777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc5bdc8448ae33538770712079d426ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fd8be107d44380989ce8e50c76d437f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd7a66bcf8d8fcea0d49264470d3c9ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_299fe644815edf7980e3fdca8f35a992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f25972c4b5749039587eedc171870a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 85], dtype='float32', min=0, max=0.5),
        ]


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