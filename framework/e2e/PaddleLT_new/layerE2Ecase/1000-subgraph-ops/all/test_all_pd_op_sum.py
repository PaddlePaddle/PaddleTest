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
class PrimitiveOp_f581d793b37c655276bec41e44874947(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a23d9dd91b6eca60f73053bda05c5fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f581d793b37c655276bec41e44874947
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a09fa73bc2a2a5a5eb908a3f5ec99e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4321], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ebad67407114cd2888246deb2b826dc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dd26db87d3126260ac6c5284eafed0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebad67407114cd2888246deb2b826dc7
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33927de077cdab2d962252d026ffb3c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e725f8f16dcc98865b16222e2aef71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e725f8f16dcc98865b16222e2aef71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6181c4c2644fdba3dac62d364c3a6aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0255865678191185, 0.006420324090868235]], [[8.320515917148441e-05, 0.000250508077442646]], [[0.0005496192607097328, 0.000657214957755059]], [[0.01133652776479721, 0.03480735048651695]], [[0.018511196598410606, 0.00013180857058614492]], [[0.0006387664470821619, 0.05736129358410835]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8646c9c92a6b9d09276b1b12f283c3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.07639110833406448, 0.09174058586359024]], [[0.020811760798096657, 7.844271749490872e-06]], [[0.005242577288299799, 0.006416682153940201]], [[0.007183857727795839, 0.028547251597046852]], [[0.015565316192805767, 0.12092229723930359]], [[0.09130977839231491, 0.022253159433603287]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a1943fcd467c716c9c9fe98068bbabb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4219ebd0a0896c08ebf3130451e09752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1943fcd467c716c9c9fe98068bbabb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c036f2658b02a50892fd7f596243f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1969531625509262, 0.27510517835617065, 0.13978993892669678, 0.057206202298402786, 0.01287273783236742, 0.16891586780548096, 0.23066169023513794, 0.08330577611923218, 0.16629137098789215, 0.21041962504386902, 0.07738223671913147, 0.2509373426437378, 0.058800287544727325, 0.1671455204486847, 0.11810829490423203, 0.17455364763736725], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8502292ffd745351eba85a39c28e06a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8691ccc3f3cbe682d32db5370c1fd112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acff5bba89495967566948b9e27d00e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_b14cb593195ebde69945d4c84691815a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc58357d1c8bc9d89a25daa555028727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_27c306d248877430ed152c9699bd02ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_173a5874e801fce3554911f8a2b4969f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_173a5874e801fce3554911f8a2b4969f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eff2da3f63346f42401b6d4315c7f25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36792793f6b04e7040182e91be12aa3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2154858112335205, 0.11849692463874817, 0.04890155792236328, 0.2522619366645813], [0.04421807825565338, 0.3680357336997986, 0.3360503315925598, 0.17184141278266907], [0.332043319940567, 0.31710129976272583, 0.010255128145217896, 0.006749331951141357], [0.32686784863471985, 0.3103213608264923, 0.13777750730514526, 0.22633054852485657], [0.04447308927774429, 0.35153087973594666, 0.08885534107685089, 0.0066995322704315186]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_947fa2b3364eaa2221440ee79e1fe59b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b22eef7c9f407dd3ee42df1ed337d6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_947fa2b3364eaa2221440ee79e1fe59b
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf15628d2a7935d10c4fbf1bb71d5579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3231605291366577, 0.04903687536716461, 0.07419966906309128, 0.03340575098991394], [0.044831693172454834, 0.05826646089553833, 0.2503412365913391, 0.11409053206443787], [0.20238913595676422, 0.10533830523490906, 0.1425924301147461, 0.17674031853675842], [0.044831693172454834, 0.05826646089553833, 0.2503412365913391, 0.11409053206443787], [0.20238913595676422, 0.10533830523490906, 0.1425924301147461, 0.17674031853675842]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1bd865dfe9ef91ef80c8c0838d5db4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebad67407114cd2888246deb2b826dc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d8e23a214aab71809c8444277716859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48a0fe34d03ac09462632c7a7c713ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48a0fe34d03ac09462632c7a7c713ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ef402b0bd6dda396ccd49846c31bffb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1390954852104187, 0.033391281962394714, 0.18100443482398987, 0.15556922554969788], [0.0509207546710968, 0.0248238667845726, 0.15702444314956665, 0.0630301833152771], [0.1339416205883026, 0.2178063988685608, 0.03409586474299431, 0.23763121664524078], [0.0509207546710968, 0.0248238667845726, 0.15702444314956665, 0.0630301833152771], [0.1339416205883026, 0.2178063988685608, 0.03409586474299431, 0.23763121664524078], [0.26555946469306946, 0.12011370062828064, 0.135416179895401, 0.2158403843641281], [0.26555946469306946, 0.12011370062828064, 0.135416179895401, 0.2158403843641281]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54936e165dd69b02828cad09c77012eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_353023a9e1777c9fb64f3d4114b0d44e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9febe40f6b3a88feb19e03842bb1071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_353023a9e1777c9fb64f3d4114b0d44e
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f88d7b6c1b03048c675c3dec62346c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc58357d1c8bc9d89a25daa555028727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d85455582e370d8d822b45b83c1aeb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d85455582e370d8d822b45b83c1aeb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eff2da3f63346f42401b6d4315c7f25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_61e10fd16de6fe0c6eb3adaf4edccc33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fc384a0313ff2c77e7833e2e0456f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e10fd16de6fe0c6eb3adaf4edccc33
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36f4d2c495f778ca15b694763642834f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1585461050271988, 0.2030963897705078, 0.07531548291444778, 0.06819995492696762, 0.06903404742479324, 0.18677054345607758, 0.15702764689922333, 0.06769457459449768, 0.16080789268016815, 0.10624753683805466, 0.027685679495334625, 0.21551458537578583, 0.03189590200781822, 0.16047513484954834, 0.23134367167949677, 0.2134716957807541, 0.05273635685443878, 0.18173585832118988, 0.07269302010536194, 0.0032052749302238226, 0.19810013473033905, 0.1544133722782135, 0.22863608598709106, 0.22110961377620697], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28b34ad5d1846cebf0f7cd1c018b8c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00ee45521bb5947047264d069f33cced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00ee45521bb5947047264d069f33cced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_bc93a4dd9fbefeb7bc3815e44d0d4e36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e478cbf7e713aa4afaef2f7421c9986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc93a4dd9fbefeb7bc3815e44d0d4e36
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33f85b4631df586640a0b67e474ec943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24059699475765228, 0.20401248335838318, 0.20136485993862152, 0.07779122143983841], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26056e24fee0c96620b3a7f0d4e4ac5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.443401038646698, 0.01879914104938507, 0.030096083879470825, 0.05628645420074463], [0.1597653031349182, 0.03256210684776306, 0.04690167307853699, 0.3367536962032318], [0.39223048090934753, 0.10811522603034973, 0.1506238430738449, 0.11638037115335464], [0.10637867450714111, 0.045472901314496994, 0.35786545276641846, 0.22567766904830933], [0.10637867450714111, 0.045472901314496994, 0.35786545276641846, 0.22567766904830933], [0.39223048090934753, 0.10811522603034973, 0.1506238430738449, 0.11638037115335464]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62013ff1d54ec28cf7015b82b77fca7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1487407386302948, 0.2564408779144287, 0.11654667556285858, 0.08824345469474792], [0.04823960363864899, 0.033418506383895874, 0.005522549152374268, 0.1534278392791748], [0.245321124792099, 0.19668203592300415, 0.02901521325111389, 0.08745982497930527], [0.02178362011909485, 0.023879624903202057, 0.16128362715244293, 0.22988560795783997], [0.1487407386302948, 0.2564408779144287, 0.11654667556285858, 0.08824345469474792]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc46a9926d338223304783a1e281272e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e478cbf7e713aa4afaef2f7421c9986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc93a4dd9fbefeb7bc3815e44d0d4e36
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fad840c4fa5a924d45fc52909021164b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3725680708885193, 0.014811739325523376, 0.08545148372650146, 0.3024505078792572], [0.17571653425693512, 0.4427066147327423, 0.16599597036838531, 0.1372987926006317], [0.2488720715045929, 0.04982655495405197, 0.06996200978755951, 0.3496866524219513], [0.04939615726470947, 0.05756248161196709, 0.2488795518875122, 0.1573629081249237]], dtype='float32').reshape([4, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_380643f43a83add64561a2f1f1295527(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c906e6ddb17e1e38fc07834efd0d9b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_380643f43a83add64561a2f1f1295527
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fc384a0313ff2c77e7833e2e0456f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e10fd16de6fe0c6eb3adaf4edccc33
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef012bb61ab49ae618b23b581beb16b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d2d313922bcf6debd7c96ab53cbe884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34964b306f6aff3eef08681a2fb88f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc8f1547af4dc44b51b16e846b0d7765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f146af1daa504e9a61a301f5c2a7b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f146af1daa504e9a61a301f5c2a7b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11c0443456334565a37ed7f954128799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_380643f43a83add64561a2f1f1295527
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6b5c19d807191d874810418f8c2a60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2546539306640625, 0.08790834248065948, 0.16270142793655396, 0.00904536247253418], [0.2546539306640625, 0.08790834248065948, 0.16270142793655396, 0.00904536247253418], [0.0754576176404953, 0.21234965324401855, 0.06143989413976669, 0.032623033970594406], [0.1800021082162857, 0.2203885167837143, 0.05525475740432739, 0.016759246587753296], [0.13451582193374634, 0.16116155683994293, 0.04648391902446747, 0.3108064532279968], [0.10491010546684265, 0.12081600725650787, 0.38842397928237915, 0.25860774517059326], [0.044446662068367004, 0.09552852809429169, 0.39697954058647156, 0.05195990204811096]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d8570e694ab13a7dd129024770c7527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2af38e2d8c7b63807c18e7205348cc93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2af38e2d8c7b63807c18e7205348cc93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adf39abd1b85df476c4e48aec0d9a7ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4844], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1cd757d9a7f893fe14457f49fc1ed90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([1222], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36f9988c825be420e92a28ce54fb0c39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebad67407114cd2888246deb2b826dc7
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33be465f5a1a52ea9e591e283e395409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6bc84b8da8686872f3a6e77f20c2601f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6bc84b8da8686872f3a6e77f20c2601f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17ae187e858d9cf95af4698924171589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0016308426856994629, 0.060334742069244385, 0.2727578282356262, 0.310065895318985], [0.45099326968193054, 0.26720455288887024, 0.20039832592010498, 0.24815897643566132], [0.45099326968193054, 0.26720455288887024, 0.20039832592010498, 0.24815897643566132], [0.04101485013961792, 0.1577802300453186, 0.34968137741088867, 0.37287673354148865], [0.2715381979942322, 0.30918967723846436, 0.12866002321243286, 0.1173793226480484], [0.04120466113090515, 0.29457390308380127, 0.18790428340435028, 0.2041577696800232]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_347351c4052d55ea212d962433a77e40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a46897f450a7b133e92d7fd0864ad502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347351c4052d55ea212d962433a77e40
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a7e08bca6ef5b9395a1c66b765eab6ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55055cf689aa537db9ac3989b4e51f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7e08bca6ef5b9395a1c66b765eab6ba
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_10b6046c666418323127d683f805ec6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dbe55b5f22236ec45e2c125e506baaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b6046c666418323127d683f805ec6e
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_253f2243c39191e9468c3e7bb0943cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f73bb83fc2bd12e1a92edb7456809eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f73bb83fc2bd12e1a92edb7456809eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_544ff788c6cf5b7ebf62ec54ee9d0c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9920dcd5911bb357a4444d500ecfec61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9920dcd5911bb357a4444d500ecfec61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0554bbbb9190a7fdeb438f1e3082b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30a6fb81b65652989ed626682e8a60cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30a6fb81b65652989ed626682e8a60cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_157735f0fc8298cb82be053c539afb36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ecd6d3da0633117aa68085eb55e58e40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe41bc4150a89fbeedc50e71dc034097(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd6d3da0633117aa68085eb55e58e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_71fcd388c0490d458a059efab7644a00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45c827a7ac8be3216c30c8081aeb3632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71fcd388c0490d458a059efab7644a00
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9febe40f6b3a88feb19e03842bb1071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_353023a9e1777c9fb64f3d4114b0d44e
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2820a1babfca51f376726ae7f270f324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16750776767730713, 0.08662114292383194, 0.1243920549750328, 0.035129185765981674, 0.08135896176099777, 0.20512865483760834, 0.18824024498462677, 0.15323543548583984, 0.2670220136642456, 0.22195294499397278, 0.1483171284198761, 0.17248742282390594, 0.1757042109966278, 0.19226054847240448, 0.26242178678512573, 0.16895940899848938, 0.1408797949552536, 0.16145479679107666, 0.16431079804897308, 0.10177677869796753], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa62352f55a07ad2f6a2b5e5ad939a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([17485], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe41bc4150a89fbeedc50e71dc034097(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd6d3da0633117aa68085eb55e58e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_067fe73dc7ed04ffa49a44e06df2f2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_820d9d57504738da6fe66500d5533b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f763aad3da82bd205a307cff7d2a7469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9eb786f2a2889a677c5964b6933cac80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9eb786f2a2889a677c5964b6933cac80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e177028f6bbdd633baf19a6ccc427651(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8146c68152bd110a924d07031de2b6cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e177028f6bbdd633baf19a6ccc427651
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_966a7dde525286f5a72039c06600fbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f048c28be4ba1c5d47ffb606ad50881f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10553447902202606, 0.11684665083885193, 0.1818733811378479, 0.3978734314441681], [0.25485116243362427, 0.1162174642086029, 0.0593176893889904, 0.05306103825569153], [0.10366234183311462, 0.2941455841064453, 0.12943437695503235, 0.22681641578674316], [0.10366234183311462, 0.2941455841064453, 0.12943437695503235, 0.22681641578674316], [0.26040518283843994, 0.1860140562057495, 0.12527427077293396, 0.01226457953453064]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55055cf689aa537db9ac3989b4e51f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7e08bca6ef5b9395a1c66b765eab6ba
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b64a20a0d0b6b81fb55369237a0bd9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac42641436898515102dc216a48bbc54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2900eb4c54d8be40362ea07d7499409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8760fc00fdd043699fe994dd526ce471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b03b7d7ebacf67a2059818b8b48fdc86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b03b7d7ebacf67a2059818b8b48fdc86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78204f2718f7e6ea26b16ef2f65017fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2249218076467514, 0.06978408992290497, 0.012984782457351685, 0.022978395223617554], [0.01998591423034668, 0.07889816164970398, 0.11353549361228943, 0.054215967655181885], [0.009033650159835815, 0.3637202978134155, 0.15329280495643616, 0.004585534334182739], [0.2249218076467514, 0.06978408992290497, 0.012984782457351685, 0.022978395223617554], [0.10977444052696228, 0.05127233266830444, 0.03762239217758179, 0.1816304624080658], [0.20305778086185455, 0.16725322604179382, 0.12122997641563416, 0.11242920905351639], [0.10977444052696228, 0.05127233266830444, 0.03762239217758179, 0.1816304624080658]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95ebdfded8b4e17099d5b78f4e2856e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45c827a7ac8be3216c30c8081aeb3632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71fcd388c0490d458a059efab7644a00
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_853876122220eea0d63c9159daec82fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c6a50fa25a59cac88eb22a14287833c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a09fa73bc2a2a5a5eb908a3f5ec99e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4321], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dd26db87d3126260ac6c5284eafed0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebad67407114cd2888246deb2b826dc7
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33927de077cdab2d962252d026ffb3c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e725f8f16dcc98865b16222e2aef71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e725f8f16dcc98865b16222e2aef71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6181c4c2644fdba3dac62d364c3a6aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0255865678191185, 0.006420324090868235]], [[8.320515917148441e-05, 0.000250508077442646]], [[0.0005496192607097328, 0.000657214957755059]], [[0.01133652776479721, 0.03480735048651695]], [[0.018511196598410606, 0.00013180857058614492]], [[0.0006387664470821619, 0.05736129358410835]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8646c9c92a6b9d09276b1b12f283c3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.07639110833406448, 0.09174058586359024]], [[0.020811760798096657, 7.844271749490872e-06]], [[0.005242577288299799, 0.006416682153940201]], [[0.007183857727795839, 0.028547251597046852]], [[0.015565316192805767, 0.12092229723930359]], [[0.09130977839231491, 0.022253159433603287]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41c073ede18dc36ce782f3dfd37d123d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c036f2658b02a50892fd7f596243f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1969531625509262, 0.27510517835617065, 0.13978993892669678, 0.057206202298402786, 0.01287273783236742, 0.16891586780548096, 0.23066169023513794, 0.08330577611923218, 0.16629137098789215, 0.21041962504386902, 0.07738223671913147, 0.2509373426437378, 0.058800287544727325, 0.1671455204486847, 0.11810829490423203, 0.17455364763736725], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8502292ffd745351eba85a39c28e06a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8691ccc3f3cbe682d32db5370c1fd112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acff5bba89495967566948b9e27d00e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc58357d1c8bc9d89a25daa555028727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a25a0b758fc1039f5b9d9222bb2b333f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a25a0b758fc1039f5b9d9222bb2b333f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eff2da3f63346f42401b6d4315c7f25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36792793f6b04e7040182e91be12aa3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2154858112335205, 0.11849692463874817, 0.04890155792236328, 0.2522619366645813], [0.04421807825565338, 0.3680357336997986, 0.3360503315925598, 0.17184141278266907], [0.332043319940567, 0.31710129976272583, 0.010255128145217896, 0.006749331951141357], [0.32686784863471985, 0.3103213608264923, 0.13777750730514526, 0.22633054852485657], [0.04447308927774429, 0.35153087973594666, 0.08885534107685089, 0.0066995322704315186]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c24d4a1aeec5dd83fd1965f4125950d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf15628d2a7935d10c4fbf1bb71d5579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3231605291366577, 0.04903687536716461, 0.07419966906309128, 0.03340575098991394], [0.044831693172454834, 0.05826646089553833, 0.2503412365913391, 0.11409053206443787], [0.20238913595676422, 0.10533830523490906, 0.1425924301147461, 0.17674031853675842], [0.044831693172454834, 0.05826646089553833, 0.2503412365913391, 0.11409053206443787], [0.20238913595676422, 0.10533830523490906, 0.1425924301147461, 0.17674031853675842]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1bd865dfe9ef91ef80c8c0838d5db4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebad67407114cd2888246deb2b826dc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d8e23a214aab71809c8444277716859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6720733621d002617de64c06181722f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6720733621d002617de64c06181722f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ef402b0bd6dda396ccd49846c31bffb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1390954852104187, 0.033391281962394714, 0.18100443482398987, 0.15556922554969788], [0.0509207546710968, 0.0248238667845726, 0.15702444314956665, 0.0630301833152771], [0.1339416205883026, 0.2178063988685608, 0.03409586474299431, 0.23763121664524078], [0.0509207546710968, 0.0248238667845726, 0.15702444314956665, 0.0630301833152771], [0.1339416205883026, 0.2178063988685608, 0.03409586474299431, 0.23763121664524078], [0.26555946469306946, 0.12011370062828064, 0.135416179895401, 0.2158403843641281], [0.26555946469306946, 0.12011370062828064, 0.135416179895401, 0.2158403843641281]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54936e165dd69b02828cad09c77012eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4151e62d366d2d9f35f80f1a184f4597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f88d7b6c1b03048c675c3dec62346c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc58357d1c8bc9d89a25daa555028727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a8ea77bdf32e92ffd95bb29853a2009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a8ea77bdf32e92ffd95bb29853a2009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eff2da3f63346f42401b6d4315c7f25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_554fe47f90c849d9de2c9dd1f5604882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36f4d2c495f778ca15b694763642834f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1585461050271988, 0.2030963897705078, 0.07531548291444778, 0.06819995492696762, 0.06903404742479324, 0.18677054345607758, 0.15702764689922333, 0.06769457459449768, 0.16080789268016815, 0.10624753683805466, 0.027685679495334625, 0.21551458537578583, 0.03189590200781822, 0.16047513484954834, 0.23134367167949677, 0.2134716957807541, 0.05273635685443878, 0.18173585832118988, 0.07269302010536194, 0.0032052749302238226, 0.19810013473033905, 0.1544133722782135, 0.22863608598709106, 0.22110961377620697], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28b34ad5d1846cebf0f7cd1c018b8c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90006b2554b03997b4029944bc2572de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90006b2554b03997b4029944bc2572de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34b1a3bc6b75cffe7654c4e19196680c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33f85b4631df586640a0b67e474ec943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24059699475765228, 0.20401248335838318, 0.20136485993862152, 0.07779122143983841], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26056e24fee0c96620b3a7f0d4e4ac5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.443401038646698, 0.01879914104938507, 0.030096083879470825, 0.05628645420074463], [0.1597653031349182, 0.03256210684776306, 0.04690167307853699, 0.3367536962032318], [0.39223048090934753, 0.10811522603034973, 0.1506238430738449, 0.11638037115335464], [0.10637867450714111, 0.045472901314496994, 0.35786545276641846, 0.22567766904830933], [0.10637867450714111, 0.045472901314496994, 0.35786545276641846, 0.22567766904830933], [0.39223048090934753, 0.10811522603034973, 0.1506238430738449, 0.11638037115335464]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62013ff1d54ec28cf7015b82b77fca7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1487407386302948, 0.2564408779144287, 0.11654667556285858, 0.08824345469474792], [0.04823960363864899, 0.033418506383895874, 0.005522549152374268, 0.1534278392791748], [0.245321124792099, 0.19668203592300415, 0.02901521325111389, 0.08745982497930527], [0.02178362011909485, 0.023879624903202057, 0.16128362715244293, 0.22988560795783997], [0.1487407386302948, 0.2564408779144287, 0.11654667556285858, 0.08824345469474792]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc46a9926d338223304783a1e281272e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34b1a3bc6b75cffe7654c4e19196680c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fad840c4fa5a924d45fc52909021164b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3725680708885193, 0.014811739325523376, 0.08545148372650146, 0.3024505078792572], [0.17571653425693512, 0.4427066147327423, 0.16599597036838531, 0.1372987926006317], [0.2488720715045929, 0.04982655495405197, 0.06996200978755951, 0.3496866524219513], [0.04939615726470947, 0.05756248161196709, 0.2488795518875122, 0.1573629081249237]], dtype='float32').reshape([4, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c906e6ddb17e1e38fc07834efd0d9b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_380643f43a83add64561a2f1f1295527
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_554fe47f90c849d9de2c9dd1f5604882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef012bb61ab49ae618b23b581beb16b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d2d313922bcf6debd7c96ab53cbe884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34964b306f6aff3eef08681a2fb88f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc8f1547af4dc44b51b16e846b0d7765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39fdf738bc7896de84ff3278d2c021c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39fdf738bc7896de84ff3278d2c021c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11c0443456334565a37ed7f954128799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_380643f43a83add64561a2f1f1295527
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6b5c19d807191d874810418f8c2a60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2546539306640625, 0.08790834248065948, 0.16270142793655396, 0.00904536247253418], [0.2546539306640625, 0.08790834248065948, 0.16270142793655396, 0.00904536247253418], [0.0754576176404953, 0.21234965324401855, 0.06143989413976669, 0.032623033970594406], [0.1800021082162857, 0.2203885167837143, 0.05525475740432739, 0.016759246587753296], [0.13451582193374634, 0.16116155683994293, 0.04648391902446747, 0.3108064532279968], [0.10491010546684265, 0.12081600725650787, 0.38842397928237915, 0.25860774517059326], [0.044446662068367004, 0.09552852809429169, 0.39697954058647156, 0.05195990204811096]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d8570e694ab13a7dd129024770c7527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1424a11252a16f6eac8d031465167413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1424a11252a16f6eac8d031465167413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adf39abd1b85df476c4e48aec0d9a7ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4844], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1cd757d9a7f893fe14457f49fc1ed90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([1222], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36f9988c825be420e92a28ce54fb0c39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebad67407114cd2888246deb2b826dc7
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33be465f5a1a52ea9e591e283e395409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d2684e291431abeb883e22ee8e94664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d2684e291431abeb883e22ee8e94664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17ae187e858d9cf95af4698924171589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0016308426856994629, 0.060334742069244385, 0.2727578282356262, 0.310065895318985], [0.45099326968193054, 0.26720455288887024, 0.20039832592010498, 0.24815897643566132], [0.45099326968193054, 0.26720455288887024, 0.20039832592010498, 0.24815897643566132], [0.04101485013961792, 0.1577802300453186, 0.34968137741088867, 0.37287673354148865], [0.2715381979942322, 0.30918967723846436, 0.12866002321243286, 0.1173793226480484], [0.04120466113090515, 0.29457390308380127, 0.18790428340435028, 0.2041577696800232]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b82c20508a3746fdfe2b256b33935690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e3ba821d800f6f00c78e8d01727cf98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f882a4a38e4241625cedaf8ace8f1902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_253f2243c39191e9468c3e7bb0943cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b726af5670646cc87e43ada0593ac9ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b726af5670646cc87e43ada0593ac9ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_544ff788c6cf5b7ebf62ec54ee9d0c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68636fbf4b345bfdef62a81033f85907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68636fbf4b345bfdef62a81033f85907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0554bbbb9190a7fdeb438f1e3082b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90d11f6577e14f5e7370f68f75141287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90d11f6577e14f5e7370f68f75141287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_157735f0fc8298cb82be053c539afb36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70db61c3ddd2476f482f20c931083a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69418e20dda696d6b8e63fc16e67e059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4151e62d366d2d9f35f80f1a184f4597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2820a1babfca51f376726ae7f270f324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16750776767730713, 0.08662114292383194, 0.1243920549750328, 0.035129185765981674, 0.08135896176099777, 0.20512865483760834, 0.18824024498462677, 0.15323543548583984, 0.2670220136642456, 0.22195294499397278, 0.1483171284198761, 0.17248742282390594, 0.1757042109966278, 0.19226054847240448, 0.26242178678512573, 0.16895940899848938, 0.1408797949552536, 0.16145479679107666, 0.16431079804897308, 0.10177677869796753], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa62352f55a07ad2f6a2b5e5ad939a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([17485], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70db61c3ddd2476f482f20c931083a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_067fe73dc7ed04ffa49a44e06df2f2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_820d9d57504738da6fe66500d5533b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f763aad3da82bd205a307cff7d2a7469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_466d1fb9355fa79d823c8f32bb3b32be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_466d1fb9355fa79d823c8f32bb3b32be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_267a49e9e4ec3ddab8b36e1230e1a369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_966a7dde525286f5a72039c06600fbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f048c28be4ba1c5d47ffb606ad50881f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10553447902202606, 0.11684665083885193, 0.1818733811378479, 0.3978734314441681], [0.25485116243362427, 0.1162174642086029, 0.0593176893889904, 0.05306103825569153], [0.10366234183311462, 0.2941455841064453, 0.12943437695503235, 0.22681641578674316], [0.10366234183311462, 0.2941455841064453, 0.12943437695503235, 0.22681641578674316], [0.26040518283843994, 0.1860140562057495, 0.12527427077293396, 0.01226457953453064]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e3ba821d800f6f00c78e8d01727cf98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b64a20a0d0b6b81fb55369237a0bd9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac42641436898515102dc216a48bbc54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2900eb4c54d8be40362ea07d7499409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8760fc00fdd043699fe994dd526ce471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d38c774cd4b3eac4a63815590bc72153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d38c774cd4b3eac4a63815590bc72153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78204f2718f7e6ea26b16ef2f65017fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2249218076467514, 0.06978408992290497, 0.012984782457351685, 0.022978395223617554], [0.01998591423034668, 0.07889816164970398, 0.11353549361228943, 0.054215967655181885], [0.009033650159835815, 0.3637202978134155, 0.15329280495643616, 0.004585534334182739], [0.2249218076467514, 0.06978408992290497, 0.012984782457351685, 0.022978395223617554], [0.10977444052696228, 0.05127233266830444, 0.03762239217758179, 0.1816304624080658], [0.20305778086185455, 0.16725322604179382, 0.12122997641563416, 0.11242920905351639], [0.10977444052696228, 0.05127233266830444, 0.03762239217758179, 0.1816304624080658]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95ebdfded8b4e17099d5b78f4e2856e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69418e20dda696d6b8e63fc16e67e059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()



if __name__ == '__main__':
    unittest.main()