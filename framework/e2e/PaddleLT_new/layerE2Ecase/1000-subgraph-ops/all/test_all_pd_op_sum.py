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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
class TestPrimitiveOp_a7854004993275e3ea7b7df9afc30bc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4370], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b942587d210077fa7d01be09d668265f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.11252374202013016, 0.06671373546123505]], [[0.0762287974357605, 1.3797877045362839e-06]], [[0.029863575473427773, 0.0016860900213941932]], [[0.010046962648630142, 0.16704696416854858]], [[0.15594160556793213, 0.011682684533298016]], [[0.03338748961687088, 0.013110277242958546]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb1482c21f0ee1890da6417dd6fc2541(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.04719573259353638, 0.007316963747143745]], [[0.028195437043905258, 0.0014178178971633315]], [[0.008944532833993435, 0.16810110211372375]], [[0.018861278891563416, 0.0042112418450415134]], [[0.0997413918375969, 6.0834619944216684e-05]], [[0.07123599201440811, 0.04520457237958908]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d05b572890a502512eefe79f0e004b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1386614739894867, 0.08434511721134186, 0.011653227731585503, 0.11889123171567917, 0.0234603900462389, 0.09853493422269821, 0.11342005431652069, 0.013193347491323948, 0.12249384820461273, 0.15950527787208557, 0.07128115743398666, 0.08687371015548706, 0.17103254795074463, 0.08307117968797684, 0.2077733427286148, 0.02830277569591999], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
class TestPrimitiveOp_c5fb65c6e17b73d0bfe681c7ab1e198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5fb65c6e17b73d0bfe681c7ab1e198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c54726695ffa475234e6396c37fe966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19972366094589233, 0.032638389617204666, 0.1985551118850708, 0.06861022114753723], [0.13500988483428955, 0.29951679706573486, 0.36281725764274597, 0.19303631782531738], [0.04201158881187439, 0.07326073944568634, 0.09581686556339264, 0.07714217901229858], [0.024873018264770508, 0.24633540213108063, 0.13596563041210175, 0.446040540933609], [0.29017892479896545, 0.0497107058763504, 0.023367710411548615, 0.18428494036197662]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e588604f12da32f1409a557e5f88716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.043433159589767456, 0.03012770414352417, 0.002067267894744873, 0.15429961681365967], [0.12482473254203796, 0.0377102792263031, 0.18082445859909058, 0.38631772994995117], [0.08110913634300232, 0.039556220173835754, 0.4196973443031311, 0.026297882199287415], [0.12482473254203796, 0.0377102792263031, 0.18082445859909058, 0.38631772994995117], [0.08110913634300232, 0.039556220173835754, 0.4196973443031311, 0.026297882199287415]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c423e6a28932dc341cd905df5816a2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c423e6a28932dc341cd905df5816a2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68f0a8a44a03918343799ca58fc98f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1041974127292633, 0.09992450475692749, 0.16216400265693665, 0.13433204591274261], [0.2555374205112457, 0.2790604829788208, 0.16100162267684937, 0.31656771898269653], [0.16396866738796234, 0.1870870143175125, 0.17119094729423523, 0.20861756801605225], [0.2555374205112457, 0.2790604829788208, 0.16100162267684937, 0.31656771898269653], [0.16396866738796234, 0.1870870143175125, 0.17119094729423523, 0.20861756801605225], [0.2263275384902954, 0.0113314688205719, 0.01187238097190857, 0.2862564027309418], [0.2263275384902954, 0.0113314688205719, 0.01187238097190857, 0.2862564027309418]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_856d399859d49668c1c7e88b657a3de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_856d399859d49668c1c7e88b657a3de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c861d2f8403c9e23cc401d2d7a04e410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14734989404678345, 0.14975415170192719, 0.2101060301065445, 0.017528915777802467, 0.062227215617895126, 0.268299400806427, 0.0712365135550499, 0.1437893658876419, 0.1381756216287613, 0.18070535361766815, 0.004081408958882093, 0.05884236469864845, 0.13376963138580322, 0.23385003209114075, 0.06353780627250671, 0.11467491090297699, 0.16991066932678223, 0.1861364096403122, 0.07073098421096802, 0.1491689234972, 0.2388898879289627, 0.18919293582439423, 0.17250730097293854, 0.09327930957078934], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da5ac59207af0398824e5e02ee8f8481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da5ac59207af0398824e5e02ee8f8481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ceeea9a9847bb2227e4c16b9cd5afc9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01943795569241047, 0.10749263316392899, 0.07608206570148468, 0.1653502732515335], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f2971645e6232fc0121eb9035da8018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05351126194000244, 0.015865057706832886, 0.16170693933963776, 0.28991347551345825], [0.35589900612831116, 0.24598413705825806, 0.008338751271367073, 0.056291863322257996], [0.08802980184555054, 0.14576570689678192, 0.31650105118751526, 0.22346355020999908], [0.3839932680130005, 0.3838215172290802, 0.10680212080478668, 0.1898486465215683], [0.3839932680130005, 0.3838215172290802, 0.10680212080478668, 0.1898486465215683], [0.08802980184555054, 0.14576570689678192, 0.31650105118751526, 0.22346355020999908]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aea9bf9495a990b7c2e909e7635f7d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.015219897031784058, 0.18850725889205933, 0.06158987432718277, 0.18032580614089966], [0.4155067205429077, 0.35695895552635193, 0.38538241386413574, 0.19103789329528809], [0.19356802105903625, 0.23713263869285583, 0.19931265711784363, 0.08076208829879761], [0.3319634199142456, 0.0276150181889534, 0.12497778236865997, 0.12103688716888428], [0.015219897031784058, 0.18850725889205933, 0.06158987432718277, 0.18032580614089966]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93a2bae60764f765ddec0e281ccc8bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13057728111743927, 0.04129461944103241, 0.008406758308410645, 0.042439982295036316], [0.033329714089632034, 0.055771782994270325, 0.17385263741016388, 0.01043766736984253], [0.06733880937099457, 0.24477678537368774, 0.039074793457984924, 0.024114012718200684], [0.37067317962646484, 0.33388274908065796, 0.4611213207244873, 0.23863281309604645]], dtype='float32').reshape([4, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b14e1b0c2affdd5d1d802c97b3329b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b14e1b0c2affdd5d1d802c97b3329b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d408b38a448e29fcb263d4b77cfb250(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19421494007110596, 0.3640298545360565, 0.25289735198020935, 0.10513561964035034], [0.19421494007110596, 0.3640298545360565, 0.25289735198020935, 0.10513561964035034], [0.012431785464286804, 0.03469005227088928, 0.4617830514907837, 0.04844389855861664], [0.06334614753723145, 0.24080072343349457, 0.2880927622318268, 0.21283990144729614], [0.101856529712677, 0.1371634155511856, 0.47256016731262207, 0.3744075298309326], [0.07248750329017639, 0.06572168320417404, 0.06581896543502808, 0.0928030014038086], [0.0532781183719635, 0.13677886128425598, 0.18745388090610504, 0.04443269968032837]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18882a4d616fbc0fe8b8eb954ef0d4bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18882a4d616fbc0fe8b8eb954ef0d4bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbd680abc27887a853997ce547678873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4899], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91b2a0ef30cccc9bbd43caad968a84ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([1237], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9934ffa770d9887d770d6951c689a554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9934ffa770d9887d770d6951c689a554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f3da44a05167f8948685e261bc82a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.016722746193408966, 0.1188654750585556, 0.16593235731124878, 0.4197741150856018], [0.26169371604919434, 0.24232712388038635, 0.10179534554481506, 0.1798313558101654], [0.26169371604919434, 0.24232712388038635, 0.10179534554481506, 0.1798313558101654], [0.2145078182220459, 0.016424626111984253, 0.09890654683113098, 0.38689732551574707], [0.2170209437608719, 0.2632460594177246, 0.07790133357048035, 0.2807905972003937], [0.006587892770767212, 0.21586290001869202, 0.15675637125968933, 0.036545753479003906]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_148c4d67bdc798afddab5db505021530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_148c4d67bdc798afddab5db505021530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23c5b4dbf070832fa0ca708d876e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23c5b4dbf070832fa0ca708d876e19f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc787f447383349ec3c5a9d623749f4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc787f447383349ec3c5a9d623749f4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7cce58ec40a48df3e28e8c30c6e1f1b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1866828054189682, 0.2055896371603012, 0.15543478727340698, 0.08194529265165329, 0.2010299563407898, 0.1310936063528061, 0.24182313680648804, 0.2366103082895279, 0.18278363347053528, 0.24047572910785675, 0.05516314134001732, 0.25647616386413574, 0.10634701699018478, 0.02525392360985279, 0.03024829365313053, 0.11830463260412216, 0.21684667468070984, 0.1802818328142166, 0.19211024045944214, 0.1744338870048523], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_075d536de240219960421c245ad339b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([17294], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_679000f520a1d37b2bbe55e3ec470377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_679000f520a1d37b2bbe55e3ec470377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f73bec71d8565cabc8bca8969a97394d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.026640087366104126, 0.08988994359970093, 0.26466962695121765, 0.33565273880958557], [0.28288114070892334, 0.06286081671714783, 0.09939444810152054, 0.059776730835437775], [0.0559675469994545, 0.027802541851997375, 0.2006770223379135, 0.403134822845459], [0.0559675469994545, 0.027802541851997375, 0.2006770223379135, 0.403134822845459], [0.15815800428390503, 0.2873311936855316, 0.21028625965118408, 0.3253607451915741]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fb6712b91d3fd28ce1a06b059e57737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fb6712b91d3fd28ce1a06b059e57737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef8a1544e517a57091c1d7dfa7809c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06515082716941833, 0.059610962867736816, 0.06099565327167511, 0.22025857865810394], [0.06051965802907944, 0.029050052165985107, 0.07764720916748047, 0.08721211552619934], [0.15354707837104797, 0.07181882858276367, 0.14888903498649597, 0.18712368607521057], [0.06515082716941833, 0.059610962867736816, 0.06099565327167511, 0.22025857865810394], [0.2073751986026764, 0.08540217578411102, 0.2514403462409973, 0.1407393515110016], [0.32810941338539124, 0.13238462805747986, 0.04331880807876587, 0.04566563665866852], [0.2073751986026764, 0.08540217578411102, 0.2514403462409973, 0.1407393515110016]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7854004993275e3ea7b7df9afc30bc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4370], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b942587d210077fa7d01be09d668265f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.11252374202013016, 0.06671373546123505]], [[0.0762287974357605, 1.3797877045362839e-06]], [[0.029863575473427773, 0.0016860900213941932]], [[0.010046962648630142, 0.16704696416854858]], [[0.15594160556793213, 0.011682684533298016]], [[0.03338748961687088, 0.013110277242958546]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb1482c21f0ee1890da6417dd6fc2541(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.04719573259353638, 0.007316963747143745]], [[0.028195437043905258, 0.0014178178971633315]], [[0.008944532833993435, 0.16810110211372375]], [[0.018861278891563416, 0.0042112418450415134]], [[0.0997413918375969, 6.0834619944216684e-05]], [[0.07123599201440811, 0.04520457237958908]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d05b572890a502512eefe79f0e004b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1386614739894867, 0.08434511721134186, 0.011653227731585503, 0.11889123171567917, 0.0234603900462389, 0.09853493422269821, 0.11342005431652069, 0.013193347491323948, 0.12249384820461273, 0.15950527787208557, 0.07128115743398666, 0.08687371015548706, 0.17103254795074463, 0.08307117968797684, 0.2077733427286148, 0.02830277569591999], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6844b92c75a5ec3341e3e49b5a137a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6844b92c75a5ec3341e3e49b5a137a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c54726695ffa475234e6396c37fe966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19972366094589233, 0.032638389617204666, 0.1985551118850708, 0.06861022114753723], [0.13500988483428955, 0.29951679706573486, 0.36281725764274597, 0.19303631782531738], [0.04201158881187439, 0.07326073944568634, 0.09581686556339264, 0.07714217901229858], [0.024873018264770508, 0.24633540213108063, 0.13596563041210175, 0.446040540933609], [0.29017892479896545, 0.0497107058763504, 0.023367710411548615, 0.18428494036197662]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e588604f12da32f1409a557e5f88716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.043433159589767456, 0.03012770414352417, 0.002067267894744873, 0.15429961681365967], [0.12482473254203796, 0.0377102792263031, 0.18082445859909058, 0.38631772994995117], [0.08110913634300232, 0.039556220173835754, 0.4196973443031311, 0.026297882199287415], [0.12482473254203796, 0.0377102792263031, 0.18082445859909058, 0.38631772994995117], [0.08110913634300232, 0.039556220173835754, 0.4196973443031311, 0.026297882199287415]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa760e7566ad38d6bd1c4cf88c236f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa760e7566ad38d6bd1c4cf88c236f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68f0a8a44a03918343799ca58fc98f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1041974127292633, 0.09992450475692749, 0.16216400265693665, 0.13433204591274261], [0.2555374205112457, 0.2790604829788208, 0.16100162267684937, 0.31656771898269653], [0.16396866738796234, 0.1870870143175125, 0.17119094729423523, 0.20861756801605225], [0.2555374205112457, 0.2790604829788208, 0.16100162267684937, 0.31656771898269653], [0.16396866738796234, 0.1870870143175125, 0.17119094729423523, 0.20861756801605225], [0.2263275384902954, 0.0113314688205719, 0.01187238097190857, 0.2862564027309418], [0.2263275384902954, 0.0113314688205719, 0.01187238097190857, 0.2862564027309418]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_725c113bba128dfd2087f1880d360fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_725c113bba128dfd2087f1880d360fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c861d2f8403c9e23cc401d2d7a04e410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14734989404678345, 0.14975415170192719, 0.2101060301065445, 0.017528915777802467, 0.062227215617895126, 0.268299400806427, 0.0712365135550499, 0.1437893658876419, 0.1381756216287613, 0.18070535361766815, 0.004081408958882093, 0.05884236469864845, 0.13376963138580322, 0.23385003209114075, 0.06353780627250671, 0.11467491090297699, 0.16991066932678223, 0.1861364096403122, 0.07073098421096802, 0.1491689234972, 0.2388898879289627, 0.18919293582439423, 0.17250730097293854, 0.09327930957078934], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6da8da119dfe86c3c674acfec94aebe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6da8da119dfe86c3c674acfec94aebe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ceeea9a9847bb2227e4c16b9cd5afc9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01943795569241047, 0.10749263316392899, 0.07608206570148468, 0.1653502732515335], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f2971645e6232fc0121eb9035da8018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05351126194000244, 0.015865057706832886, 0.16170693933963776, 0.28991347551345825], [0.35589900612831116, 0.24598413705825806, 0.008338751271367073, 0.056291863322257996], [0.08802980184555054, 0.14576570689678192, 0.31650105118751526, 0.22346355020999908], [0.3839932680130005, 0.3838215172290802, 0.10680212080478668, 0.1898486465215683], [0.3839932680130005, 0.3838215172290802, 0.10680212080478668, 0.1898486465215683], [0.08802980184555054, 0.14576570689678192, 0.31650105118751526, 0.22346355020999908]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aea9bf9495a990b7c2e909e7635f7d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.015219897031784058, 0.18850725889205933, 0.06158987432718277, 0.18032580614089966], [0.4155067205429077, 0.35695895552635193, 0.38538241386413574, 0.19103789329528809], [0.19356802105903625, 0.23713263869285583, 0.19931265711784363, 0.08076208829879761], [0.3319634199142456, 0.0276150181889534, 0.12497778236865997, 0.12103688716888428], [0.015219897031784058, 0.18850725889205933, 0.06158987432718277, 0.18032580614089966]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93a2bae60764f765ddec0e281ccc8bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13057728111743927, 0.04129461944103241, 0.008406758308410645, 0.042439982295036316], [0.033329714089632034, 0.055771782994270325, 0.17385263741016388, 0.01043766736984253], [0.06733880937099457, 0.24477678537368774, 0.039074793457984924, 0.024114012718200684], [0.37067317962646484, 0.33388274908065796, 0.4611213207244873, 0.23863281309604645]], dtype='float32').reshape([4, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf23041cf30f0b1fbbbdea22e1ec2c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf23041cf30f0b1fbbbdea22e1ec2c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d408b38a448e29fcb263d4b77cfb250(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19421494007110596, 0.3640298545360565, 0.25289735198020935, 0.10513561964035034], [0.19421494007110596, 0.3640298545360565, 0.25289735198020935, 0.10513561964035034], [0.012431785464286804, 0.03469005227088928, 0.4617830514907837, 0.04844389855861664], [0.06334614753723145, 0.24080072343349457, 0.2880927622318268, 0.21283990144729614], [0.101856529712677, 0.1371634155511856, 0.47256016731262207, 0.3744075298309326], [0.07248750329017639, 0.06572168320417404, 0.06581896543502808, 0.0928030014038086], [0.0532781183719635, 0.13677886128425598, 0.18745388090610504, 0.04443269968032837]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dbc4f31dc41af642511277be4f9d602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dbc4f31dc41af642511277be4f9d602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbd680abc27887a853997ce547678873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4899], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91b2a0ef30cccc9bbd43caad968a84ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([1237], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dd647722b46e32a875274f6cee0fb68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dd647722b46e32a875274f6cee0fb68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f3da44a05167f8948685e261bc82a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.016722746193408966, 0.1188654750585556, 0.16593235731124878, 0.4197741150856018], [0.26169371604919434, 0.24232712388038635, 0.10179534554481506, 0.1798313558101654], [0.26169371604919434, 0.24232712388038635, 0.10179534554481506, 0.1798313558101654], [0.2145078182220459, 0.016424626111984253, 0.09890654683113098, 0.38689732551574707], [0.2170209437608719, 0.2632460594177246, 0.07790133357048035, 0.2807905972003937], [0.006587892770767212, 0.21586290001869202, 0.15675637125968933, 0.036545753479003906]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f021d29a94f6745f664eddcf86281e24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f021d29a94f6745f664eddcf86281e24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6afcc94e3e23efc8a7272ac5e834942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6afcc94e3e23efc8a7272ac5e834942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c7ec93c1f35ced1f1949ddf3a5f111e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c7ec93c1f35ced1f1949ddf3a5f111e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7cce58ec40a48df3e28e8c30c6e1f1b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1866828054189682, 0.2055896371603012, 0.15543478727340698, 0.08194529265165329, 0.2010299563407898, 0.1310936063528061, 0.24182313680648804, 0.2366103082895279, 0.18278363347053528, 0.24047572910785675, 0.05516314134001732, 0.25647616386413574, 0.10634701699018478, 0.02525392360985279, 0.03024829365313053, 0.11830463260412216, 0.21684667468070984, 0.1802818328142166, 0.19211024045944214, 0.1744338870048523], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_075d536de240219960421c245ad339b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([17294], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de79347e0ae42d5073bd8165a7e32074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de79347e0ae42d5073bd8165a7e32074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f73bec71d8565cabc8bca8969a97394d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.026640087366104126, 0.08988994359970093, 0.26466962695121765, 0.33565273880958557], [0.28288114070892334, 0.06286081671714783, 0.09939444810152054, 0.059776730835437775], [0.0559675469994545, 0.027802541851997375, 0.2006770223379135, 0.403134822845459], [0.0559675469994545, 0.027802541851997375, 0.2006770223379135, 0.403134822845459], [0.15815800428390503, 0.2873311936855316, 0.21028625965118408, 0.3253607451915741]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8d748606fcd2c39e9068e1ec6bcfb02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8d748606fcd2c39e9068e1ec6bcfb02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef8a1544e517a57091c1d7dfa7809c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06515082716941833, 0.059610962867736816, 0.06099565327167511, 0.22025857865810394], [0.06051965802907944, 0.029050052165985107, 0.07764720916748047, 0.08721211552619934], [0.15354707837104797, 0.07181882858276367, 0.14888903498649597, 0.18712368607521057], [0.06515082716941833, 0.059610962867736816, 0.06099565327167511, 0.22025857865810394], [0.2073751986026764, 0.08540217578411102, 0.2514403462409973, 0.1407393515110016], [0.32810941338539124, 0.13238462805747986, 0.04331880807876587, 0.04566563665866852], [0.2073751986026764, 0.08540217578411102, 0.2514403462409973, 0.1407393515110016]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()



if __name__ == '__main__':
    unittest.main()