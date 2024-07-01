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
            PADDLE_DEBUG_ENABLE_CINN=True,
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

last_cinn_stage_exit_code = None
def LastCINNStageFailed():
    global last_cinn_stage_exit_code
    if last_cinn_stage_exit_code is not None:
        return last_cinn_stage_exit_code != 0
    last_stage = GetPrevCinnStage(GetCurrentCinnStage())
    if last_stage is None:
        return False
    env_vars = dict(
        PADDLE_DEBUG_CINN_STAGE_NAME=last_stage.name,
        PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
    )
    env_vars_str = " ".join(
        f"{env_var}={value}"
        for env_var, value in env_vars.items()
    )
    last_cinn_stage_exit_code = os.system(
        f"{env_vars_str} {sys.executable} {__file__} > /dev/null 2>&1"
    )
    return last_cinn_stage_exit_code != 0

def SetDefaultEnv(**env_var2value):
    for env_var, value in env_var2value.items():
        if os.getenv(env_var) is None:
            os.environ[env_var] = str(value)

SetDefaultEnv(
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

    def test_train(self):
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





if not (IsCinnStageEnableDiff() and LastCINNStageFailed()):
    class PrimitiveOp_b66ec9373a6d99ceef26165ec39261d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bdfbabfb74f814b7110909926ab01d04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66ec9373a6d99ceef26165ec39261d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5e83bad42ad8b21e849e23cdce4bba67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f93dac1a118436b423d544a1e1eb0eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e83bad42ad8b21e849e23cdce4bba67
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.512653350830078, 5.044649124145508, 4.655038356781006, 4.622213363647461, 4.632609844207764, 3.9768052101135254, 5.023960113525391, 4.7768354415893555, 5.421357154846191, 4.686634063720703, 5.148271083831787, 3.8738279342651367, 5.151643753051758, 4.582184314727783, 4.700376987457275, 4.343968868255615, 4.77421760559082, 4.917938232421875]], dtype='float32').reshape([1, 18]),
            ]


    
    class PrimitiveOp_929cf8f24a8bbf836b553db1df582393(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89fe0adb358ae3c9dbeaa41f0f29a8c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_929cf8f24a8bbf836b553db1df582393
        def get_inputs(self):
            return [
                paddle.to_tensor([[6.194968223571777, 5.21784782409668, 6.012714385986328, 4.90752649307251, 5.907154083251953, 5.492164611816406, 5.724031448364258, 5.579870223999023, 5.25261116027832, 5.262575149536133, 4.954087257385254, 5.3679680824279785, 4.764451503753662, 5.245046615600586, 6.278081893920898, 5.532181262969971, 6.717957496643066, 5.762195110321045, 5.715753078460693, 5.413527965545654, 4.952664375305176, 5.497104167938232, 5.146821022033691]], dtype='float32').reshape([1, 23]),
            ]


    
    class PrimitiveOp_f5410dd7e21cc8a15b888720b3345ac7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_640bf5323d92ac0930ca434848e3d5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5410dd7e21cc8a15b888720b3345ac7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bb8e8ef19bcb85ef7d4a3087914586b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_00b2ee41fcd12aa1bfa4503ae47e537d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb8e8ef19bcb85ef7d4a3087914586b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8f401632ac5cce67cad70c8d908922d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_241529bfb613603de5461bc7f49e2ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f401632ac5cce67cad70c8d908922d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bbaca726853922d39ae3de53aafbbd53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b227f5017bd2f8016cbb8291590e7888(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e53dbaec81c4f1bf5ade08d45fc43abc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4fcdbb1a73afc04bf41dbb6d351c7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e53dbaec81c4f1bf5ade08d45fc43abc
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4fcdbb1a73afc04bf41dbb6d351c7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e53dbaec81c4f1bf5ade08d45fc43abc
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_314efc29fd280f3ef667441f59effd18(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_666fc2b2be7db47fb2592978084b415e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 30, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12dbb179e8c36930059de5102ec270c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.990804195404053]], [[7.303852081298828]], [[7.5863847732543945]], [[8.034000396728516]], [[6.682290077209473]], [[6.763601303100586]], [[8.005084037780762]], [[7.190065860748291]], [[7.1677021980285645]], [[7.97681188583374]], [[7.243475914001465]], [[7.156096935272217]], [[7.190069675445557]], [[7.3471455574035645]], [[7.202025890350342]], [[7.867246150970459]], [[7.899160861968994]], [[8.041866302490234]], [[7.189223289489746]], [[8.056235313415527]], [[7.894644737243652]], [[7.0927839279174805]], [[7.3161773681640625]], [[7.718397617340088]], [[6.824087142944336]], [[7.708375453948975]], [[8.410866737365723]], [[8.488816261291504]], [[7.6982855796813965]], [[7.605227947235107]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_c0a08cd263ed348af366d1ff7d703581(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_470d79ba68fa9475769130b6702b37fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0a08cd263ed348af366d1ff7d703581
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6fb00e03cfe1952d3693ed406da57458(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f6ef264095d1e1f1ca65ed510414bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f6ef264095d1e1f1ca65ed510414bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f6ef264095d1e1f1ca65ed510414bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f6ef264095d1e1f1ca65ed510414bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f6ef264095d1e1f1ca65ed510414bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f6ef264095d1e1f1ca65ed510414bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f6ef264095d1e1f1ca65ed510414bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f6ef264095d1e1f1ca65ed510414bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2413dbb4ac6c447ef4128742c1c5798d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2413dbb4ac6c447ef4128742c1c5798d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2413dbb4ac6c447ef4128742c1c5798d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2413dbb4ac6c447ef4128742c1c5798d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2413dbb4ac6c447ef4128742c1c5798d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2413dbb4ac6c447ef4128742c1c5798d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2413dbb4ac6c447ef4128742c1c5798d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2413dbb4ac6c447ef4128742c1c5798d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f636d8ed6303e0ec38193cb5b0e777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f636d8ed6303e0ec38193cb5b0e777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f636d8ed6303e0ec38193cb5b0e777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f636d8ed6303e0ec38193cb5b0e777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f636d8ed6303e0ec38193cb5b0e777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f636d8ed6303e0ec38193cb5b0e777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f636d8ed6303e0ec38193cb5b0e777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f636d8ed6303e0ec38193cb5b0e777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5baff9ff39a7b781eef683ac3ac3946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5baff9ff39a7b781eef683ac3ac3946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5baff9ff39a7b781eef683ac3ac3946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5baff9ff39a7b781eef683ac3ac3946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5baff9ff39a7b781eef683ac3ac3946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5baff9ff39a7b781eef683ac3ac3946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5baff9ff39a7b781eef683ac3ac3946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5baff9ff39a7b781eef683ac3ac3946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59f3ac3a2e265d12f8455f17a81bd531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.483332633972168]], [[8.580183982849121]], [[8.237578392028809]], [[8.947674751281738]], [[7.881694316864014]], [[8.806875228881836]], [[7.8643364906311035]], [[7.818826198577881]], [[8.723896026611328]], [[8.522080421447754]], [[7.446710586547852]], [[7.711270332336426]], [[8.313339233398438]], [[8.100343704223633]], [[7.585079669952393]], [[7.719268798828125]], [[8.0361967086792]], [[7.735347747802734]], [[8.298324584960938]], [[7.859841346740723]], [[8.472572326660156]], [[8.9426851272583]], [[7.678374767303467]], [[7.888156414031982]], [[8.077996253967285]], [[8.684389114379883]], [[7.907255172729492]], [[7.864846706390381]], [[7.371816635131836]], [[8.74284553527832]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_55867f88cb130136e51ea27a23dd042b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bebd043c1092eba0279d9881acf9ea4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 5, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6cdb59830568de8ec10e5e14bc871280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bebd043c1092eba0279d9881acf9ea4c
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.6099520921707153]], [[1.3218650817871094]], [[1.5464215278625488]], [[1.3357179164886475]], [[1.4156224727630615]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    
    class PrimitiveOp_7655ad88391ddaf1de39aa3e53d86fcd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df78f8d61fb638705c4dc9dbfbadb81e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7655ad88391ddaf1de39aa3e53d86fcd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.4961612224578857]], [[2.829477071762085]], [[2.832637071609497]], [[2.7194604873657227]], [[2.246098279953003]], [[2.6214306354522705]], [[3.1797657012939453]], [[3.231794834136963]], [[2.268321990966797]], [[2.6925604343414307]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    
    class PrimitiveOp_1ebea6d2caa4f21d58171dfaad3095ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06bb18e5c40db17a4459edca3cad7252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ebea6d2caa4f21d58171dfaad3095ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f26a146c29bc2055fb0949e503a5223(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08f12eeab7a0587552b57f8cff47285e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.255369186401367]], [[7.026007175445557]], [[6.470477104187012]], [[7.313347339630127]], [[6.378765106201172]], [[7.421102523803711]], [[6.025125980377197]], [[5.941402912139893]], [[7.002050399780273]], [[7.059742450714111]], [[7.259979724884033]], [[6.680147171020508]], [[6.630357265472412]], [[7.096652984619141]], [[6.853813171386719]], [[6.618465423583984]], [[7.474974155426025]], [[6.692164897918701]], [[7.260404586791992]], [[6.467423439025879]], [[7.05855655670166]], [[6.48403787612915]], [[7.445367813110352]], [[6.541408538818359]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_740184315fb12c1437b5d4710b4e6130(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e80eb019ef0c5559ffe2d487c9bd3239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_af91863c96e4c4ea3c47bc055891379a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8ee39994515679cae24f5f28d0d806d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af91863c96e4c4ea3c47bc055891379a
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 18, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44ef9814474a0327cb1448cd40010d5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.84508752822876]], [[5.073159694671631]], [[5.19683837890625]], [[5.763732433319092]], [[5.3940558433532715]], [[5.156362056732178]], [[5.496768474578857]], [[4.507992744445801]], [[5.311750888824463]], [[5.81207799911499]], [[5.574479103088379]], [[5.305727958679199]], [[5.405195713043213]], [[5.113437175750732]], [[4.843127250671387]], [[5.608423709869385]], [[5.8059773445129395]], [[4.789793491363525]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_06bb18e5c40db17a4459edca3cad7252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ebea6d2caa4f21d58171dfaad3095ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1918c78e824205e03cd6e44b62ddf19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.374038219451904]], [[6.828001022338867]], [[6.4136176109313965]], [[6.300159454345703]], [[7.29701042175293]], [[7.216587543487549]], [[6.84116792678833]], [[7.992212772369385]], [[6.857626438140869]], [[6.7683000564575195]], [[6.817885875701904]], [[7.325019359588623]], [[6.7759528160095215]], [[6.628880023956299]], [[6.700521945953369]], [[6.536348342895508]], [[6.566918849945068]], [[7.208926200866699]], [[6.991861343383789]], [[7.377551078796387]], [[6.411651611328125]], [[7.489564418792725]], [[5.967304229736328]], [[5.613235950469971]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_c09cbd6dace3c23c44414c33395f223b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0a08cd263ed348af366d1ff7d703581
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_829ae5fe88852d104a0d4812eb48c6d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_331c68240a16a39b8c14d67319da63b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fef74c380061f1dae53eebe4ce47b8f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_331c68240a16a39b8c14d67319da63b5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9948275089263916]], [[1.5264453887939453]], [[1.226925253868103]], [[1.1448898315429688]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_c09cbd6dace3c23c44414c33395f223b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0a08cd263ed348af366d1ff7d703581
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc2fb3a9a0470503530e9afaa2254eda(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 11, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbc59465fb9d533e266ef1343831b521(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc2fb3a9a0470503530e9afaa2254eda
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.9157357215881348]], [[2.75087308883667]], [[2.3742504119873047]], [[2.9814884662628174]], [[2.8051421642303467]], [[2.452035665512085]], [[2.8351681232452393]], [[2.9855198860168457]], [[2.973231077194214]], [[2.719639539718628]], [[2.89060378074646]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_640bf5323d92ac0930ca434848e3d5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5410dd7e21cc8a15b888720b3345ac7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06bb18e5c40db17a4459edca3cad7252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ebea6d2caa4f21d58171dfaad3095ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e833adcd102450e0a619d85e78d28cb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.217819213867188]], [[7.3988237380981445]], [[8.072800636291504]], [[8.118842124938965]], [[8.233335494995117]], [[8.393362998962402]], [[7.98402214050293]], [[7.8484787940979]], [[7.684823989868164]], [[7.703281402587891]], [[8.268752098083496]], [[8.193763732910156]], [[8.559454917907715]], [[7.790647029876709]], [[8.329914093017578]], [[7.0845184326171875]], [[8.107961654663086]], [[8.191975593566895]], [[7.5113019943237305]], [[8.455312728881836]], [[7.586492538452148]], [[7.396222114562988]], [[7.046698093414307]], [[7.9782514572143555]], [[7.5932087898254395]], [[8.972037315368652]], [[7.453706741333008]], [[7.4995551109313965]], [[7.706169128417969]], [[8.040281295776367]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a4fd1c60c8b9fd19d0238ba297ace9a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1db8f9acb21b31a0998527dfd317bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4fd1c60c8b9fd19d0238ba297ace9a1
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c15bbc8ee9f13e23a93b6ebc492b5b53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a0983a52812b20dda9b6010ed66e42cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a595aa10ba7adb59f1a60fa9f15985cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0983a52812b20dda9b6010ed66e42cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.53007698059082]], [[4.731031894683838]], [[5.071558952331543]], [[4.159222602844238]], [[4.631857395172119]], [[4.282650470733643]], [[4.188570976257324]], [[4.655451774597168]], [[4.119109153747559]], [[4.581943988800049]], [[4.6004557609558105]], [[3.6670613288879395]], [[4.5001068115234375]], [[4.365671634674072]], [[4.397197723388672]], [[4.9889421463012695]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_2b1ebf3b10030b5260981c0800651a04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd0da753a3deeae3e873367063651985(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d605e2e66ac1286fe7290f770209e404(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b50ac300ab8b642c176b9f90b0133af2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b4c369ad733a03668d7421880aad9a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af91863c96e4c4ea3c47bc055891379a
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dd2475f4d47e298dbfe15d80b67d62d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d47cc1683bfaec6af5c7a4b14d761bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd2475f4d47e298dbfe15d80b67d62d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_460036ea74e805d878499c271129c31f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.688800811767578]], [[7.571673393249512]], [[7.303963661193848]], [[8.038948059082031]], [[7.213616371154785]], [[7.31330680847168]], [[8.098185539245605]], [[7.003323078155518]], [[7.517828464508057]], [[8.359289169311523]], [[8.598203659057617]], [[7.486069202423096]], [[7.572921276092529]], [[8.42301082611084]], [[7.036831855773926]], [[7.792578220367432]], [[8.356368064880371]], [[7.543519020080566]], [[7.518827438354492]], [[8.323749542236328]], [[7.809026718139648]], [[7.231009006500244]], [[8.173846244812012]], [[7.390050888061523]], [[7.922072887420654]], [[6.892157554626465]], [[8.021842956542969]], [[7.805875301361084]], [[7.734395980834961]], [[6.467280387878418]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_52cff90ac49fe5a2fc4cd42bdbfb16ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac2966cc9eae47de7c7a1ef5aeedee38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52cff90ac49fe5a2fc4cd42bdbfb16ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ca10a88fc3911c228e838f809989946(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cfc9fb6881229c05e105c8cf2fa2fe16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ca10a88fc3911c228e838f809989946
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_225cacc337ea99ce9c08c9f5382d4554(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 25, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c45d7a86077937811560520dd8634e34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_225cacc337ea99ce9c08c9f5382d4554
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.0990824699401855]], [[7.608682155609131]], [[7.259829521179199]], [[6.562521934509277]], [[7.5483903884887695]], [[7.331745147705078]], [[6.962090969085693]], [[7.162661075592041]], [[7.244020938873291]], [[6.97843599319458]], [[7.068371295928955]], [[7.260822772979736]], [[7.184074878692627]], [[7.414368152618408]], [[6.817461967468262]], [[7.097340106964111]], [[6.920147895812988]], [[6.856722354888916]], [[6.658313274383545]], [[7.813545227050781]], [[7.304065227508545]], [[6.44976282119751]], [[6.9875168800354]], [[6.560308933258057]], [[7.412644863128662]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_06bb18e5c40db17a4459edca3cad7252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ebea6d2caa4f21d58171dfaad3095ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22033f7a176a369f1fdb8e5577fb2b06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bdac2b28a9b34d31ddd5c7efa8a710c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75ed756c4a6f2b42a523bd827a73edaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bdac2b28a9b34d31ddd5c7efa8a710c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0feac3b4b57aa679d979d3ca0c4e32fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e53dbaec81c4f1bf5ade08d45fc43abc
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0feac3b4b57aa679d979d3ca0c4e32fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e53dbaec81c4f1bf5ade08d45fc43abc
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9437274a97223d105302fee4db3c1c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0a08cd263ed348af366d1ff7d703581
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_50a6f7e8db39fb1476ca99f2cbcea897(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14f22c4ff25b3d9d837a47b26f680eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50a6f7e8db39fb1476ca99f2cbcea897
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_adce7e92c2949a3a16513fe787327735(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2db06af460d6c9adc36ae98b8f0627a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adce7e92c2949a3a16513fe787327735
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.148994445800781]], [[5.436485767364502]], [[5.777336597442627]], [[5.813238620758057]], [[5.008029937744141]], [[5.292862892150879]], [[5.6771087646484375]], [[5.6961164474487305]], [[5.450930595397949]], [[5.666596412658691]], [[6.009949684143066]], [[5.628330230712891]], [[4.612966537475586]], [[5.137825012207031]], [[5.496090888977051]], [[4.571969032287598]], [[4.844310283660889]], [[5.447681427001953]], [[5.125885963439941]], [[5.789875507354736]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac2966cc9eae47de7c7a1ef5aeedee38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52cff90ac49fe5a2fc4cd42bdbfb16ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e4ca900ff81a4af367a536ffae87eae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.401656627655029]], [[5.851325988769531]], [[4.934147834777832]], [[6.043065071105957]], [[6.093008518218994]], [[5.5484490394592285]], [[5.932394504547119]], [[6.1603240966796875]], [[6.049316883087158]], [[6.410783767700195]], [[5.968497276306152]], [[5.426406383514404]], [[5.591534614562988]], [[5.737298488616943]], [[6.143668174743652]], [[5.389349460601807]], [[6.342817306518555]], [[5.814663410186768]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_d23f28e21c713f230f438e28f7664ecb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4fd1c60c8b9fd19d0238ba297ace9a1
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdfbabfb74f814b7110909926ab01d04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66ec9373a6d99ceef26165ec39261d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d42bb0b32531c9c07b5d055e562761ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_862d281343c3397d531dbcbc2096ea35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6bc7f7089968a9b81136a55d1e3f092a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_862d281343c3397d531dbcbc2096ea35
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c640dcc587786abcfcbaf5a783bd1205(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_425a8eb222ff40da15451969e694cf36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c640dcc587786abcfcbaf5a783bd1205
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d26bb81b6bcf939ee07c4ce70645315d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d26bb81b6bcf939ee07c4ce70645315d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_425a8eb222ff40da15451969e694cf36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c640dcc587786abcfcbaf5a783bd1205
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d26bb81b6bcf939ee07c4ce70645315d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d26bb81b6bcf939ee07c4ce70645315d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3761a2c82bcf6eacd165715c90f1e641(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90a521ecfdda0a7ccdb1ae6e7d1ac244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3761a2c82bcf6eacd165715c90f1e641
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ff4165b9c461c13e87855db8742c1499(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f90c69fe1d9135eed180d91dbf4c3806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f90c69fe1d9135eed180d91dbf4c3806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e5222878e45dac2667f7f9aa8854be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3761a2c82bcf6eacd165715c90f1e641
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_450722e6a4603a2e0d9a9574bcb6a19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_450722e6a4603a2e0d9a9574bcb6a19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_61a20e2d83e41076d035bb1eb3296572(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aee51e6ca3729f82ac7e93465023f0af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a20e2d83e41076d035bb1eb3296572
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc02a659290188b2126ac981e0d04a0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc02a659290188b2126ac981e0d04a0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aee51e6ca3729f82ac7e93465023f0af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a20e2d83e41076d035bb1eb3296572
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc02a659290188b2126ac981e0d04a0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc02a659290188b2126ac981e0d04a0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fea09cd552f6519a736863aa1d74e18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3afa3c1d398f61a67bfb983869a5caf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3afa3c1d398f61a67bfb983869a5caf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa3862d08335ba854fbaa89be650ff73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5fefcaa3e218ca551a4da65b4ea2e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5fefcaa3e218ca551a4da65b4ea2e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd4fbe795f3d2bd6a6bf2c1d447a3913(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa372727c9033bc99a456d046301d8ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4fbe795f3d2bd6a6bf2c1d447a3913
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f22c4ff25b3d9d837a47b26f680eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50a6f7e8db39fb1476ca99f2cbcea897
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b58dccba7aab96b11844b17272ffb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.168549060821533]], [[3.9504952430725098]], [[4.888088226318359]], [[4.88590669631958]], [[4.539901256561279]], [[4.17132043838501]], [[4.221492767333984]], [[4.6668829917907715]], [[4.126776218414307]], [[4.761663913726807]], [[4.332804203033447]], [[4.309434413909912]], [[4.466814041137695]], [[4.468220233917236]], [[3.8908464908599854]], [[4.757633209228516]], [[4.532098770141602]], [[4.14126443862915]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_470d79ba68fa9475769130b6702b37fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0a08cd263ed348af366d1ff7d703581
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f6f3c79344a218e7d0d85d7d9bbcbe6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.38932991027832]], [[6.288872718811035]], [[4.80051326751709]], [[5.338171005249023]], [[5.631000518798828]], [[5.330127716064453]], [[5.0523834228515625]], [[5.389227867126465]], [[6.118121147155762]], [[6.174495220184326]], [[5.186315059661865]], [[6.076703071594238]], [[5.503951549530029]], [[5.625654220581055]], [[5.648626804351807]], [[5.764240741729736]], [[5.685770034790039]], [[4.80469274520874]], [[5.563747882843018]], [[5.11710786819458]], [[5.289473533630371]], [[5.886743068695068]], [[5.284122467041016]], [[5.664219856262207]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_ea684a665bc18ec635e14555d725bf02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5f1a11f5d41353d93a5d096397d194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.322781085968018]], [[4.311613082885742]], [[4.61679744720459]], [[4.344810485839844]], [[3.650794506072998]], [[4.33554220199585]], [[4.261188507080078]], [[3.9049463272094727]], [[3.7968709468841553]], [[3.907479763031006]], [[4.081305027008057]], [[4.279694080352783]], [[4.2520856857299805]], [[4.151562213897705]], [[3.9846291542053223]], [[4.152926445007324]], [[4.334284782409668]], [[3.5534818172454834]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_3cb0342d19c82cff616537299560a09e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e13027ddad853ac6beec6149a0d68cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cb0342d19c82cff616537299560a09e
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad563b75e11e93fa4ceb0109d28743b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c746cccb02a438cda438eab1f7b22476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.0413899421691895]], [[4.813213348388672]], [[4.553470134735107]], [[5.081023693084717]], [[4.93549108505249]], [[4.358945369720459]], [[4.546994686126709]], [[4.717782974243164]], [[4.752879619598389]], [[5.128915786743164]], [[5.44639253616333]], [[4.639196395874023]], [[4.349574565887451]], [[4.3120903968811035]], [[5.0812296867370605]], [[4.466093063354492]], [[5.013637065887451]], [[4.585987091064453]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_1e13027ddad853ac6beec6149a0d68cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cb0342d19c82cff616537299560a09e
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6475d6510b9bb381631c1172abfd396(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a7a01604e35fb90183c8ec529925cea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6475d6510b9bb381631c1172abfd396
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39dd6aba0d96c7269bdaa5ab726614a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43f756645af26c00d8e092250c8b9c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_862d281343c3397d531dbcbc2096ea35
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bcd6e139575a4d03995134881d2685a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c640dcc587786abcfcbaf5a783bd1205
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e26becf3ed4e1b72d62ee4b4b6f661ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e26becf3ed4e1b72d62ee4b4b6f661ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bcd6e139575a4d03995134881d2685a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c640dcc587786abcfcbaf5a783bd1205
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e26becf3ed4e1b72d62ee4b4b6f661ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e26becf3ed4e1b72d62ee4b4b6f661ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bc8a1e3111ff76200a9a85e886e60de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3761a2c82bcf6eacd165715c90f1e641
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_002af9b63dce79136e14f4819b94a307(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_002af9b63dce79136e14f4819b94a307(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73cce92bc59bb000981a8ca503f9d424(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3761a2c82bcf6eacd165715c90f1e641
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b25fc12a4eb96981ee8768071106204d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b25fc12a4eb96981ee8768071106204d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bff8865a93600aa358f75b3ca1caad2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a20e2d83e41076d035bb1eb3296572
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16f860a3e9515e886dbbbdb3d4661e47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16f860a3e9515e886dbbbdb3d4661e47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bff8865a93600aa358f75b3ca1caad2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a20e2d83e41076d035bb1eb3296572
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16f860a3e9515e886dbbbdb3d4661e47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16f860a3e9515e886dbbbdb3d4661e47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1372d313013f756918835efdfc53e1c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdbc30aa0e7fc757a294a59473bd8827(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdbc30aa0e7fc757a294a59473bd8827(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff9fd3a84aba2f8710b94e2bb85df6bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22043de8e962bca2bfc24cb6cb37cf73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22043de8e962bca2bfc24cb6cb37cf73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_257f4ea79d91083fbac4fc9a9fff4e6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4fbe795f3d2bd6a6bf2c1d447a3913
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3276ba1b6e1ec28a00745b6f19cf18b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f401632ac5cce67cad70c8d908922d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d47cc1683bfaec6af5c7a4b14d761bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd2475f4d47e298dbfe15d80b67d62d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f9d95c85b1c0dd86f641545499001bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0a08cd263ed348af366d1ff7d703581
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d79285be9788f912c99226c56a24e6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_187475791af8b62595e339faac1a64eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4fd1c60c8b9fd19d0238ba297ace9a1
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9437274a97223d105302fee4db3c1c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0a08cd263ed348af366d1ff7d703581
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bff8fc6f4f2069384d5b259096a428e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bff8fc6f4f2069384d5b259096a428e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89382923bddde37b86eb6d5b1fce0435(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89382923bddde37b86eb6d5b1fce0435(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c80610f9394eb30b01d387f0294223c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c80610f9394eb30b01d387f0294223c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c80610f9394eb30b01d387f0294223c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f3f760cda8af33318e267f655f96cdb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_710a37a94e933d09090580a0a2b49779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3f760cda8af33318e267f655f96cdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_710a37a94e933d09090580a0a2b49779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3f760cda8af33318e267f655f96cdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_710a37a94e933d09090580a0a2b49779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3f760cda8af33318e267f655f96cdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6cc24657f2fbc2b910f467e59e5a1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3f760cda8af33318e267f655f96cdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6cc24657f2fbc2b910f467e59e5a1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3f760cda8af33318e267f655f96cdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6cc24657f2fbc2b910f467e59e5a1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3f760cda8af33318e267f655f96cdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ff19d89645a6e5715f03c084c9816b33(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4371bae4738707f7b5f50b0db95ef6f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff19d89645a6e5715f03c084c9816b33
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4371bae4738707f7b5f50b0db95ef6f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff19d89645a6e5715f03c084c9816b33
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b9a64be7829984298ec4c62c3835c13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c2aa9c1c03caacf6a75761be3ceefa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3f760cda8af33318e267f655f96cdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_617a7726b92e9a69369b165c7ac23650(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_807923a520c6da15c3c2e7f6bd0f0487(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_faee75763bd4197e5dea612d4ba8f32d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2cad70b07615e96595733feeff45133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8159498627467c49709a7cd56c4d9a36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87217af9c0b812fcd0f250008dc08e50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8c03ade385a894467644c0cc4f3625a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_927f60876688ec43162bf35504a5137a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af91863c96e4c4ea3c47bc055891379a
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93503fd08d887ddc5d0c6e5cb630901e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4fd1c60c8b9fd19d0238ba297ace9a1
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_627908293dd6260c8ab2ca0996dce0df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11557eb2dc4f241d19be0201c869e380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.496325969696045]], [[4.676479816436768]], [[4.272510528564453]], [[4.639997959136963]], [[3.9382097721099854]], [[4.531309127807617]], [[5.07270622253418]], [[4.9114837646484375]], [[4.314321994781494]], [[4.516523838043213]], [[4.526065349578857]], [[4.657249450683594]], [[4.689118385314941]], [[4.198484420776367]], [[4.094197750091553]], [[4.901497840881348]], [[5.006951808929443]], [[4.796614646911621]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_14f22c4ff25b3d9d837a47b26f680eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50a6f7e8db39fb1476ca99f2cbcea897
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_002596cc205ad3799e2df435c49bc3ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b04d09903076757f6f0434b06900f4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_002596cc205ad3799e2df435c49bc3ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_640bf5323d92ac0930ca434848e3d5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5410dd7e21cc8a15b888720b3345ac7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f796f38fece6be95ebbfab3b2de872f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cdec35cb8ef5c9d34eba26f2ab6d8c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0983a52812b20dda9b6010ed66e42cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.3317461013793945]], [[3.7822155952453613]], [[3.8322110176086426]], [[3.857466220855713]], [[3.882558584213257]], [[3.3941285610198975]], [[3.9580178260803223]], [[3.9200615882873535]], [[4.120512962341309]], [[3.724442958831787]], [[3.991763114929199]], [[3.8680496215820312]], [[3.990579843521118]], [[3.73437762260437]], [[3.345764636993408]], [[3.999772071838379]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_38a1aa272096435bb0dbbc4612f28b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6475d6510b9bb381631c1172abfd396
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9ba58c0312e0fcf9384fef36a04e869(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.674563407897949]], [[3.7325024604797363]], [[4.349916934967041]], [[4.446511745452881]], [[4.36876106262207]], [[4.073473930358887]], [[4.822630405426025]], [[5.114545822143555]], [[5.140368461608887]], [[4.044999599456787]], [[4.667781352996826]], [[4.308403968811035]], [[4.528727054595947]], [[4.867443561553955]], [[3.711825132369995]], [[4.7906904220581055]], [[4.651676654815674]], [[3.9279441833496094]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_6017fb748834b8d9bcee43d1238f169d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_331c68240a16a39b8c14d67319da63b5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4343864917755127]], [[1.2709107398986816]], [[1.1146671772003174]], [[1.3615840673446655]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_a07a0910de44d4d166b34a0c8e77257b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_862d281343c3397d531dbcbc2096ea35
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82b8a204a080a1357bfd5325f740f2a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c640dcc587786abcfcbaf5a783bd1205
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a49eee2595a88835aebf9a65e497624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a49eee2595a88835aebf9a65e497624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82b8a204a080a1357bfd5325f740f2a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c640dcc587786abcfcbaf5a783bd1205
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a49eee2595a88835aebf9a65e497624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a49eee2595a88835aebf9a65e497624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9d03ecadbf8819ccfe5a8ab725b046e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3761a2c82bcf6eacd165715c90f1e641
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a4bd49e2de26b4c24daaa0dd0b8aa64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a4bd49e2de26b4c24daaa0dd0b8aa64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b83866c6938a416c4003f24aceed85f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3761a2c82bcf6eacd165715c90f1e641
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ee23e77d1b9a11f4ca44d6fecacdaa3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ee23e77d1b9a11f4ca44d6fecacdaa3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15fc245aae7fba2b4988bd9e46aba45a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a20e2d83e41076d035bb1eb3296572
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc21bcbaa5e16f6a2289900a52fbd13b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc21bcbaa5e16f6a2289900a52fbd13b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15fc245aae7fba2b4988bd9e46aba45a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a20e2d83e41076d035bb1eb3296572
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc21bcbaa5e16f6a2289900a52fbd13b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc21bcbaa5e16f6a2289900a52fbd13b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b612d0cbaa8cee4f4af71e426e389d0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61319b2b14c465210bacddf2b8bed25c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61319b2b14c465210bacddf2b8bed25c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f3388aaa7093717abff720ce48ddbe5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87188700719e72b9ab3fd47cb69aec3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87188700719e72b9ab3fd47cb69aec3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd79db6a05455231cde0b14b7be25a29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4fbe795f3d2bd6a6bf2c1d447a3913
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f22c4ff25b3d9d837a47b26f680eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50a6f7e8db39fb1476ca99f2cbcea897
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ed756c4a6f2b42a523bd827a73edaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bdac2b28a9b34d31ddd5c7efa8a710c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b49d1b1abad7c8194b57dc111d62107b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af91863c96e4c4ea3c47bc055891379a
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_05961c07033836b352cc77668124cc46(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3942964f8cc8c8fb3d3f85dcfb9db53a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05961c07033836b352cc77668124cc46
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2acf820f622634463f5cfd3707e8b9dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d99193252ec767ae335660e06f82099(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2acf820f622634463f5cfd3707e8b9dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f9d95c85b1c0dd86f641545499001bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0a08cd263ed348af366d1ff7d703581
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f22c4ff25b3d9d837a47b26f680eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50a6f7e8db39fb1476ca99f2cbcea897
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e13027ddad853ac6beec6149a0d68cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cb0342d19c82cff616537299560a09e
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48d3644e571fb9c1df02beb6bdfc499b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adce7e92c2949a3a16513fe787327735
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.935728549957275]], [[5.193583011627197]], [[5.405919075012207]], [[5.343483924865723]], [[4.915558338165283]], [[5.3574299812316895]], [[5.5367655754089355]], [[5.295175075531006]], [[5.847105503082275]], [[4.853701591491699]], [[5.489466667175293]], [[5.752553462982178]], [[5.8476457595825195]], [[5.416018009185791]], [[5.76278829574585]], [[5.513062477111816]], [[5.44843864440918]], [[5.8336710929870605]], [[5.499468803405762]], [[5.231005668640137]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_afb7675172c2e089708bc51d7e2860b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 84, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f223f957edaf6584280d491870c99812(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afb7675172c2e089708bc51d7e2860b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c6ac268c1ce3461b2ed49030c8b7f2b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90e98dbaecc2b3c1d81b9110ed8c7a8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6ac268c1ce3461b2ed49030c8b7f2b0
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.838440418243408]], [[3.725783348083496]], [[3.4627926349639893]], [[2.86716365814209]], [[3.1309502124786377]], [[3.1671664714813232]], [[2.832571506500244]], [[3.4852497577667236]], [[2.6229331493377686]], [[3.4359145164489746]], [[2.955007553100586]], [[3.234879970550537]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_1a662a94df236af7e5876d6a509459f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adce7e92c2949a3a16513fe787327735
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.189188003540039]], [[5.522616386413574]], [[6.239436626434326]], [[5.228600025177002]], [[5.076842784881592]], [[4.971395969390869]], [[5.320101737976074]], [[5.200826168060303]], [[5.677047252655029]], [[4.709473133087158]], [[4.8291730880737305]], [[4.934872150421143]], [[5.219989776611328]], [[4.80227518081665]], [[5.564419269561768]], [[5.456704139709473]], [[4.736117362976074]], [[5.041024684906006]], [[5.031208038330078]], [[5.072656154632568]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_8e4d36649e8ece7a61fe94720c1d5c52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc2fb3a9a0470503530e9afaa2254eda
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.9690845012664795]], [[2.9818687438964844]], [[2.8145813941955566]], [[3.0310134887695312]], [[3.1999592781066895]], [[2.9077987670898438]], [[3.10152006149292]], [[2.8881208896636963]], [[3.345881462097168]], [[2.6285908222198486]], [[3.7109107971191406]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d99193252ec767ae335660e06f82099(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2acf820f622634463f5cfd3707e8b9dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a5f13e421112c73d53e1726fa81bbbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_90e8b0224ba56c91b9e2e77ac41050f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 14, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3979ceea4300e406affec29929f00745(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e8b0224ba56c91b9e2e77ac41050f6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.728445053100586]], [[4.151237487792969]], [[3.621351718902588]], [[3.568361759185791]], [[4.5313801765441895]], [[4.230593204498291]], [[3.3904149532318115]], [[3.592285394668579]], [[4.054468154907227]], [[3.8145062923431396]], [[3.9085946083068848]], [[3.8280105590820312]], [[3.6504974365234375]], [[3.925114393234253]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    
    class PrimitiveOp_49626a2e3217b6d4e4107ceaaca788a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46555038f8560d16f0b705b58dd02dcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49626a2e3217b6d4e4107ceaaca788a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e80eb019ef0c5559ffe2d487c9bd3239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_640bf5323d92ac0930ca434848e3d5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5410dd7e21cc8a15b888720b3345ac7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7c043d048f84c7ef28a3ed6a1631212(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adce7e92c2949a3a16513fe787327735
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.040356636047363]], [[4.171722888946533]], [[6.066471576690674]], [[4.901216506958008]], [[4.848870754241943]], [[4.409139633178711]], [[5.113746643066406]], [[5.3654465675354]], [[4.759063243865967]], [[5.03289794921875]], [[4.981382369995117]], [[5.127835273742676]], [[4.808407783508301]], [[5.101073741912842]], [[5.186868667602539]], [[5.055799961090088]], [[5.173654079437256]], [[4.846164226531982]], [[4.867003917694092]], [[4.343432426452637]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_83c409faabc9bee522edb627920d9768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83c409faabc9bee522edb627920d9768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83c409faabc9bee522edb627920d9768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83c409faabc9bee522edb627920d9768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_20af8e107ca644bafc225f39f04bd0d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ec1630afdb43079199a03d64f0e42294(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20af8e107ca644bafc225f39f04bd0d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33790.01171875]], [[33246.22265625]], [[30786.228515625]], [[30725.201171875]], [[38834.828125]], [[33852.90234375]]], [[[34128.40234375]], [[33572.453125]], [[31094.193359375]], [[31031.59765625]], [[39217.91015625]], [[34191.59765625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_c8e2840d3a39b9e04ea8b8ee1d46f1f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20af8e107ca644bafc225f39f04bd0d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[36145.1640625]], [[42202.109375]], [[40099.84375]], [[37980.41015625]], [[31233.35546875]], [[32060.4609375]]], [[[36810.8984375]], [[42984.38671875]], [[40846.484375]], [[38684.21484375]], [[31809.748046875]], [[32650.68359375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_35e061665e24ee81206f40a260200451(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20af8e107ca644bafc225f39f04bd0d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[32827.54296875]], [[38912.5078125]], [[41232.10546875]], [[42047.49609375]], [[42913.828125]], [[47525.63671875]]], [[[32996.40625]], [[39109.4140625]], [[41443.84765625]], [[42258.4296875]], [[43128.10546875]], [[47765.79296875]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_86daff7738da4ed58632cd673894c957(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20af8e107ca644bafc225f39f04bd0d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34932.6171875]], [[30921.677734375]], [[40886.67578125]], [[36311.15234375]], [[43067.72265625]], [[48601.33203125]]], [[[35221.71875]], [[31168.6796875]], [[41222.20703125]], [[36605.85546875]], [[43419.46875]], [[48995.37890625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_683e8e81f25d4db8566f7634998acc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_683e8e81f25d4db8566f7634998acc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_683e8e81f25d4db8566f7634998acc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_683e8e81f25d4db8566f7634998acc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_683e8e81f25d4db8566f7634998acc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_683e8e81f25d4db8566f7634998acc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_683e8e81f25d4db8566f7634998acc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_683e8e81f25d4db8566f7634998acc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bba8a7597565896584bd7a95cbc7bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bba8a7597565896584bd7a95cbc7bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bba8a7597565896584bd7a95cbc7bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bba8a7597565896584bd7a95cbc7bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bba8a7597565896584bd7a95cbc7bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bba8a7597565896584bd7a95cbc7bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bba8a7597565896584bd7a95cbc7bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bba8a7597565896584bd7a95cbc7bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f68551f0ca9de4018a7693cf37e2969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f68551f0ca9de4018a7693cf37e2969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f68551f0ca9de4018a7693cf37e2969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f68551f0ca9de4018a7693cf37e2969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f68551f0ca9de4018a7693cf37e2969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f68551f0ca9de4018a7693cf37e2969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f68551f0ca9de4018a7693cf37e2969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f68551f0ca9de4018a7693cf37e2969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e13027ddad853ac6beec6149a0d68cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cb0342d19c82cff616537299560a09e
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9267e7487044d5e0ce957652cc4f55d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.25681209564209]], [[8.124125480651855]], [[7.101140975952148]], [[8.303726196289062]], [[7.532835006713867]], [[7.419724464416504]], [[7.211884498596191]], [[7.81193208694458]], [[7.384727478027344]], [[7.545863151550293]], [[8.518484115600586]], [[8.447880744934082]], [[7.607048511505127]], [[8.261672973632812]], [[8.002284049987793]], [[8.208133697509766]], [[7.965606689453125]], [[8.26777458190918]], [[7.481735706329346]], [[7.620168685913086]], [[9.100288391113281]], [[7.642550945281982]], [[7.4412007331848145]], [[7.278196811676025]], [[7.8947672843933105]], [[8.380417823791504]], [[7.684148788452148]], [[8.735790252685547]], [[8.403761863708496]], [[7.420577049255371]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_e51ff50f167615bc9beabb305d40c031(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.159058570861816]], [[8.9575777053833]], [[7.920828342437744]], [[8.26949405670166]], [[8.606870651245117]], [[7.703291893005371]], [[9.033196449279785]], [[7.2510600090026855]], [[8.894136428833008]], [[8.381927490234375]], [[8.117466926574707]], [[8.615775108337402]], [[8.195988655090332]], [[8.25282096862793]], [[9.394417762756348]], [[8.261106491088867]], [[8.069947242736816]], [[8.460314750671387]], [[9.121085166931152]], [[8.430448532104492]], [[8.327909469604492]], [[7.7775983810424805]], [[8.79182243347168]], [[7.773296356201172]], [[8.375483512878418]], [[8.84751033782959]], [[8.769116401672363]], [[7.66787052154541]], [[8.125309944152832]], [[7.6072678565979]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_cf3d319de037b85aa5d6a1eda39b20c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13d9c3c5d4e44e0c369448be967358e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.9664530754089355]], [[6.8404951095581055]], [[8.002744674682617]], [[7.52061128616333]], [[7.19194221496582]], [[6.707631587982178]], [[6.572324752807617]], [[6.8679728507995605]], [[6.855906963348389]], [[7.773515701293945]], [[7.6090497970581055]], [[7.906286716461182]], [[6.690423011779785]], [[6.965617656707764]], [[7.726587295532227]], [[7.490701198577881]], [[7.930479526519775]], [[7.766480922698975]], [[7.732168197631836]], [[6.953736305236816]], [[6.223891258239746]], [[7.326486110687256]], [[7.540115833282471]], [[7.350153923034668]], [[7.1114702224731445]], [[7.341914653778076]], [[6.942921161651611]], [[7.209981441497803]], [[7.393255710601807]], [[7.669460296630859]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_b5bcda71c04d1ed4cf8eaa9e6e4c6d8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 50, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_deab263667c8374f3852e280b7009ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5bcda71c04d1ed4cf8eaa9e6e4c6d8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f22c4ff25b3d9d837a47b26f680eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50a6f7e8db39fb1476ca99f2cbcea897
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0a336fb3bf314c2d05d85acc5ce2d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.4437432289123535]], [[8.28723430633545]], [[7.912923812866211]], [[7.163331508636475]], [[7.706422805786133]], [[7.904669761657715]], [[7.870941162109375]], [[8.095754623413086]], [[7.887270450592041]], [[7.398651599884033]], [[8.18539047241211]], [[7.227769374847412]], [[8.68224811553955]], [[8.48115062713623]], [[7.778359413146973]], [[7.674652576446533]], [[8.059738159179688]], [[7.699238300323486]], [[8.101340293884277]], [[7.794434070587158]], [[7.354639530181885]], [[7.219238758087158]], [[7.487193584442139]], [[7.905284404754639]], [[7.686999797821045]], [[8.279914855957031]], [[8.093050956726074]], [[7.652628421783447]], [[8.023968696594238]], [[7.812108516693115]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_a4204bde0baf83ae85904cddb165d74a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6ac268c1ce3461b2ed49030c8b7f2b0
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.8361976146698]], [[3.135993003845215]], [[2.961843490600586]], [[3.4332213401794434]], [[3.0051279067993164]], [[2.8730978965759277]], [[3.358252763748169]], [[3.0638809204101562]], [[3.0245022773742676]], [[3.537898063659668]], [[2.83929705619812]], [[2.9802045822143555]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_9d17582a88aad2b80b1d588347d264d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6ac268c1ce3461b2ed49030c8b7f2b0
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.4488754272460938]], [[3.1694984436035156]], [[3.566474676132202]], [[3.0803067684173584]], [[3.2484779357910156]], [[2.6101138591766357]], [[3.772961139678955]], [[3.672159194946289]], [[3.6704392433166504]], [[3.145970344543457]], [[3.907890558242798]], [[3.1002845764160156]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_69e4cea53838e74491bc20cd0ac447b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_225cacc337ea99ce9c08c9f5382d4554
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.898226261138916]], [[7.46968412399292]], [[6.872394561767578]], [[6.250451564788818]], [[7.033980369567871]], [[6.551626205444336]], [[7.342005252838135]], [[7.041738033294678]], [[6.796108722686768]], [[6.893855571746826]], [[6.811644554138184]], [[6.453948020935059]], [[6.911863803863525]], [[6.331982612609863]], [[7.10107946395874]], [[6.501152992248535]], [[7.3300909996032715]], [[6.193831443786621]], [[6.802927494049072]], [[6.822544097900391]], [[6.546031951904297]], [[6.251367568969727]], [[7.175201892852783]], [[5.89363431930542]], [[6.777373313903809]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    
    class PrimitiveOp_7438dd0c84756ec683e47985a50055b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c28c7b117f32a567ef0563ae1fe9ba85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7438dd0c84756ec683e47985a50055b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07b3970a4137cb427c85d3142cadf4c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7dbddccd7ace78dc7ff2fe5a43844423(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07b3970a4137cb427c85d3142cadf4c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90c55add5a09b9aebf9b3e0ba454171a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f401632ac5cce67cad70c8d908922d7
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e2b28880581a8e09f3d8209ef1ac37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6475d6510b9bb381631c1172abfd396
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c80ddd3ad7475f48e6601519cc672628(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_edbacdfef589bbc756501d2c3f187872(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.34008264541626]], [[4.743483066558838]], [[5.1139140129089355]], [[4.816980361938477]], [[5.030374050140381]], [[5.1157050132751465]], [[5.042732238769531]], [[4.310696601867676]], [[5.24165153503418]], [[5.031465530395508]], [[4.28033447265625]], [[4.85125207901001]], [[5.221953392028809]], [[5.398641586303711]], [[4.63865327835083]], [[4.977076053619385]], [[5.138279438018799]], [[4.5899553298950195]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_199235b01cdafbf052c97b024261bd62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_174b0e0f225967a85a38dec7d1283fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_199235b01cdafbf052c97b024261bd62
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cfebdb7d718d767de7eb07bcba33cd0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92a7bd3697bf8bcbd05c3fbfac2d05e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfebdb7d718d767de7eb07bcba33cd0e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.1509110927581787]], [[1.1300290822982788]], [[1.3730714321136475]], [[1.2706753015518188]], [[1.0934555530548096]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    
    class PrimitiveOp_f71bea4d55483c97cc5f7e6a017c8237(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2956de81fc812b0cf0302f18ef7fe81b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f71bea4d55483c97cc5f7e6a017c8237
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.8189101219177246]], [[2.6422502994537354]], [[2.8085169792175293]], [[2.7595033645629883]], [[2.954659938812256]], [[2.689716339111328]], [[2.191497564315796]], [[2.3258798122406006]], [[2.354254722595215]], [[2.6504807472229004]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    
    class PrimitiveOp_10585ce93439fa0c1a5b6d47213a9329(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1599126b083de9df34f8f500ffef482a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_10585ce93439fa0c1a5b6d47213a9329
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.700591564178467]], [[5.198940277099609]], [[5.845884799957275]], [[5.625529766082764]], [[5.1735100746154785]], [[4.8036723136901855]], [[5.090089797973633]], [[4.773566722869873]], [[5.558661937713623]], [[5.0450568199157715]], [[5.424689292907715]], [[4.559719562530518]], [[5.617161273956299]], [[4.593595504760742]], [[5.566433906555176]], [[5.211007118225098]], [[5.107222557067871]], [[4.509981632232666]], [[4.843706130981445]], [[5.598837852478027]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_f45c0ea49a6d11f003e781835a74cdd8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7399be1db2ae8bec5ab9a07f3b33039c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c0ea49a6d11f003e781835a74cdd8
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_deab263667c8374f3852e280b7009ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5bcda71c04d1ed4cf8eaa9e6e4c6d8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac2966cc9eae47de7c7a1ef5aeedee38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52cff90ac49fe5a2fc4cd42bdbfb16ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfc9fb6881229c05e105c8cf2fa2fe16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ca10a88fc3911c228e838f809989946
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c5159dc9f650efb9574fff788a12dc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.023273944854736]], [[5.066927909851074]], [[4.140905857086182]], [[5.343730926513672]], [[5.36994743347168]], [[4.6580400466918945]], [[5.503517150878906]], [[4.593698024749756]], [[4.692727088928223]], [[5.1036295890808105]], [[5.059436321258545]], [[5.221367359161377]], [[5.29227876663208]], [[4.9216837882995605]], [[5.376871109008789]], [[6.023849010467529]], [[4.857785701751709]], [[4.9857001304626465]], [[4.833615303039551]], [[5.491693496704102]], [[5.115340232849121]], [[5.572267532348633]], [[5.060335159301758]], [[5.243594646453857]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f21fc4581fee63110e38d115e1f680b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f401632ac5cce67cad70c8d908922d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b58de4b8d78be5a383232c7303e5ebc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7655ad88391ddaf1de39aa3e53d86fcd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.470754384994507]], [[3.0341553688049316]], [[3.070213794708252]], [[2.381013870239258]], [[2.537949323654175]], [[3.0229506492614746]], [[2.40401554107666]], [[2.5496742725372314]], [[2.6882126331329346]], [[2.872746229171753]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_28f00ed818eb5b41a71cf0e5ad5c80e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f401632ac5cce67cad70c8d908922d7
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13ac872f9edf6dce0470aa50158ef158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4780b534b4812886f78e85d81c2ceea6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da40d9ec958d1dd414fef6744a416369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4780b534b4812886f78e85d81c2ceea6
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_faa69deae39e2630afcebdc00d78bb1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6475d6510b9bb381631c1172abfd396
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c31a5f8bccdcb1558fb1900b62191655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a0b00daf3d79514f03815bc680e2cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.73128604888916]], [[5.034897804260254]], [[4.891191005706787]], [[4.268646717071533]], [[4.652425289154053]], [[4.6083478927612305]], [[4.871151447296143]], [[4.301667213439941]], [[4.273050785064697]], [[4.274194717407227]], [[4.2991414070129395]], [[5.229434967041016]], [[4.240005970001221]], [[4.6630859375]], [[5.333066940307617]], [[4.8534698486328125]], [[4.757251262664795]], [[3.8645853996276855]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_82a395c84341a3dc59211786205f20f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a31bdb49eed8decd3068b0c4d4cf8f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82a395c84341a3dc59211786205f20f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.777217388153076, 8.48762035369873, 7.249322414398193, 7.724056243896484, 7.930503845214844, 7.815942287445068, 8.609721183776855, 8.345749855041504, 9.051900863647461, 8.07912826538086, 8.785380363464355, 8.09261417388916, 8.483768463134766, 7.842811107635498, 8.470643043518066, 9.749964714050293, 8.221514701843262, 7.603100299835205, 8.097103118896484, 8.626155853271484, 8.103931427001953, 7.6056227684021, 7.974206924438477, 8.232665061950684, 8.936210632324219, 8.685194969177246, 8.046462059020996, 7.636215686798096, 7.272360801696777, 8.230209350585938]], dtype='float32').reshape([1, 30]),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3942964f8cc8c8fb3d3f85dcfb9db53a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05961c07033836b352cc77668124cc46
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2aaa2074192ab81104e50f42d8cbc203(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.9253621101379395]], [[8.135875701904297]], [[7.322157382965088]], [[8.395102500915527]], [[7.788322448730469]], [[8.539194107055664]], [[7.630413055419922]], [[7.462310314178467]], [[7.580341815948486]], [[7.088729381561279]], [[7.536751747131348]], [[7.9274821281433105]], [[7.52095365524292]], [[7.826623916625977]], [[7.471090316772461]], [[7.233271598815918]], [[8.233546257019043]], [[8.312676429748535]], [[8.348246574401855]], [[8.340503692626953]], [[7.820869445800781]], [[8.069595336914062]], [[8.100695610046387]], [[7.995584011077881]], [[7.848565101623535]], [[7.846622467041016]], [[7.562673091888428]], [[8.764762878417969]], [[8.024958610534668]], [[8.593799591064453]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_2589d774a497f16988088508c935df3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bebd043c1092eba0279d9881acf9ea4c
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.7045177221298218]], [[1.6164036989212036]], [[1.4427224397659302]], [[1.5230357646942139]], [[1.8313349485397339]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_4fc3245bd3244ab3b3f7c6c9f8bc65b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7655ad88391ddaf1de39aa3e53d86fcd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.6212942600250244]], [[2.881885051727295]], [[2.826146125793457]], [[3.1407322883605957]], [[2.967379093170166]], [[2.947307586669922]], [[2.8775577545166016]], [[2.7747371196746826]], [[2.985468864440918]], [[3.1787304878234863]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_b0d1fff4a328521b8731902cfe1d5543(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adce7e92c2949a3a16513fe787327735
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.665043354034424]], [[6.21737813949585]], [[6.037357807159424]], [[5.357994556427002]], [[5.321767330169678]], [[5.587750434875488]], [[5.597776412963867]], [[5.610191822052002]], [[5.384403228759766]], [[5.28219747543335]], [[5.194252967834473]], [[5.6317667961120605]], [[5.777074337005615]], [[5.500901222229004]], [[5.264638423919678]], [[5.222471237182617]], [[6.135081768035889]], [[5.563098430633545]], [[5.966292381286621]], [[5.470348834991455]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_640bf5323d92ac0930ca434848e3d5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5410dd7e21cc8a15b888720b3345ac7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73a5da063b7788a397728041347977c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0983a52812b20dda9b6010ed66e42cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.8203845024108887]], [[4.383570194244385]], [[4.741254806518555]], [[3.7061283588409424]], [[3.4336838722229004]], [[4.516073703765869]], [[4.192562103271484]], [[4.824357032775879]], [[4.380889415740967]], [[4.580226421356201]], [[4.271740436553955]], [[4.298461437225342]], [[5.017632007598877]], [[4.143437385559082]], [[4.676990032196045]], [[3.806002140045166]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_d47cc1683bfaec6af5c7a4b14d761bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd2475f4d47e298dbfe15d80b67d62d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4dc217a45b6bf4c88bb00cdccbdd08a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56f82562838deaad38875d7cb7f5942e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dc217a45b6bf4c88bb00cdccbdd08a7
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83459bbd4b72b622cafd85e54e7d0fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83459bbd4b72b622cafd85e54e7d0fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83459bbd4b72b622cafd85e54e7d0fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83459bbd4b72b622cafd85e54e7d0fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83459bbd4b72b622cafd85e54e7d0fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83459bbd4b72b622cafd85e54e7d0fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83459bbd4b72b622cafd85e54e7d0fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83459bbd4b72b622cafd85e54e7d0fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_555ce7727a686e5e09b15c8450353091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_555ce7727a686e5e09b15c8450353091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_555ce7727a686e5e09b15c8450353091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_555ce7727a686e5e09b15c8450353091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_555ce7727a686e5e09b15c8450353091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_555ce7727a686e5e09b15c8450353091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_555ce7727a686e5e09b15c8450353091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_555ce7727a686e5e09b15c8450353091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e298f7f4f6f96a457db2c64eca4e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e298f7f4f6f96a457db2c64eca4e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e298f7f4f6f96a457db2c64eca4e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e298f7f4f6f96a457db2c64eca4e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e298f7f4f6f96a457db2c64eca4e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e298f7f4f6f96a457db2c64eca4e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e298f7f4f6f96a457db2c64eca4e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e298f7f4f6f96a457db2c64eca4e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721018a8f53e61da37c2c796540ef0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab40d7c77485f199e90faf585be14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f223f957edaf6584280d491870c99812(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afb7675172c2e089708bc51d7e2860b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46555038f8560d16f0b705b58dd02dcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49626a2e3217b6d4e4107ceaaca788a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d3e06839b273320c3ca2c41f87a4387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e8b0224ba56c91b9e2e77ac41050f6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.6342556476593018]], [[3.1763720512390137]], [[3.495896339416504]], [[3.4828929901123047]], [[3.3462088108062744]], [[3.6012353897094727]], [[2.8251118659973145]], [[3.4390714168548584]], [[2.9350075721740723]], [[3.2876410484313965]], [[3.9005303382873535]], [[3.4099371433258057]], [[3.195777416229248]], [[3.1334729194641113]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_9f368caca4d0fddad37063c3dff03346(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adce7e92c2949a3a16513fe787327735
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.698542594909668]], [[5.4758477210998535]], [[4.34893274307251]], [[5.1414713859558105]], [[5.090174674987793]], [[5.6204833984375]], [[5.572890281677246]], [[5.212032318115234]], [[5.199526309967041]], [[4.979649543762207]], [[4.388289451599121]], [[4.937069416046143]], [[4.850468158721924]], [[4.67232084274292]], [[4.780160903930664]], [[5.114904880523682]], [[5.140608310699463]], [[4.899909019470215]], [[4.694981575012207]], [[5.400130748748779]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_9fbd4f9b2a4f7a32ed4039958a6c7c58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac2966cc9eae47de7c7a1ef5aeedee38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52cff90ac49fe5a2fc4cd42bdbfb16ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2792e5ae378e5df5e757c712966f45ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_666fc2b2be7db47fb2592978084b415e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.587968826293945]], [[7.046293258666992]], [[7.726832866668701]], [[6.813288688659668]], [[7.791248321533203]], [[6.617620468139648]], [[7.223494529724121]], [[6.607419967651367]], [[7.120966911315918]], [[8.214742660522461]], [[7.768636703491211]], [[7.641348838806152]], [[6.252624988555908]], [[7.227708339691162]], [[6.829111099243164]], [[7.346861839294434]], [[7.112125873565674]], [[6.733704090118408]], [[8.058210372924805]], [[7.438938617706299]], [[7.837202548980713]], [[7.037036895751953]], [[7.634639263153076]], [[7.035560607910156]], [[7.132640838623047]], [[6.164754867553711]], [[7.45093297958374]], [[7.50039005279541]], [[7.918460369110107]], [[7.027027606964111]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_06bb18e5c40db17a4459edca3cad7252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ebea6d2caa4f21d58171dfaad3095ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_640bf5323d92ac0930ca434848e3d5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5410dd7e21cc8a15b888720b3345ac7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56f82562838deaad38875d7cb7f5942e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dc217a45b6bf4c88bb00cdccbdd08a7
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_665799d4d08a7ce9dbca23712152afdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_862d281343c3397d531dbcbc2096ea35
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0712dacbcaac1aeef729d380b93743d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c640dcc587786abcfcbaf5a783bd1205
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b877efc589de691b6c22dc73300ca0f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b877efc589de691b6c22dc73300ca0f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0712dacbcaac1aeef729d380b93743d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c640dcc587786abcfcbaf5a783bd1205
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b877efc589de691b6c22dc73300ca0f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b877efc589de691b6c22dc73300ca0f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af7947be35ec3309c0987b379ea3830a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3761a2c82bcf6eacd165715c90f1e641
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_256de80dbf359a4e3eef65d6aaffb1f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_256de80dbf359a4e3eef65d6aaffb1f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dfa86ae8bb86ac2fd48be5d3ced7037(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3761a2c82bcf6eacd165715c90f1e641
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af529e80a235910d0eab3d155813c4e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af529e80a235910d0eab3d155813c4e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4165b9c461c13e87855db8742c1499
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa5e7d8a9b0e1ff723a64c03bab5cf09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a20e2d83e41076d035bb1eb3296572
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1135bbda98068ec3c311b01f74d1bd1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1135bbda98068ec3c311b01f74d1bd1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa5e7d8a9b0e1ff723a64c03bab5cf09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a20e2d83e41076d035bb1eb3296572
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1135bbda98068ec3c311b01f74d1bd1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1135bbda98068ec3c311b01f74d1bd1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665b2f176bb7b8cda1fd21c3d536dea2
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8a4693dbf7343199b624f38e06d4f30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d13712b81da20813ca4788f292bb775c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d13712b81da20813ca4788f292bb775c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dab924fcb8d935ef535189b8032e8e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d867ce7d36fa105ba3792c92eb90971
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_386f844406ea31e4e82815fdb1a95735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_386f844406ea31e4e82815fdb1a95735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6021660edcb064f3edd0e900864c749d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4fbe795f3d2bd6a6bf2c1d447a3913
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_deab263667c8374f3852e280b7009ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5bcda71c04d1ed4cf8eaa9e6e4c6d8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_917fd8020412f29dc1e9e6a99cc92a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d99193252ec767ae335660e06f82099(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2acf820f622634463f5cfd3707e8b9dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_805403963406cab61a0bbb1d70a7ca43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.113337516784668]], [[6.361061096191406]], [[6.383002281188965]], [[6.8510637283325195]], [[6.504920482635498]], [[6.268310546875]], [[6.040866851806641]], [[6.8944292068481445]], [[6.598022937774658]], [[6.050089359283447]], [[6.557819843292236]], [[6.211484432220459]], [[6.971176624298096]], [[5.849992752075195]], [[5.925839424133301]], [[7.056762218475342]], [[6.003842830657959]], [[6.703570365905762]], [[7.226595878601074]], [[7.061709403991699]], [[6.613278865814209]], [[6.183558464050293]], [[6.57665491104126]], [[6.672055721282959]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_1f93128fe18a61fb8bc37e73c40b5ab1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_225cacc337ea99ce9c08c9f5382d4554
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.585529327392578]], [[5.8684468269348145]], [[6.182231426239014]], [[5.794556617736816]], [[6.117100715637207]], [[6.029637813568115]], [[7.346761703491211]], [[5.943231582641602]], [[6.368186950683594]], [[5.535693645477295]], [[7.286806583404541]], [[5.88812780380249]], [[6.799605369567871]], [[6.342577934265137]], [[5.253039836883545]], [[6.462026596069336]], [[6.507450580596924]], [[5.606255054473877]], [[5.662771701812744]], [[5.6854329109191895]], [[5.967799186706543]], [[6.681526184082031]], [[6.520153045654297]], [[5.843494415283203]], [[6.265067100524902]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_43ee805171d7bd9b638d8b8200307be5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6ac268c1ce3461b2ed49030c8b7f2b0
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.105079174041748]], [[3.126370906829834]], [[3.035320281982422]], [[2.878377676010132]], [[2.9810948371887207]], [[3.045088052749634]], [[3.228938579559326]], [[3.203735828399658]], [[3.3749754428863525]], [[3.210139513015747]], [[2.8544631004333496]], [[3.5723633766174316]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_640bf5323d92ac0930ca434848e3d5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5410dd7e21cc8a15b888720b3345ac7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac2966cc9eae47de7c7a1ef5aeedee38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52cff90ac49fe5a2fc4cd42bdbfb16ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9267ca7bb9c8c84638eb6f3793640b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb00e03cfe1952d3693ed406da57458
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b07851fcdfc5af5b39a0d1fdcdae4d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff2335dd4947bb39d1d772d6fab4a098(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b07851fcdfc5af5b39a0d1fdcdae4d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d47cc1683bfaec6af5c7a4b14d761bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd2475f4d47e298dbfe15d80b67d62d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1e681f9f2b9eebb2363343d1bfdc541(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_640bf5323d92ac0930ca434848e3d5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5410dd7e21cc8a15b888720b3345ac7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f21c2e93d619794cba7de5e555b34b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74ac548a8a55c38a6e25a27281f0d5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a9279176de56e82ac62ef3d35259fbb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19c5a57fa6e0dbf5ee7a9126427025a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a9279176de56e82ac62ef3d35259fbb
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2a37311dc0d6c9c69f7a8f46cee0bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[728.019775390625]], [[746.1237182617188]], [[692.19873046875]], [[762.6597900390625]], [[671.3251953125]], [[708.3541870117188]], [[721.9227905273438]], [[783.6730346679688]], [[717.413330078125]], [[715.1480712890625]], [[717.2114868164062]], [[724.47607421875]], [[700.197509765625]], [[678.7390747070312]], [[709.4534301757812]], [[765.017333984375]], [[743.7413940429688]], [[696.52001953125]], [[729.6466674804688]], [[724.2402954101562]], [[742.330078125]], [[707.8065185546875]], [[697.4320068359375]], [[735.0343017578125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f534e01a2b2320c3c6b87166f2222500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[88.4103012084961]], [[83.92156219482422]], [[87.11354064941406]], [[95.5699234008789]], [[85.66553497314453]], [[73.65477752685547]], [[91.61893463134766]], [[89.295654296875]], [[89.7334213256836]], [[88.86317443847656]], [[88.36162567138672]], [[95.03060150146484]], [[92.3189926147461]], [[95.11656188964844]], [[86.61528015136719]], [[90.17096710205078]], [[87.8534164428711]], [[89.96961212158203]], [[91.15142822265625]], [[86.67532348632812]], [[84.73998260498047]], [[90.36804962158203]], [[85.57946014404297]], [[88.91419219970703]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f573235354ba5e2bb86b1e541ae1f2b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41.77033233642578]], [[36.550838470458984]], [[38.81454849243164]], [[40.53050231933594]], [[39.61156463623047]], [[37.262638092041016]], [[38.70488739013672]], [[37.151390075683594]], [[40.440311431884766]], [[40.4919319152832]], [[36.12879180908203]], [[41.12125015258789]], [[37.479007720947266]], [[38.575531005859375]], [[36.90913391113281]], [[40.257232666015625]], [[34.73536682128906]], [[38.0360221862793]], [[35.37649154663086]], [[38.790096282958984]], [[36.533573150634766]], [[40.45515823364258]], [[34.850669860839844]], [[33.67195129394531]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_800549d37477ad2c5e28002dc3e222aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[26.883615493774414]], [[26.930633544921875]], [[27.861154556274414]], [[25.487024307250977]], [[27.49747085571289]], [[28.103042602539062]], [[26.305788040161133]], [[28.329017639160156]], [[25.491519927978516]], [[28.231576919555664]], [[28.3348331451416]], [[28.69142723083496]], [[25.975393295288086]], [[26.011674880981445]], [[26.594940185546875]], [[25.055524826049805]], [[29.05447769165039]], [[30.182493209838867]], [[26.662887573242188]], [[25.961088180541992]], [[24.21113395690918]], [[26.02586555480957]], [[25.330015182495117]], [[26.766361236572266]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_db4fb7d8db810495179755daf3867851(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20af8e107ca644bafc225f39f04bd0d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37687.91796875]], [[31513.66796875]], [[32339.71484375]], [[26676.244140625]], [[32849.390625]], [[37629.57421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_5bceb73467b7d75c46f3139a5e96de08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20af8e107ca644bafc225f39f04bd0d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41447.62890625]], [[40241.578125]], [[39892.1015625]], [[34655.28125]], [[42560.5078125]], [[35762.80078125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_648ff651d06b8ec7a522a75cc01a6f1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20af8e107ca644bafc225f39f04bd0d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33178.44921875]], [[44838.65625]], [[41198.328125]], [[40192.58984375]], [[42689.53515625]], [[40445.8671875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_a47e24c1f57a393d7afce8e2489633df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20af8e107ca644bafc225f39f04bd0d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[36067.48828125]], [[44561.07421875]], [[46742.91796875]], [[42671.6640625]], [[43674.9453125]], [[47246.0625]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_0176300d73bc8e4f4019142cf8c9e2b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06bb18e5c40db17a4459edca3cad7252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ebea6d2caa4f21d58171dfaad3095ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c28c7b117f32a567ef0563ae1fe9ba85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7438dd0c84756ec683e47985a50055b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fba4bf34021428649bbd168dd0b920d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbaca726853922d39ae3de53aafbbd53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61eb76df51303870b76ea6e84641eabf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f26a146c29bc2055fb0949e503a5223
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.819546222686768]], [[6.063246250152588]], [[6.408529758453369]], [[5.9528303146362305]], [[5.815957546234131]], [[6.398426055908203]], [[5.963871955871582]], [[6.095754146575928]], [[5.961697101593018]], [[5.839191436767578]], [[6.142646789550781]], [[5.494499206542969]], [[5.5466766357421875]], [[5.561925411224365]], [[6.77968692779541]], [[6.119173049926758]], [[5.838266372680664]], [[6.984004497528076]], [[5.953611850738525]], [[5.549185276031494]], [[6.731591701507568]], [[6.322219371795654]], [[5.558440685272217]], [[5.827775001525879]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    
    class PrimitiveOp_b32c2bf197adaf65809bafd12183aea3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39a75e0616bd2366d01653b0881c6032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b32c2bf197adaf65809bafd12183aea3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4ea70571ddfabcf45c724d5c3f902e89(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_66ced19b59057a48cc22b43d68bd97b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ea70571ddfabcf45c724d5c3f902e89
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a963e74f76d31cf54c78ca4397c9f1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314efc29fd280f3ef667441f59effd18
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5d0c8172ebfece328895b6ecde60955(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a75166aa014e996054a7a5b6a219e5d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.512653350830078, 5.044649124145508, 4.655038356781006, 4.622213363647461, 4.632609844207764, 3.9768052101135254, 5.023960113525391, 4.7768354415893555, 5.421357154846191, 4.686634063720703, 5.148271083831787, 3.8738279342651367, 5.151643753051758, 4.582184314727783, 4.700376987457275, 4.343968868255615, 4.77421760559082, 4.917938232421875]], dtype='float32').reshape([1, 18]),
            ]


    class TestPrimitiveOp_6f124553a60e0b21ef7c258eee9e9141(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.to_tensor([[6.194968223571777, 5.21784782409668, 6.012714385986328, 4.90752649307251, 5.907154083251953, 5.492164611816406, 5.724031448364258, 5.579870223999023, 5.25261116027832, 5.262575149536133, 4.954087257385254, 5.3679680824279785, 4.764451503753662, 5.245046615600586, 6.278081893920898, 5.532181262969971, 6.717957496643066, 5.762195110321045, 5.715753078460693, 5.413527965545654, 4.952664375305176, 5.497104167938232, 5.146821022033691]], dtype='float32').reshape([1, 23]),
            ]


    class TestPrimitiveOp_bc19d6209077d4a8b13226311386b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c30432e99bc3640e73d3c357422f9ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13c8630def1e15cac1fded36845a8bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c786f6b71e120811fc843b7a2199cbda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d51ed97bf653e4091df4f137e6b7156(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d51ed97bf653e4091df4f137e6b7156(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58259bf0ed88ed1ad10b648f865afb5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.990804195404053]], [[7.303852081298828]], [[7.5863847732543945]], [[8.034000396728516]], [[6.682290077209473]], [[6.763601303100586]], [[8.005084037780762]], [[7.190065860748291]], [[7.1677021980285645]], [[7.97681188583374]], [[7.243475914001465]], [[7.156096935272217]], [[7.190069675445557]], [[7.3471455574035645]], [[7.202025890350342]], [[7.867246150970459]], [[7.899160861968994]], [[8.041866302490234]], [[7.189223289489746]], [[8.056235313415527]], [[7.894644737243652]], [[7.0927839279174805]], [[7.3161773681640625]], [[7.718397617340088]], [[6.824087142944336]], [[7.708375453948975]], [[8.410866737365723]], [[8.488816261291504]], [[7.6982855796813965]], [[7.605227947235107]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_71dd4ead797f782c482bff55ab75ca57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a977fd39537c15e19786a8b9ca88e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a977fd39537c15e19786a8b9ca88e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a977fd39537c15e19786a8b9ca88e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a977fd39537c15e19786a8b9ca88e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a977fd39537c15e19786a8b9ca88e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a977fd39537c15e19786a8b9ca88e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a977fd39537c15e19786a8b9ca88e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a977fd39537c15e19786a8b9ca88e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cdfa3080d79bd11fdd1cc787413f9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cdfa3080d79bd11fdd1cc787413f9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cdfa3080d79bd11fdd1cc787413f9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cdfa3080d79bd11fdd1cc787413f9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cdfa3080d79bd11fdd1cc787413f9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cdfa3080d79bd11fdd1cc787413f9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cdfa3080d79bd11fdd1cc787413f9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cdfa3080d79bd11fdd1cc787413f9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d1ad5611b21c5aefe3f2be6a19836f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d1ad5611b21c5aefe3f2be6a19836f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d1ad5611b21c5aefe3f2be6a19836f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d1ad5611b21c5aefe3f2be6a19836f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d1ad5611b21c5aefe3f2be6a19836f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d1ad5611b21c5aefe3f2be6a19836f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d1ad5611b21c5aefe3f2be6a19836f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d1ad5611b21c5aefe3f2be6a19836f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec3992230ba1cbabf381fb4ce0756668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec3992230ba1cbabf381fb4ce0756668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec3992230ba1cbabf381fb4ce0756668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec3992230ba1cbabf381fb4ce0756668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec3992230ba1cbabf381fb4ce0756668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec3992230ba1cbabf381fb4ce0756668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec3992230ba1cbabf381fb4ce0756668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec3992230ba1cbabf381fb4ce0756668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e97aee7825a840c0e8af96536856989(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.483332633972168]], [[8.580183982849121]], [[8.237578392028809]], [[8.947674751281738]], [[7.881694316864014]], [[8.806875228881836]], [[7.8643364906311035]], [[7.818826198577881]], [[8.723896026611328]], [[8.522080421447754]], [[7.446710586547852]], [[7.711270332336426]], [[8.313339233398438]], [[8.100343704223633]], [[7.585079669952393]], [[7.719268798828125]], [[8.0361967086792]], [[7.735347747802734]], [[8.298324584960938]], [[7.859841346740723]], [[8.472572326660156]], [[8.9426851272583]], [[7.678374767303467]], [[7.888156414031982]], [[8.077996253967285]], [[8.684389114379883]], [[7.907255172729492]], [[7.864846706390381]], [[7.371816635131836]], [[8.74284553527832]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_5e2c57027a6e1d0547f0a70d469c1df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cefe822b828d5ec399183c3c755afd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.6099520921707153]], [[1.3218650817871094]], [[1.5464215278625488]], [[1.3357179164886475]], [[1.4156224727630615]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_860c588262c2633b6b6300fa94b367b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.4961612224578857]], [[2.829477071762085]], [[2.832637071609497]], [[2.7194604873657227]], [[2.246098279953003]], [[2.6214306354522705]], [[3.1797657012939453]], [[3.231794834136963]], [[2.268321990966797]], [[2.6925604343414307]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_26620a79a73a9ca0502c0f3c20f0be44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd15df69654afea79383188438bfc698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.255369186401367]], [[7.026007175445557]], [[6.470477104187012]], [[7.313347339630127]], [[6.378765106201172]], [[7.421102523803711]], [[6.025125980377197]], [[5.941402912139893]], [[7.002050399780273]], [[7.059742450714111]], [[7.259979724884033]], [[6.680147171020508]], [[6.630357265472412]], [[7.096652984619141]], [[6.853813171386719]], [[6.618465423583984]], [[7.474974155426025]], [[6.692164897918701]], [[7.260404586791992]], [[6.467423439025879]], [[7.05855655670166]], [[6.48403787612915]], [[7.445367813110352]], [[6.541408538818359]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_6ad13be6175a20570488b3c1df6a5e9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fccbf3cc8e703180713c0875a809bb15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44b9fa0e27a121a2a79d8401b082d3ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9531cf157743dc5f3324c0ffa95aa83b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.84508752822876]], [[5.073159694671631]], [[5.19683837890625]], [[5.763732433319092]], [[5.3940558433532715]], [[5.156362056732178]], [[5.496768474578857]], [[4.507992744445801]], [[5.311750888824463]], [[5.81207799911499]], [[5.574479103088379]], [[5.305727958679199]], [[5.405195713043213]], [[5.113437175750732]], [[4.843127250671387]], [[5.608423709869385]], [[5.8059773445129395]], [[4.789793491363525]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_26620a79a73a9ca0502c0f3c20f0be44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6ac58c86979c7c58d4c72a266d7d7b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.374038219451904]], [[6.828001022338867]], [[6.4136176109313965]], [[6.300159454345703]], [[7.29701042175293]], [[7.216587543487549]], [[6.84116792678833]], [[7.992212772369385]], [[6.857626438140869]], [[6.7683000564575195]], [[6.817885875701904]], [[7.325019359588623]], [[6.7759528160095215]], [[6.628880023956299]], [[6.700521945953369]], [[6.536348342895508]], [[6.566918849945068]], [[7.208926200866699]], [[6.991861343383789]], [[7.377551078796387]], [[6.411651611328125]], [[7.489564418792725]], [[5.967304229736328]], [[5.613235950469971]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_3ccc372b66fc19088450c08831478521(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2c7ecc96761ad9969de4cc6e4837275(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab747fb1d6bd48de3adb7afa4baaec55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9948275089263916]], [[1.5264453887939453]], [[1.226925253868103]], [[1.1448898315429688]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_3ccc372b66fc19088450c08831478521(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b803c9d6529c90dc712f1e0c0b6e6510(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.9157357215881348]], [[2.75087308883667]], [[2.3742504119873047]], [[2.9814884662628174]], [[2.8051421642303467]], [[2.452035665512085]], [[2.8351681232452393]], [[2.9855198860168457]], [[2.973231077194214]], [[2.719639539718628]], [[2.89060378074646]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_bc19d6209077d4a8b13226311386b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26620a79a73a9ca0502c0f3c20f0be44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e7e0b05feba38eb6bce14a9838e7836(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.217819213867188]], [[7.3988237380981445]], [[8.072800636291504]], [[8.118842124938965]], [[8.233335494995117]], [[8.393362998962402]], [[7.98402214050293]], [[7.8484787940979]], [[7.684823989868164]], [[7.703281402587891]], [[8.268752098083496]], [[8.193763732910156]], [[8.559454917907715]], [[7.790647029876709]], [[8.329914093017578]], [[7.0845184326171875]], [[8.107961654663086]], [[8.191975593566895]], [[7.5113019943237305]], [[8.455312728881836]], [[7.586492538452148]], [[7.396222114562988]], [[7.046698093414307]], [[7.9782514572143555]], [[7.5932087898254395]], [[8.972037315368652]], [[7.453706741333008]], [[7.4995551109313965]], [[7.706169128417969]], [[8.040281295776367]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_104dfe59ce0482ea13fd160f6b984d54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fa13c005ab86215a9a9ed2f5fc301f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d37d6c7d26d45417de2a633df2358c72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.53007698059082]], [[4.731031894683838]], [[5.071558952331543]], [[4.159222602844238]], [[4.631857395172119]], [[4.282650470733643]], [[4.188570976257324]], [[4.655451774597168]], [[4.119109153747559]], [[4.581943988800049]], [[4.6004557609558105]], [[3.6670613288879395]], [[4.5001068115234375]], [[4.365671634674072]], [[4.397197723388672]], [[4.9889421463012695]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_e66a8becf841dd65ae646f46f3e55b04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6217e72e2d6a084236dfa5f92a735db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79aef7c7dc321002fb87468493b96c15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ff20543c4d4d17894bc0fd0423d21af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58f36fbb1ba47d2e49372f989b48475e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecc45a59933c59d424f8b258f4d4618a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd1002972d1dff64002b9ef3fac7e3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.688800811767578]], [[7.571673393249512]], [[7.303963661193848]], [[8.038948059082031]], [[7.213616371154785]], [[7.31330680847168]], [[8.098185539245605]], [[7.003323078155518]], [[7.517828464508057]], [[8.359289169311523]], [[8.598203659057617]], [[7.486069202423096]], [[7.572921276092529]], [[8.42301082611084]], [[7.036831855773926]], [[7.792578220367432]], [[8.356368064880371]], [[7.543519020080566]], [[7.518827438354492]], [[8.323749542236328]], [[7.809026718139648]], [[7.231009006500244]], [[8.173846244812012]], [[7.390050888061523]], [[7.922072887420654]], [[6.892157554626465]], [[8.021842956542969]], [[7.805875301361084]], [[7.734395980834961]], [[6.467280387878418]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_1b862e6dd13b068ebad1fd4f0e85348d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99509a32e7c0e28518f48e1c78a93087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccbc1f47ac12481de7084506693a20f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.0990824699401855]], [[7.608682155609131]], [[7.259829521179199]], [[6.562521934509277]], [[7.5483903884887695]], [[7.331745147705078]], [[6.962090969085693]], [[7.162661075592041]], [[7.244020938873291]], [[6.97843599319458]], [[7.068371295928955]], [[7.260822772979736]], [[7.184074878692627]], [[7.414368152618408]], [[6.817461967468262]], [[7.097340106964111]], [[6.920147895812988]], [[6.856722354888916]], [[6.658313274383545]], [[7.813545227050781]], [[7.304065227508545]], [[6.44976282119751]], [[6.9875168800354]], [[6.560308933258057]], [[7.412644863128662]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_26620a79a73a9ca0502c0f3c20f0be44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67c4a092c388891fadc1b71fdcbd38cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_633339e6ebc28e1b8c6d1f23d525d8b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a283fbef4df6949e57a354bef01f1e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a283fbef4df6949e57a354bef01f1e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_951752697fcacfc2235e679f8f1e7738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9adcea6b37342d519acef6ed2422f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f456660bb9cb96703d9218a16adb36e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.148994445800781]], [[5.436485767364502]], [[5.777336597442627]], [[5.813238620758057]], [[5.008029937744141]], [[5.292862892150879]], [[5.6771087646484375]], [[5.6961164474487305]], [[5.450930595397949]], [[5.666596412658691]], [[6.009949684143066]], [[5.628330230712891]], [[4.612966537475586]], [[5.137825012207031]], [[5.496090888977051]], [[4.571969032287598]], [[4.844310283660889]], [[5.447681427001953]], [[5.125885963439941]], [[5.789875507354736]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b862e6dd13b068ebad1fd4f0e85348d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74fe91b53977183f7ecd8c994f2d8a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.401656627655029]], [[5.851325988769531]], [[4.934147834777832]], [[6.043065071105957]], [[6.093008518218994]], [[5.5484490394592285]], [[5.932394504547119]], [[6.1603240966796875]], [[6.049316883087158]], [[6.410783767700195]], [[5.968497276306152]], [[5.426406383514404]], [[5.591534614562988]], [[5.737298488616943]], [[6.143668174743652]], [[5.389349460601807]], [[6.342817306518555]], [[5.814663410186768]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_fcd49ea2efdd62a017cd4a6591776b9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5d0c8172ebfece328895b6ecde60955(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_973f155ee32e2a97f84ed6da43221812(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e4564fec7062a9d535ce3043c068f00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69ecbad5233e2f737d01b5a266808a58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4edd2385176951e5b7a99b33a103ba66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4edd2385176951e5b7a99b33a103ba66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69ecbad5233e2f737d01b5a266808a58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4edd2385176951e5b7a99b33a103ba66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4edd2385176951e5b7a99b33a103ba66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1ff846fc41714490e22c218fe68b642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_106752a50edddc9bb8ca8fda40d97ea2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_106752a50edddc9bb8ca8fda40d97ea2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cdf27f26c7c5186403901e250a19230(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5736294ca7d44036194a70ba348da86e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5736294ca7d44036194a70ba348da86e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0ad62b91647cbc219fb7e5b97f1b9bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6263ec43b622d30adc19bfad7732cf91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6263ec43b622d30adc19bfad7732cf91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0ad62b91647cbc219fb7e5b97f1b9bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6263ec43b622d30adc19bfad7732cf91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6263ec43b622d30adc19bfad7732cf91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862605c34f99a7996e21011c66347d77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c86575cb9c2a5c0eb9df7d0b000604e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c86575cb9c2a5c0eb9df7d0b000604e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d344e71d42c482253022ca0a19db2709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee20b2c27e7c6ae90733f264015d479b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee20b2c27e7c6ae90733f264015d479b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5688fdb49e289cdd3d6ee6b253e8148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9adcea6b37342d519acef6ed2422f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ba4ce4f8bd7648da3aedc7d1909ca2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.168549060821533]], [[3.9504952430725098]], [[4.888088226318359]], [[4.88590669631958]], [[4.539901256561279]], [[4.17132043838501]], [[4.221492767333984]], [[4.6668829917907715]], [[4.126776218414307]], [[4.761663913726807]], [[4.332804203033447]], [[4.309434413909912]], [[4.466814041137695]], [[4.468220233917236]], [[3.8908464908599854]], [[4.757633209228516]], [[4.532098770141602]], [[4.14126443862915]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_71dd4ead797f782c482bff55ab75ca57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c7561c07b21de8882943707f3a00550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.38932991027832]], [[6.288872718811035]], [[4.80051326751709]], [[5.338171005249023]], [[5.631000518798828]], [[5.330127716064453]], [[5.0523834228515625]], [[5.389227867126465]], [[6.118121147155762]], [[6.174495220184326]], [[5.186315059661865]], [[6.076703071594238]], [[5.503951549530029]], [[5.625654220581055]], [[5.648626804351807]], [[5.764240741729736]], [[5.685770034790039]], [[4.80469274520874]], [[5.563747882843018]], [[5.11710786819458]], [[5.289473533630371]], [[5.886743068695068]], [[5.284122467041016]], [[5.664219856262207]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_3ec459f8eca5d9c79b91d00e012888f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45e77413712a524ed48bd3989922b42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.322781085968018]], [[4.311613082885742]], [[4.61679744720459]], [[4.344810485839844]], [[3.650794506072998]], [[4.33554220199585]], [[4.261188507080078]], [[3.9049463272094727]], [[3.7968709468841553]], [[3.907479763031006]], [[4.081305027008057]], [[4.279694080352783]], [[4.2520856857299805]], [[4.151562213897705]], [[3.9846291542053223]], [[4.152926445007324]], [[4.334284782409668]], [[3.5534818172454834]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_129c31020f47a9b865caad0f51fa26d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40b580f95bcd850098780ac3f2a61208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4673b5e395c4fed26b730fc226c5d40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.0413899421691895]], [[4.813213348388672]], [[4.553470134735107]], [[5.081023693084717]], [[4.93549108505249]], [[4.358945369720459]], [[4.546994686126709]], [[4.717782974243164]], [[4.752879619598389]], [[5.128915786743164]], [[5.44639253616333]], [[4.639196395874023]], [[4.349574565887451]], [[4.3120903968811035]], [[5.0812296867370605]], [[4.466093063354492]], [[5.013637065887451]], [[4.585987091064453]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_129c31020f47a9b865caad0f51fa26d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90f8409d9f05521693d6c9e899da9fef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41bbc6a78eadd78e700422b71fa4703a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58f5f21d28c82eb2f36dee36f7ed9cd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08a9ffa9d6c6eed2859a76d841053215(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0969a18b573906bb567ea629ff394256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0969a18b573906bb567ea629ff394256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08a9ffa9d6c6eed2859a76d841053215(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0969a18b573906bb567ea629ff394256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0969a18b573906bb567ea629ff394256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76b01cc534e10930cd79ae62133b8fe2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51bade47f80793956de22033b6d1bb0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51bade47f80793956de22033b6d1bb0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93a9dbc0b30ebd082f9304552a1e9cad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ae10e4e984441abbddf2939c13823c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ae10e4e984441abbddf2939c13823c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a18dbcb41e974ac44118928f706da0d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d595a2baf2439d78b21f44ee40a79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d595a2baf2439d78b21f44ee40a79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a18dbcb41e974ac44118928f706da0d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d595a2baf2439d78b21f44ee40a79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d595a2baf2439d78b21f44ee40a79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b09740a375bc95071a672e9ee7c0e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e97d0ed96ef27d8028b730c9eb68f9da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e97d0ed96ef27d8028b730c9eb68f9da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92dd4020c232d7cd04338cae532e9385(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14ddec3b09e7e286c6e01402dd0f269b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14ddec3b09e7e286c6e01402dd0f269b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_041995b3c0135b362f7ea4ba22b0639a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e76d0920fa6b8f10aa9c9c8d1030309(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecc45a59933c59d424f8b258f4d4618a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d68bb859b43547431dd00a415b632a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_374b0f15384544d85ac9c41c72acf443(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6560dbcd3fe340f8af3e768357943636(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_951752697fcacfc2235e679f8f1e7738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_480e49bb3159535097759d03ec8ef7b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_480e49bb3159535097759d03ec8ef7b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f9c0705592073ba4ccee4ad7be96291(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f9c0705592073ba4ccee4ad7be96291(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98faa0bc28e9d060a09aae6f9a87fd53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98faa0bc28e9d060a09aae6f9a87fd53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98faa0bc28e9d060a09aae6f9a87fd53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85f49de96b679cfae50dd7daefaca18d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85f49de96b679cfae50dd7daefaca18d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85f49de96b679cfae50dd7daefaca18d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cead3724befffef8eae23fa6e2ac9c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cead3724befffef8eae23fa6e2ac9c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cead3724befffef8eae23fa6e2ac9c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_614709445a27f1bfbac2c97c885a792a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_614709445a27f1bfbac2c97c885a792a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dc6c20ceca4cf4bef5c877f595399ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e53e769cdcb20cc08ab4dc0728f0c3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeaad98d168fa77f71e52cccfa228fe6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffe82c79499223184632ec5056e8d8d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fee8b4fc1866cd68c4ab6546a7af1b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5857fe8ab2ad136fac7e8d0f01adf36a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97c43522d835225992b75618634c32ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_251e358d5f1f27f507425e1b0f4c2518(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebe6f160c7c72165d086dd2775421df6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_241e3ec73e2f91a6ab548182d59a81e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d444fb38a73cba56783de870196e00e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d80f92a14c87de9636280215e0f3ca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59878426d356dc5c45a2eccb4001bd83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.496325969696045]], [[4.676479816436768]], [[4.272510528564453]], [[4.639997959136963]], [[3.9382097721099854]], [[4.531309127807617]], [[5.07270622253418]], [[4.9114837646484375]], [[4.314321994781494]], [[4.516523838043213]], [[4.526065349578857]], [[4.657249450683594]], [[4.689118385314941]], [[4.198484420776367]], [[4.094197750091553]], [[4.901497840881348]], [[5.006951808929443]], [[4.796614646911621]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_9adcea6b37342d519acef6ed2422f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_893213a74632e5153d63349416fef481(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc19d6209077d4a8b13226311386b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61af10d3f8e6f6d51bca138e4fef3fbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9258e6cbba09f995b672804ba92f0e70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.3317461013793945]], [[3.7822155952453613]], [[3.8322110176086426]], [[3.857466220855713]], [[3.882558584213257]], [[3.3941285610198975]], [[3.9580178260803223]], [[3.9200615882873535]], [[4.120512962341309]], [[3.724442958831787]], [[3.991763114929199]], [[3.8680496215820312]], [[3.990579843521118]], [[3.73437762260437]], [[3.345764636993408]], [[3.999772071838379]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_d1f787b6b5bb900679a7767cc7b2d932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8233c6b88bf1c36bba07bc16d851efd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.674563407897949]], [[3.7325024604797363]], [[4.349916934967041]], [[4.446511745452881]], [[4.36876106262207]], [[4.073473930358887]], [[4.822630405426025]], [[5.114545822143555]], [[5.140368461608887]], [[4.044999599456787]], [[4.667781352996826]], [[4.308403968811035]], [[4.528727054595947]], [[4.867443561553955]], [[3.711825132369995]], [[4.7906904220581055]], [[4.651676654815674]], [[3.9279441833496094]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_81cb6685c630facdf9c8c7a7b3185cba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4343864917755127]], [[1.2709107398986816]], [[1.1146671772003174]], [[1.3615840673446655]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_81e13858e325f69dbe312b6930e7f705(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f1237704004c6806b94739dd7fae521(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4779971e8fb8ddd99cf979c1fdff803c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4779971e8fb8ddd99cf979c1fdff803c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f1237704004c6806b94739dd7fae521(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4779971e8fb8ddd99cf979c1fdff803c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4779971e8fb8ddd99cf979c1fdff803c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de39690ea78739697511db6788c5f78d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a557bd5e05349e7abee1468f0d8ca99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a557bd5e05349e7abee1468f0d8ca99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b63d9a583ae694c79d67fc22fe636c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1868bd651c344d6094d823f18cd022e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1868bd651c344d6094d823f18cd022e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06608ca7262d41595a810c35faf2de80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8077408bc856f649f3d14198cd09c8a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8077408bc856f649f3d14198cd09c8a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06608ca7262d41595a810c35faf2de80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8077408bc856f649f3d14198cd09c8a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8077408bc856f649f3d14198cd09c8a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2fbaa96e5ad6301f9ec66953c250fe0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74a4eda77b613754cca4b8f8d67e2d1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74a4eda77b613754cca4b8f8d67e2d1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf900f02259eea470323abd70c08f243(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3034fcc566a5dc4fd55f7ef862e993d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3034fcc566a5dc4fd55f7ef862e993d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24ee0305e8a393c3a1d25c808b9ad2a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9adcea6b37342d519acef6ed2422f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_633339e6ebc28e1b8c6d1f23d525d8b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e6e714984bf361a6ed8b69dd3e29854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d15ade59d8b9229ede264d161094761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_502306313038655bb4a5325ddde086ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d68bb859b43547431dd00a415b632a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9adcea6b37342d519acef6ed2422f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_129c31020f47a9b865caad0f51fa26d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a614d0e6bc9217b7207f52f3eb9a3ced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.935728549957275]], [[5.193583011627197]], [[5.405919075012207]], [[5.343483924865723]], [[4.915558338165283]], [[5.3574299812316895]], [[5.5367655754089355]], [[5.295175075531006]], [[5.847105503082275]], [[4.853701591491699]], [[5.489466667175293]], [[5.752553462982178]], [[5.8476457595825195]], [[5.416018009185791]], [[5.76278829574585]], [[5.513062477111816]], [[5.44843864440918]], [[5.8336710929870605]], [[5.499468803405762]], [[5.231005668640137]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_1955f3b2737355adc6140e76c4a8ddde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_248ade3731d91b2c0b750354b6938f53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.838440418243408]], [[3.725783348083496]], [[3.4627926349639893]], [[2.86716365814209]], [[3.1309502124786377]], [[3.1671664714813232]], [[2.832571506500244]], [[3.4852497577667236]], [[2.6229331493377686]], [[3.4359145164489746]], [[2.955007553100586]], [[3.234879970550537]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_bddb64ba8af852decdf43e9db3fe353b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.189188003540039]], [[5.522616386413574]], [[6.239436626434326]], [[5.228600025177002]], [[5.076842784881592]], [[4.971395969390869]], [[5.320101737976074]], [[5.200826168060303]], [[5.677047252655029]], [[4.709473133087158]], [[4.8291730880737305]], [[4.934872150421143]], [[5.219989776611328]], [[4.80227518081665]], [[5.564419269561768]], [[5.456704139709473]], [[4.736117362976074]], [[5.041024684906006]], [[5.031208038330078]], [[5.072656154632568]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_af088ad708b91570d1288f12725c0529(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.9690845012664795]], [[2.9818687438964844]], [[2.8145813941955566]], [[3.0310134887695312]], [[3.1999592781066895]], [[2.9077987670898438]], [[3.10152006149292]], [[2.8881208896636963]], [[3.345881462097168]], [[2.6285908222198486]], [[3.7109107971191406]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_502306313038655bb4a5325ddde086ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6d0a485f59d1055941086c7c6d0c8c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa2f7db584567bea9f6ae4a439ceaaa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.728445053100586]], [[4.151237487792969]], [[3.621351718902588]], [[3.568361759185791]], [[4.5313801765441895]], [[4.230593204498291]], [[3.3904149532318115]], [[3.592285394668579]], [[4.054468154907227]], [[3.8145062923431396]], [[3.9085946083068848]], [[3.8280105590820312]], [[3.6504974365234375]], [[3.925114393234253]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_c177002d910489fcdbe974794ee1bdef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fccbf3cc8e703180713c0875a809bb15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc19d6209077d4a8b13226311386b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f01e19e3f78e4ebd712b2aeeaaa7c74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.040356636047363]], [[4.171722888946533]], [[6.066471576690674]], [[4.901216506958008]], [[4.848870754241943]], [[4.409139633178711]], [[5.113746643066406]], [[5.3654465675354]], [[4.759063243865967]], [[5.03289794921875]], [[4.981382369995117]], [[5.127835273742676]], [[4.808407783508301]], [[5.101073741912842]], [[5.186868667602539]], [[5.055799961090088]], [[5.173654079437256]], [[4.846164226531982]], [[4.867003917694092]], [[4.343432426452637]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_1a9f854099bafde55b6024dc416314f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a9f854099bafde55b6024dc416314f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a9f854099bafde55b6024dc416314f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a9f854099bafde55b6024dc416314f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9b5929d6d3ff4a6d29d62e73f59f512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33790.01171875]], [[33246.22265625]], [[30786.228515625]], [[30725.201171875]], [[38834.828125]], [[33852.90234375]]], [[[34128.40234375]], [[33572.453125]], [[31094.193359375]], [[31031.59765625]], [[39217.91015625]], [[34191.59765625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_163bcdbe4469ba1d535c73871307c0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[36145.1640625]], [[42202.109375]], [[40099.84375]], [[37980.41015625]], [[31233.35546875]], [[32060.4609375]]], [[[36810.8984375]], [[42984.38671875]], [[40846.484375]], [[38684.21484375]], [[31809.748046875]], [[32650.68359375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_0a2a77704a362b4dfd7f199ba5dd9da7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[32827.54296875]], [[38912.5078125]], [[41232.10546875]], [[42047.49609375]], [[42913.828125]], [[47525.63671875]]], [[[32996.40625]], [[39109.4140625]], [[41443.84765625]], [[42258.4296875]], [[43128.10546875]], [[47765.79296875]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_9572858bb73941beb9b252be299f0f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34932.6171875]], [[30921.677734375]], [[40886.67578125]], [[36311.15234375]], [[43067.72265625]], [[48601.33203125]]], [[[35221.71875]], [[31168.6796875]], [[41222.20703125]], [[36605.85546875]], [[43419.46875]], [[48995.37890625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_57898f52bf7690bf2f30faf4969a1d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57898f52bf7690bf2f30faf4969a1d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57898f52bf7690bf2f30faf4969a1d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57898f52bf7690bf2f30faf4969a1d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57898f52bf7690bf2f30faf4969a1d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57898f52bf7690bf2f30faf4969a1d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57898f52bf7690bf2f30faf4969a1d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57898f52bf7690bf2f30faf4969a1d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c06301d72466e14ba3b2ad49703a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c06301d72466e14ba3b2ad49703a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c06301d72466e14ba3b2ad49703a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c06301d72466e14ba3b2ad49703a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c06301d72466e14ba3b2ad49703a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c06301d72466e14ba3b2ad49703a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c06301d72466e14ba3b2ad49703a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c06301d72466e14ba3b2ad49703a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb8070a181e64cd1ba84f6a78e402efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb8070a181e64cd1ba84f6a78e402efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb8070a181e64cd1ba84f6a78e402efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb8070a181e64cd1ba84f6a78e402efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb8070a181e64cd1ba84f6a78e402efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb8070a181e64cd1ba84f6a78e402efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb8070a181e64cd1ba84f6a78e402efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb8070a181e64cd1ba84f6a78e402efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_129c31020f47a9b865caad0f51fa26d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe246f29847c245f6d2bc5dff7b8781(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.25681209564209]], [[8.124125480651855]], [[7.101140975952148]], [[8.303726196289062]], [[7.532835006713867]], [[7.419724464416504]], [[7.211884498596191]], [[7.81193208694458]], [[7.384727478027344]], [[7.545863151550293]], [[8.518484115600586]], [[8.447880744934082]], [[7.607048511505127]], [[8.261672973632812]], [[8.002284049987793]], [[8.208133697509766]], [[7.965606689453125]], [[8.26777458190918]], [[7.481735706329346]], [[7.620168685913086]], [[9.100288391113281]], [[7.642550945281982]], [[7.4412007331848145]], [[7.278196811676025]], [[7.8947672843933105]], [[8.380417823791504]], [[7.684148788452148]], [[8.735790252685547]], [[8.403761863708496]], [[7.420577049255371]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_e01c9ea1509304213b81c3ec89342145(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.159058570861816]], [[8.9575777053833]], [[7.920828342437744]], [[8.26949405670166]], [[8.606870651245117]], [[7.703291893005371]], [[9.033196449279785]], [[7.2510600090026855]], [[8.894136428833008]], [[8.381927490234375]], [[8.117466926574707]], [[8.615775108337402]], [[8.195988655090332]], [[8.25282096862793]], [[9.394417762756348]], [[8.261106491088867]], [[8.069947242736816]], [[8.460314750671387]], [[9.121085166931152]], [[8.430448532104492]], [[8.327909469604492]], [[7.7775983810424805]], [[8.79182243347168]], [[7.773296356201172]], [[8.375483512878418]], [[8.84751033782959]], [[8.769116401672363]], [[7.66787052154541]], [[8.125309944152832]], [[7.6072678565979]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_e3c60f9e22bb9c82d3b96abcacddccff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94b70f1e0f7d8c747738b561e82994f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.9664530754089355]], [[6.8404951095581055]], [[8.002744674682617]], [[7.52061128616333]], [[7.19194221496582]], [[6.707631587982178]], [[6.572324752807617]], [[6.8679728507995605]], [[6.855906963348389]], [[7.773515701293945]], [[7.6090497970581055]], [[7.906286716461182]], [[6.690423011779785]], [[6.965617656707764]], [[7.726587295532227]], [[7.490701198577881]], [[7.930479526519775]], [[7.766480922698975]], [[7.732168197631836]], [[6.953736305236816]], [[6.223891258239746]], [[7.326486110687256]], [[7.540115833282471]], [[7.350153923034668]], [[7.1114702224731445]], [[7.341914653778076]], [[6.942921161651611]], [[7.209981441497803]], [[7.393255710601807]], [[7.669460296630859]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_963281940843c6674b73bb7dd5301eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9adcea6b37342d519acef6ed2422f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abca3a555152747cd7a16578dc5a451f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.4437432289123535]], [[8.28723430633545]], [[7.912923812866211]], [[7.163331508636475]], [[7.706422805786133]], [[7.904669761657715]], [[7.870941162109375]], [[8.095754623413086]], [[7.887270450592041]], [[7.398651599884033]], [[8.18539047241211]], [[7.227769374847412]], [[8.68224811553955]], [[8.48115062713623]], [[7.778359413146973]], [[7.674652576446533]], [[8.059738159179688]], [[7.699238300323486]], [[8.101340293884277]], [[7.794434070587158]], [[7.354639530181885]], [[7.219238758087158]], [[7.487193584442139]], [[7.905284404754639]], [[7.686999797821045]], [[8.279914855957031]], [[8.093050956726074]], [[7.652628421783447]], [[8.023968696594238]], [[7.812108516693115]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_7dfb513e36f9dcae9a8d35354e5950b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.8361976146698]], [[3.135993003845215]], [[2.961843490600586]], [[3.4332213401794434]], [[3.0051279067993164]], [[2.8730978965759277]], [[3.358252763748169]], [[3.0638809204101562]], [[3.0245022773742676]], [[3.537898063659668]], [[2.83929705619812]], [[2.9802045822143555]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_d2c558b2292a13087d3fb722aa892185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.4488754272460938]], [[3.1694984436035156]], [[3.566474676132202]], [[3.0803067684173584]], [[3.2484779357910156]], [[2.6101138591766357]], [[3.772961139678955]], [[3.672159194946289]], [[3.6704392433166504]], [[3.145970344543457]], [[3.907890558242798]], [[3.1002845764160156]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_f9e738e7e4ed8e96a5b0d5fd91d0b828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.898226261138916]], [[7.46968412399292]], [[6.872394561767578]], [[6.250451564788818]], [[7.033980369567871]], [[6.551626205444336]], [[7.342005252838135]], [[7.041738033294678]], [[6.796108722686768]], [[6.893855571746826]], [[6.811644554138184]], [[6.453948020935059]], [[6.911863803863525]], [[6.331982612609863]], [[7.10107946395874]], [[6.501152992248535]], [[7.3300909996032715]], [[6.193831443786621]], [[6.802927494049072]], [[6.822544097900391]], [[6.546031951904297]], [[6.251367568969727]], [[7.175201892852783]], [[5.89363431930542]], [[6.777373313903809]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_ff281d444e6690f647db4ceec431791c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49f9b281b2f84bef1a64123706ed0205(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39f6e1ad9fb1b664cfd7ae3596a59ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f08033c06dee422e25f0e5c767706944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e122933eecd16b4d695462d01bd1ad10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6deb003341b12b591b787ac0f1ee8a43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.34008264541626]], [[4.743483066558838]], [[5.1139140129089355]], [[4.816980361938477]], [[5.030374050140381]], [[5.1157050132751465]], [[5.042732238769531]], [[4.310696601867676]], [[5.24165153503418]], [[5.031465530395508]], [[4.28033447265625]], [[4.85125207901001]], [[5.221953392028809]], [[5.398641586303711]], [[4.63865327835083]], [[4.977076053619385]], [[5.138279438018799]], [[4.5899553298950195]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_327c80cf4e69b3ddc74e26b787deb685(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52d7110c513b01bb86409fe03faf4f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.1509110927581787]], [[1.1300290822982788]], [[1.3730714321136475]], [[1.2706753015518188]], [[1.0934555530548096]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_88174ba2a51b03f3dcc275151794e056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.8189101219177246]], [[2.6422502994537354]], [[2.8085169792175293]], [[2.7595033645629883]], [[2.954659938812256]], [[2.689716339111328]], [[2.191497564315796]], [[2.3258798122406006]], [[2.354254722595215]], [[2.6504807472229004]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_c46854d7e0c6e7f67530acb8fa49a917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.700591564178467]], [[5.198940277099609]], [[5.845884799957275]], [[5.625529766082764]], [[5.1735100746154785]], [[4.8036723136901855]], [[5.090089797973633]], [[4.773566722869873]], [[5.558661937713623]], [[5.0450568199157715]], [[5.424689292907715]], [[4.559719562530518]], [[5.617161273956299]], [[4.593595504760742]], [[5.566433906555176]], [[5.211007118225098]], [[5.107222557067871]], [[4.509981632232666]], [[4.843706130981445]], [[5.598837852478027]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_bc19d6209077d4a8b13226311386b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_963281940843c6674b73bb7dd5301eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b862e6dd13b068ebad1fd4f0e85348d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99509a32e7c0e28518f48e1c78a93087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99fc8d6b37dfcd03a6e5ad36824bfc06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.023273944854736]], [[5.066927909851074]], [[4.140905857086182]], [[5.343730926513672]], [[5.36994743347168]], [[4.6580400466918945]], [[5.503517150878906]], [[4.593698024749756]], [[4.692727088928223]], [[5.1036295890808105]], [[5.059436321258545]], [[5.221367359161377]], [[5.29227876663208]], [[4.9216837882995605]], [[5.376871109008789]], [[6.023849010467529]], [[4.857785701751709]], [[4.9857001304626465]], [[4.833615303039551]], [[5.491693496704102]], [[5.115340232849121]], [[5.572267532348633]], [[5.060335159301758]], [[5.243594646453857]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_637be84cb07254b4713a66c8c9e0597f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194748cde49e95f613d564753808f9a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.470754384994507]], [[3.0341553688049316]], [[3.070213794708252]], [[2.381013870239258]], [[2.537949323654175]], [[3.0229506492614746]], [[2.40401554107666]], [[2.5496742725372314]], [[2.6882126331329346]], [[2.872746229171753]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_51b5490be2a7e707863977b1a874f982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab6cb7c3a97a7edbac0eb41dda308cfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f421047159269b15636f0c5a7c44d0c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d694265e3a67a7353020001e06c35f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81ff33048e5a87419a61dba5389c3865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.73128604888916]], [[5.034897804260254]], [[4.891191005706787]], [[4.268646717071533]], [[4.652425289154053]], [[4.6083478927612305]], [[4.871151447296143]], [[4.301667213439941]], [[4.273050785064697]], [[4.274194717407227]], [[4.2991414070129395]], [[5.229434967041016]], [[4.240005970001221]], [[4.6630859375]], [[5.333066940307617]], [[4.8534698486328125]], [[4.757251262664795]], [[3.8645853996276855]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_e7fd67ac0556fa9e1001a9fe6508a64c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.777217388153076, 8.48762035369873, 7.249322414398193, 7.724056243896484, 7.930503845214844, 7.815942287445068, 8.609721183776855, 8.345749855041504, 9.051900863647461, 8.07912826538086, 8.785380363464355, 8.09261417388916, 8.483768463134766, 7.842811107635498, 8.470643043518066, 9.749964714050293, 8.221514701843262, 7.603100299835205, 8.097103118896484, 8.626155853271484, 8.103931427001953, 7.6056227684021, 7.974206924438477, 8.232665061950684, 8.936210632324219, 8.685194969177246, 8.046462059020996, 7.636215686798096, 7.272360801696777, 8.230209350585938]], dtype='float32').reshape([1, 30]),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d15ade59d8b9229ede264d161094761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ff6142f3b08af4bba0e831fa59afd74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.9253621101379395]], [[8.135875701904297]], [[7.322157382965088]], [[8.395102500915527]], [[7.788322448730469]], [[8.539194107055664]], [[7.630413055419922]], [[7.462310314178467]], [[7.580341815948486]], [[7.088729381561279]], [[7.536751747131348]], [[7.9274821281433105]], [[7.52095365524292]], [[7.826623916625977]], [[7.471090316772461]], [[7.233271598815918]], [[8.233546257019043]], [[8.312676429748535]], [[8.348246574401855]], [[8.340503692626953]], [[7.820869445800781]], [[8.069595336914062]], [[8.100695610046387]], [[7.995584011077881]], [[7.848565101623535]], [[7.846622467041016]], [[7.562673091888428]], [[8.764762878417969]], [[8.024958610534668]], [[8.593799591064453]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_545b51c66489568d5748a6783c53a966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.7045177221298218]], [[1.6164036989212036]], [[1.4427224397659302]], [[1.5230357646942139]], [[1.8313349485397339]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_931667a5086f35143496f5d127de9090(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.6212942600250244]], [[2.881885051727295]], [[2.826146125793457]], [[3.1407322883605957]], [[2.967379093170166]], [[2.947307586669922]], [[2.8775577545166016]], [[2.7747371196746826]], [[2.985468864440918]], [[3.1787304878234863]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_6440c1cb50929c989825542924cba3c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.665043354034424]], [[6.21737813949585]], [[6.037357807159424]], [[5.357994556427002]], [[5.321767330169678]], [[5.587750434875488]], [[5.597776412963867]], [[5.610191822052002]], [[5.384403228759766]], [[5.28219747543335]], [[5.194252967834473]], [[5.6317667961120605]], [[5.777074337005615]], [[5.500901222229004]], [[5.264638423919678]], [[5.222471237182617]], [[6.135081768035889]], [[5.563098430633545]], [[5.966292381286621]], [[5.470348834991455]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_bc19d6209077d4a8b13226311386b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a47e13a7d93d319be60910b64ff41b28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.8203845024108887]], [[4.383570194244385]], [[4.741254806518555]], [[3.7061283588409424]], [[3.4336838722229004]], [[4.516073703765869]], [[4.192562103271484]], [[4.824357032775879]], [[4.380889415740967]], [[4.580226421356201]], [[4.271740436553955]], [[4.298461437225342]], [[5.017632007598877]], [[4.143437385559082]], [[4.676990032196045]], [[3.806002140045166]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_ecc45a59933c59d424f8b258f4d4618a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44980bb929d5cf1a7ecbd50c3f8c31b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b253ee4a5c41dc55dfcd19efe212835c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b253ee4a5c41dc55dfcd19efe212835c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b253ee4a5c41dc55dfcd19efe212835c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b253ee4a5c41dc55dfcd19efe212835c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b253ee4a5c41dc55dfcd19efe212835c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b253ee4a5c41dc55dfcd19efe212835c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b253ee4a5c41dc55dfcd19efe212835c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b253ee4a5c41dc55dfcd19efe212835c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93e26455d5cc8fba1cc25185e6bf0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93e26455d5cc8fba1cc25185e6bf0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93e26455d5cc8fba1cc25185e6bf0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93e26455d5cc8fba1cc25185e6bf0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93e26455d5cc8fba1cc25185e6bf0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93e26455d5cc8fba1cc25185e6bf0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93e26455d5cc8fba1cc25185e6bf0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93e26455d5cc8fba1cc25185e6bf0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6e0cf119d36b3c34a9f130e58a73e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6e0cf119d36b3c34a9f130e58a73e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6e0cf119d36b3c34a9f130e58a73e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6e0cf119d36b3c34a9f130e58a73e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6e0cf119d36b3c34a9f130e58a73e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6e0cf119d36b3c34a9f130e58a73e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6e0cf119d36b3c34a9f130e58a73e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6e0cf119d36b3c34a9f130e58a73e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b98b9e2c0dd54ecf1a2cff6939fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1c976d36e221b53fca4ef5afda2f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1955f3b2737355adc6140e76c4a8ddde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c177002d910489fcdbe974794ee1bdef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90b0e74c4e6a1e2401f2230344ee8ff1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.6342556476593018]], [[3.1763720512390137]], [[3.495896339416504]], [[3.4828929901123047]], [[3.3462088108062744]], [[3.6012353897094727]], [[2.8251118659973145]], [[3.4390714168548584]], [[2.9350075721740723]], [[3.2876410484313965]], [[3.9005303382873535]], [[3.4099371433258057]], [[3.195777416229248]], [[3.1334729194641113]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_42e4b9d38fe87a26b61889d2da46ee2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.698542594909668]], [[5.4758477210998535]], [[4.34893274307251]], [[5.1414713859558105]], [[5.090174674987793]], [[5.6204833984375]], [[5.572890281677246]], [[5.212032318115234]], [[5.199526309967041]], [[4.979649543762207]], [[4.388289451599121]], [[4.937069416046143]], [[4.850468158721924]], [[4.67232084274292]], [[4.780160903930664]], [[5.114904880523682]], [[5.140608310699463]], [[4.899909019470215]], [[4.694981575012207]], [[5.400130748748779]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_fab77dce207c9d6b9dd8f368dc0aa235(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b862e6dd13b068ebad1fd4f0e85348d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc5ebdf4c5d6a9dce6e93de9f579641e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.587968826293945]], [[7.046293258666992]], [[7.726832866668701]], [[6.813288688659668]], [[7.791248321533203]], [[6.617620468139648]], [[7.223494529724121]], [[6.607419967651367]], [[7.120966911315918]], [[8.214742660522461]], [[7.768636703491211]], [[7.641348838806152]], [[6.252624988555908]], [[7.227708339691162]], [[6.829111099243164]], [[7.346861839294434]], [[7.112125873565674]], [[6.733704090118408]], [[8.058210372924805]], [[7.438938617706299]], [[7.837202548980713]], [[7.037036895751953]], [[7.634639263153076]], [[7.035560607910156]], [[7.132640838623047]], [[6.164754867553711]], [[7.45093297958374]], [[7.50039005279541]], [[7.918460369110107]], [[7.027027606964111]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_26620a79a73a9ca0502c0f3c20f0be44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc19d6209077d4a8b13226311386b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44980bb929d5cf1a7ecbd50c3f8c31b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e1743624057739bca1d69fa1006ec30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_404a6e42a1504a1ff3ed1e1a0320d7ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_807d579c81c12432f07d5befe83732c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_807d579c81c12432f07d5befe83732c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_404a6e42a1504a1ff3ed1e1a0320d7ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_807d579c81c12432f07d5befe83732c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_807d579c81c12432f07d5befe83732c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa255976e6c35c7d5eb3232d0fbab379(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b31f6d7ab75b9d8377d2af1091a2f84d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b31f6d7ab75b9d8377d2af1091a2f84d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ec2bb24fd6157f2a9f37faabc984d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ff6e92083eae2eadf0791f44cfd619a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ff6e92083eae2eadf0791f44cfd619a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4f0911c7ecaa7967d2128c673c034c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8ed809e4bcccd576bc3e742a410a57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8ed809e4bcccd576bc3e742a410a57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4f0911c7ecaa7967d2128c673c034c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8ed809e4bcccd576bc3e742a410a57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8ed809e4bcccd576bc3e742a410a57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71be6479c21be2383281df9808df5cd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf267fd741b61a4f61fedfe5d613246(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf267fd741b61a4f61fedfe5d613246(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c440a349e90658fdf92d6a4648c2f16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0283c30bfdd05206ca18bad33139ace7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0283c30bfdd05206ca18bad33139ace7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_233f4ee46da73f49f06c445f18961a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_963281940843c6674b73bb7dd5301eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59170aece9b20712cac85f7e72ed46d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_502306313038655bb4a5325ddde086ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5f15982dfab63168111c94e26251a1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.113337516784668]], [[6.361061096191406]], [[6.383002281188965]], [[6.8510637283325195]], [[6.504920482635498]], [[6.268310546875]], [[6.040866851806641]], [[6.8944292068481445]], [[6.598022937774658]], [[6.050089359283447]], [[6.557819843292236]], [[6.211484432220459]], [[6.971176624298096]], [[5.849992752075195]], [[5.925839424133301]], [[7.056762218475342]], [[6.003842830657959]], [[6.703570365905762]], [[7.226595878601074]], [[7.061709403991699]], [[6.613278865814209]], [[6.183558464050293]], [[6.57665491104126]], [[6.672055721282959]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f72a24e798f46f77e0fa672bfd0cd8eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.585529327392578]], [[5.8684468269348145]], [[6.182231426239014]], [[5.794556617736816]], [[6.117100715637207]], [[6.029637813568115]], [[7.346761703491211]], [[5.943231582641602]], [[6.368186950683594]], [[5.535693645477295]], [[7.286806583404541]], [[5.88812780380249]], [[6.799605369567871]], [[6.342577934265137]], [[5.253039836883545]], [[6.462026596069336]], [[6.507450580596924]], [[5.606255054473877]], [[5.662771701812744]], [[5.6854329109191895]], [[5.967799186706543]], [[6.681526184082031]], [[6.520153045654297]], [[5.843494415283203]], [[6.265067100524902]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_6eb1592ab7831579fdd7ff2ef0e2addf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.105079174041748]], [[3.126370906829834]], [[3.035320281982422]], [[2.878377676010132]], [[2.9810948371887207]], [[3.045088052749634]], [[3.228938579559326]], [[3.203735828399658]], [[3.3749754428863525]], [[3.210139513015747]], [[2.8544631004333496]], [[3.5723633766174316]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc19d6209077d4a8b13226311386b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b862e6dd13b068ebad1fd4f0e85348d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0db12dc46be615c622aff5849e5b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af2c3a10e5de7483d3c9d15b5233edf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecc45a59933c59d424f8b258f4d4618a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6c0403deb18dbc2a5f184d4abacdb31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc19d6209077d4a8b13226311386b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82bb125f9bf2a79903abe6dca05bed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68ef574f7458db24d859e2bd64d3274f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f50397ac823d2e58d9f9913431882de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[728.019775390625]], [[746.1237182617188]], [[692.19873046875]], [[762.6597900390625]], [[671.3251953125]], [[708.3541870117188]], [[721.9227905273438]], [[783.6730346679688]], [[717.413330078125]], [[715.1480712890625]], [[717.2114868164062]], [[724.47607421875]], [[700.197509765625]], [[678.7390747070312]], [[709.4534301757812]], [[765.017333984375]], [[743.7413940429688]], [[696.52001953125]], [[729.6466674804688]], [[724.2402954101562]], [[742.330078125]], [[707.8065185546875]], [[697.4320068359375]], [[735.0343017578125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_93723fec9b2cf5d97b833ba0c9cf0a21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[88.4103012084961]], [[83.92156219482422]], [[87.11354064941406]], [[95.5699234008789]], [[85.66553497314453]], [[73.65477752685547]], [[91.61893463134766]], [[89.295654296875]], [[89.7334213256836]], [[88.86317443847656]], [[88.36162567138672]], [[95.03060150146484]], [[92.3189926147461]], [[95.11656188964844]], [[86.61528015136719]], [[90.17096710205078]], [[87.8534164428711]], [[89.96961212158203]], [[91.15142822265625]], [[86.67532348632812]], [[84.73998260498047]], [[90.36804962158203]], [[85.57946014404297]], [[88.91419219970703]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_fbb36874499dd1000b160b5f2ae5d770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41.77033233642578]], [[36.550838470458984]], [[38.81454849243164]], [[40.53050231933594]], [[39.61156463623047]], [[37.262638092041016]], [[38.70488739013672]], [[37.151390075683594]], [[40.440311431884766]], [[40.4919319152832]], [[36.12879180908203]], [[41.12125015258789]], [[37.479007720947266]], [[38.575531005859375]], [[36.90913391113281]], [[40.257232666015625]], [[34.73536682128906]], [[38.0360221862793]], [[35.37649154663086]], [[38.790096282958984]], [[36.533573150634766]], [[40.45515823364258]], [[34.850669860839844]], [[33.67195129394531]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_ac41a72085d846b71cfe27cf092499f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[26.883615493774414]], [[26.930633544921875]], [[27.861154556274414]], [[25.487024307250977]], [[27.49747085571289]], [[28.103042602539062]], [[26.305788040161133]], [[28.329017639160156]], [[25.491519927978516]], [[28.231576919555664]], [[28.3348331451416]], [[28.69142723083496]], [[25.975393295288086]], [[26.011674880981445]], [[26.594940185546875]], [[25.055524826049805]], [[29.05447769165039]], [[30.182493209838867]], [[26.662887573242188]], [[25.961088180541992]], [[24.21113395690918]], [[26.02586555480957]], [[25.330015182495117]], [[26.766361236572266]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_70913b691b7d80b25f545e634f3916d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37687.91796875]], [[31513.66796875]], [[32339.71484375]], [[26676.244140625]], [[32849.390625]], [[37629.57421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_1bc19126ef40286c060e1169d3be0650(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41447.62890625]], [[40241.578125]], [[39892.1015625]], [[34655.28125]], [[42560.5078125]], [[35762.80078125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_b944545fe2a5b853ad967f6e88490030(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33178.44921875]], [[44838.65625]], [[41198.328125]], [[40192.58984375]], [[42689.53515625]], [[40445.8671875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_d1619cbe341ffba0e92b58ddf8693cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[36067.48828125]], [[44561.07421875]], [[46742.91796875]], [[42671.6640625]], [[43674.9453125]], [[47246.0625]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_b2cfffcb6ff1255dd779e76128697d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26620a79a73a9ca0502c0f3c20f0be44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff281d444e6690f647db4ceec431791c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cea70bca2c1e1eca51d82da5668e622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fbccf5b32798574d4216d57b0a966b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.819546222686768]], [[6.063246250152588]], [[6.408529758453369]], [[5.9528303146362305]], [[5.815957546234131]], [[6.398426055908203]], [[5.963871955871582]], [[6.095754146575928]], [[5.961697101593018]], [[5.839191436767578]], [[6.142646789550781]], [[5.494499206542969]], [[5.5466766357421875]], [[5.561925411224365]], [[6.77968692779541]], [[6.119173049926758]], [[5.838266372680664]], [[6.984004497528076]], [[5.953611850738525]], [[5.549185276031494]], [[6.731591701507568]], [[6.322219371795654]], [[5.558440685272217]], [[5.827775001525879]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_9d83de5808bf762a051e169ddf5dec99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ea5eeaed84209ff124be614d9744c08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f1c486f10fd20d93cd401dc9279022
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ee70400265197a475301c57636c354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0368db06f948d1fdef49c1f7f5c04880
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()