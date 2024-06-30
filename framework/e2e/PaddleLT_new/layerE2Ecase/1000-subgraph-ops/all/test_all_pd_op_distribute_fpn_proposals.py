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
    class PrimitiveOp_8bd00c3dc1b45549e0a1388f27c5d206(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.distribute_fpn_proposals(input_0, input_1, 2, 5, 4, 224, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f95523c3c3ebefc03b2cdc656d7e4ddf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bd00c3dc1b45549e0a1388f27c5d206
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.distribute_fpn_proposals(input_0, input_1, 2, 5, 4, 224, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccd6e3d14cf6c31394b679691fc02a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_dbc8767851248bdce86cb342ce644532(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.distribute_fpn_proposals(input_0, input_1, 2, 5, 4, 224, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[512, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_50b50d7eb6f72847e5356ec219421f47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbc8767851248bdce86cb342ce644532
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3caca39ba4e365c96d8cdd62f8507fc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bd00c3dc1b45549e0a1388f27c5d206
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bc65613faea7d7fc9c4c17abf77780a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_50b50d7eb6f72847e5356ec219421f47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbc8767851248bdce86cb342ce644532
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ccd6e3d14cf6c31394b679691fc02a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5cbf812fedcf5b03070f11cb0011ebaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f3dfaeb160e4285072f724f8dc91eb97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.12931282818317413, 0.48861533403396606, 0.17718838155269623, 0.12520994246006012], [0.07523564994335175, 0.4386076331138611, 0.3620389699935913, 0.07658398151397705], [0.14756430685520172, 0.07572465389966965, 0.26838070154190063, 0.37414035201072693], [0.3915621042251587, 0.19104118645191193, 0.01597001403570175, 0.05103715509176254], [0.22588086128234863, 0.0050764307379722595, 0.31528225541114807, 0.298330157995224], [0.177719846367836, 0.25171852111816406, 0.10758128017187119, 0.3024943470954895]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d4773bb02c5e71d41e18e6f6529cb128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_50b50d7eb6f72847e5356ec219421f47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbc8767851248bdce86cb342ce644532
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6ce606e6f82e87b477a7375cc2828dec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03567306697368622, 0.41631054878234863, 0.14406552910804749, 0.48773443698883057], [0.4846419095993042, 0.046012621372938156, 0.3439387083053589, 0.3439829349517822]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_26b6b0120e03eb60738b194b8730eded(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([390, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_846fc6bce5b69d22899c546d0399b7bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4272349774837494, 0.47877711057662964, 0.08770851790904999, 0.08134938776493073], [0.3754722774028778, 0.3275013566017151, 0.46637246012687683, 0.4262460768222809]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b6142b556da29c646241ee50797d7950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5cbf812fedcf5b03070f11cb0011ebaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5cbf812fedcf5b03070f11cb0011ebaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ef06612de533eab54c1f9329f8d70498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06026938185095787, 0.10351205617189407, 0.010360955260694027, 0.05767056718468666], [0.04079157114028931, 0.11247462779283524, 0.43626201152801514, 0.36621132493019104]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f95523c3c3ebefc03b2cdc656d7e4ddf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bd00c3dc1b45549e0a1388f27c5d206
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b6142b556da29c646241ee50797d7950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f228a2c18a6c483b34a21204d5ed0381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae259df098b7c8dc9a8b96e8989da75e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4692842364311218, 0.07899530231952667, 0.43798011541366577, 0.0650915876030922], [0.3472828269004822, 0.06536136567592621, 0.14612704515457153, 0.24545007944107056]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d00e457df682baf936e8397e9e3725ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08356991410255432, 0.26200056076049805, 0.2756381928920746, 0.4261353611946106], [0.3760230839252472, 0.15816281735897064, 0.4333730638027191, 0.17451728880405426], [0.07281681895256042, 0.1318535953760147, 0.18752706050872803, 0.27600735425949097], [0.110041543841362, 0.07978207617998123, 0.08638979494571686, 0.38184553384780884]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3caca39ba4e365c96d8cdd62f8507fc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bd00c3dc1b45549e0a1388f27c5d206
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d8ccd828b1f357f03ca0312d2c12ed12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2400294840335846, 0.33962640166282654, 0.2767137289047241, 0.09159930795431137], [0.2195415049791336, 0.06079661473631859, 0.4498836100101471, 0.3681503236293793]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5cbf812fedcf5b03070f11cb0011ebaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a6b3e390779f4855ac7c490f6f1e919e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29233288764953613, 0.36788859963417053, 0.006307649426162243, 0.2807435095310211], [0.3606562316417694, 0.492276668548584, 0.20333808660507202, 0.06060683727264404], [0.14438258111476898, 0.47764408588409424, 0.38392844796180725, 0.3385300934314728]], dtype='float32').reshape([3, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_13ced7c85ce52781b377f7c0ff46460b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ccd6e3d14cf6c31394b679691fc02a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_4cc0e33c14b5801bf5a2e1be210c649a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.distribute_fpn_proposals(input_0, input_1, 2, 5, 4, 224, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0283d11f96901e4dbfcb7b63cf44f5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cc0e33c14b5801bf5a2e1be210c649a
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cc4de6814dac623db0672920fb1c564d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bc65613faea7d7fc9c4c17abf77780a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f0283d11f96901e4dbfcb7b63cf44f5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cc0e33c14b5801bf5a2e1be210c649a
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ccd6e3d14cf6c31394b679691fc02a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5cbf812fedcf5b03070f11cb0011ebaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f3dfaeb160e4285072f724f8dc91eb97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.12931282818317413, 0.48861533403396606, 0.17718838155269623, 0.12520994246006012], [0.07523564994335175, 0.4386076331138611, 0.3620389699935913, 0.07658398151397705], [0.14756430685520172, 0.07572465389966965, 0.26838070154190063, 0.37414035201072693], [0.3915621042251587, 0.19104118645191193, 0.01597001403570175, 0.05103715509176254], [0.22588086128234863, 0.0050764307379722595, 0.31528225541114807, 0.298330157995224], [0.177719846367836, 0.25171852111816406, 0.10758128017187119, 0.3024943470954895]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d4773bb02c5e71d41e18e6f6529cb128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f0283d11f96901e4dbfcb7b63cf44f5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cc0e33c14b5801bf5a2e1be210c649a
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6ce606e6f82e87b477a7375cc2828dec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03567306697368622, 0.41631054878234863, 0.14406552910804749, 0.48773443698883057], [0.4846419095993042, 0.046012621372938156, 0.3439387083053589, 0.3439829349517822]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_26b6b0120e03eb60738b194b8730eded(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([390, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_846fc6bce5b69d22899c546d0399b7bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4272349774837494, 0.47877711057662964, 0.08770851790904999, 0.08134938776493073], [0.3754722774028778, 0.3275013566017151, 0.46637246012687683, 0.4262460768222809]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b6142b556da29c646241ee50797d7950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5cbf812fedcf5b03070f11cb0011ebaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5cbf812fedcf5b03070f11cb0011ebaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ef06612de533eab54c1f9329f8d70498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06026938185095787, 0.10351205617189407, 0.010360955260694027, 0.05767056718468666], [0.04079157114028931, 0.11247462779283524, 0.43626201152801514, 0.36621132493019104]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_13ced7c85ce52781b377f7c0ff46460b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b6142b556da29c646241ee50797d7950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f228a2c18a6c483b34a21204d5ed0381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae259df098b7c8dc9a8b96e8989da75e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4692842364311218, 0.07899530231952667, 0.43798011541366577, 0.0650915876030922], [0.3472828269004822, 0.06536136567592621, 0.14612704515457153, 0.24545007944107056]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d00e457df682baf936e8397e9e3725ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08356991410255432, 0.26200056076049805, 0.2756381928920746, 0.4261353611946106], [0.3760230839252472, 0.15816281735897064, 0.4333730638027191, 0.17451728880405426], [0.07281681895256042, 0.1318535953760147, 0.18752706050872803, 0.27600735425949097], [0.110041543841362, 0.07978207617998123, 0.08638979494571686, 0.38184553384780884]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cc4de6814dac623db0672920fb1c564d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d8ccd828b1f357f03ca0312d2c12ed12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2400294840335846, 0.33962640166282654, 0.2767137289047241, 0.09159930795431137], [0.2195415049791336, 0.06079661473631859, 0.4498836100101471, 0.3681503236293793]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5cbf812fedcf5b03070f11cb0011ebaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a6b3e390779f4855ac7c490f6f1e919e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d240ee4aa354c63b1a8987567e8dc66
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29233288764953613, 0.36788859963417053, 0.006307649426162243, 0.2807435095310211], [0.3606562316417694, 0.492276668548584, 0.20333808660507202, 0.06060683727264404], [0.14438258111476898, 0.47764408588409424, 0.38392844796180725, 0.3385300934314728]], dtype='float32').reshape([3, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()