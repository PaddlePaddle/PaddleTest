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
    class PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_323982a569a3e42f50e3d9833d0688d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb2824e9670788dbdcd7b58c0e82defb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05ca305f4484e2397c103531cf7a78a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad2d6251191afe013914101afdd938a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c90dface6356a1874c25530d3953d4be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1c8b3cb8c922a776e2cc944b7c1b2f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d98d10224c4c60321937b12dad7bdca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05614259093999863, -0.2648021876811981, 0.14120104908943176, 0.11532451212406158], [-0.0013617724180221558, 0.029156655073165894, 0.20148354768753052, 0.27942192554473877], [0.10520064830780029, 0.225392147898674, 0.20918087661266327, 0.17586283385753632], [0.3509211242198944, -0.3742552399635315, 0.25012320280075073, -0.011851891875267029], [-0.033196449279785156, -0.07336187362670898, 0.22557489573955536, 0.398066908121109]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_0e304e5f1b498745126b7fafd58df85c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.030529730021953583, -0.2831009328365326, 0.36617031693458557, -0.04184707999229431], [-0.07069867849349976, -0.2227173000574112, 0.041081346571445465, -0.03687387704849243], [-0.38079363107681274, 0.0006191432476043701, 0.3535638451576233, 0.09530767798423767], [-0.07069867849349976, -0.2227173000574112, 0.041081346571445465, -0.03687387704849243], [-0.38079363107681274, 0.0006191432476043701, 0.3535638451576233, 0.09530767798423767]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_6de4af97a33f244f0a5df9952252f7d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21537d4bb40c00689c77a44389ac5091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6de4af97a33f244f0a5df9952252f7d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5182f13ee9b0c639bb76ee0f0f4e05c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_142eb2b99489f7f4dce93d0ce79ad2ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.027252674102783203, -0.39509862661361694, -0.02363678812980652, 0.1274213194847107], [0.20206919312477112, -0.032816849648952484, -0.32369446754455566, -0.09777773916721344], [0.2741281986236572, 0.16037379205226898, 0.44714775681495667, 0.0011175870895385742], [0.20206919312477112, -0.032816849648952484, -0.32369446754455566, -0.09777773916721344], [0.2741281986236572, 0.16037379205226898, 0.44714775681495667, 0.0011175870895385742], [-0.3455130159854889, 0.1529913991689682, -0.24695870280265808, -0.2722965180873871], [-0.3455130159854889, 0.1529913991689682, -0.24695870280265808, -0.2722965180873871]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_4c235f1a8fce3b78197645cdf9eee8e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97e63d0670ddbe187ad7252fc3851153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a862f5c26cf231d5b01017ba619f5a79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_446f16e00bcd2a8f681f0110e198cca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a875ce5a7ae4d01c00fd7006d5a73df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2679559886455536, 0.09963895380496979, 0.050239741802215576, 0.09699589759111404], [-0.4907260537147522, -0.0988239049911499, 0.043211743235588074, 0.13195684552192688], [0.22280314564704895, -0.2555414140224457, 0.20873220264911652, 0.1573859304189682], [0.0817495584487915, 0.20968347787857056, 0.3012875020503998, -0.022008880972862244], [0.0817495584487915, 0.20968347787857056, 0.3012875020503998, -0.022008880972862244], [0.22280314564704895, -0.2555414140224457, 0.20873220264911652, 0.1573859304189682]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_63dde857c551bc58454f971ce8b32e45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42347490787506104, 0.44859740138053894, -0.025512784719467163, -0.1642150580883026], [-0.38556623458862305, -0.3216448426246643, 0.11168976873159409, 0.03786981850862503], [0.0014653801918029785, 0.13820701837539673, 0.07339861989021301, -0.0377705842256546], [0.3172072768211365, 0.15113165974617004, -0.22106342017650604, -0.08921520411968231], [0.42347490787506104, 0.44859740138053894, -0.025512784719467163, -0.1642150580883026]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_fcae0dc2a2c722cb837fd38f192fe476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9cd7b0ad5c95de58b0630b1f5acb655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.03413599729537964, -0.28878089785575867, 0.3469504117965698, 0.19062592089176178], [0.2298850268125534, -0.05640196055173874, -0.03873452544212341, 0.22728106379508972], [0.02583347260951996, 0.3125740587711334, -0.10351571440696716, 0.242221400141716], [-0.33769890666007996, -0.06839853525161743, 0.11585550010204315, -0.06547404825687408]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_8e8c9c94ea323521a1f7fc50a4c07dfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6b859552560203c0bb57dd13d32929e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5253b7f04a076ee70be0408169394e7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.140933558344841, -0.00036913156509399414, -0.33528417348861694, 0.08211620897054672], [0.140933558344841, -0.00036913156509399414, -0.33528417348861694, 0.08211620897054672], [-0.1097898781299591, 0.3261854350566864, 0.044446952641010284, 0.019107185304164886], [0.20964205265045166, -0.009211540222167969, -0.2750697135925293, -0.14300581812858582], [-0.06909973919391632, 0.23634669184684753, -0.23610402643680573, -0.2775801420211792], [-0.30173084139823914, -0.3142993748188019, -0.1273236721754074, -0.06189805269241333], [-0.09281511604785919, -0.13114047050476074, 0.019350245594978333, 0.03895244002342224]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_58192e70de51850ae15bd77d8e63ee35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6d3eb61cc68ac0dbf2e54bd121d4363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6de4af97a33f244f0a5df9952252f7d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff81e1a645afe9f47dbc759d305e8efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d27362a507bed5a8e4c0efa47a5310c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c5986b4367c2602968dc7ec718f5674(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07022564113140106, 0.24063928425312042, 0.2388477772474289, -0.0949905663728714], [0.03909817337989807, -0.15637561678886414, 0.20350569486618042, -0.2634051442146301], [0.03909817337989807, -0.15637561678886414, 0.20350569486618042, -0.2634051442146301], [-0.06787297129631042, 0.008341282606124878, 0.10706061124801636, -0.25277483463287354], [0.29549020528793335, -0.19191592931747437, -0.07872748374938965, 0.09537695348262787], [-0.10860984027385712, -0.09693586826324463, -0.16498376429080963, 0.18836523592472076]], dtype='float32').reshape([6, 4]),
            ]


    
    class PrimitiveOp_736d709c25c862e9aafc5f51dc0081cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_caffb2bce991a28985236646f20a9e5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_736d709c25c862e9aafc5f51dc0081cc
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5f8faca5ed12ca77936e999c4622ac5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d0091336b47594829b775b382cb14e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f8faca5ed12ca77936e999c4622ac5c
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cefa88e59fd8c2dc5c00cb53c92a966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6de4af97a33f244f0a5df9952252f7d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56de58c909793e4e3e75537e1812383e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7410fabf70e9ae1f4706ef3dfc53341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78a947ee13beb73f4080579277bf1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bcdcbcb43fa7e21ec1e5219f169fee6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6de4af97a33f244f0a5df9952252f7d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d47325b79fffba42a28e9d97997422c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af313d08cc6fdcc1637b071dc8949dc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_132d086056d9bf13ba564e07162bdf7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_419d8c4eae350cf0c147628fd02bbb6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6de4af97a33f244f0a5df9952252f7d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52e1b1bdb568cb1dee355bab72d23e28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06421732902526855, -0.04781962186098099, 0.16483047604560852, -0.16644680500030518], [0.10803155601024628, 0.03807184100151062, 0.24309112131595612, -0.014126971364021301], [-0.1445263922214508, 0.45514774322509766, 0.1134442538022995, -0.06528478860855103], [-0.1445263922214508, 0.45514774322509766, 0.1134442538022995, -0.06528478860855103], [-0.34093427658081055, -0.1452919840812683, -0.3238971531391144, 0.25240251421928406]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_bfbee49ba5dbc5ee76617ec1c93f217c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5697f76fe3d3ae81f011b821cf3619ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c90dface6356a1874c25530d3953d4be
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2abcba58ac17e61530721b82429be0c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1356925368309021, -0.3836618661880493, -0.2497148960828781, -0.10594353079795837], [-0.13692587614059448, -0.18728870153427124, -0.11343744397163391, -0.3271639943122864], [-0.17433902621269226, -0.08971483260393143, -0.13315066695213318, 0.015453487634658813], [0.1356925368309021, -0.3836618661880493, -0.2497148960828781, -0.10594353079795837], [-0.11636693775653839, -0.08864608407020569, 0.22304943203926086, -0.282735139131546], [0.14897876977920532, -0.186306431889534, 0.06858530640602112, 0.24515454471111298], [-0.11636693775653839, -0.08864608407020569, 0.22304943203926086, -0.282735139131546]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_96f575ab4ee4d0900e64fc015ccd7dc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_323982a569a3e42f50e3d9833d0688d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb2824e9670788dbdcd7b58c0e82defb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05ca305f4484e2397c103531cf7a78a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad2d6251191afe013914101afdd938a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51fdcb1f45aab69103eac7cf84ee4e8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d98d10224c4c60321937b12dad7bdca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05614259093999863, -0.2648021876811981, 0.14120104908943176, 0.11532451212406158], [-0.0013617724180221558, 0.029156655073165894, 0.20148354768753052, 0.27942192554473877], [0.10520064830780029, 0.225392147898674, 0.20918087661266327, 0.17586283385753632], [0.3509211242198944, -0.3742552399635315, 0.25012320280075073, -0.011851891875267029], [-0.033196449279785156, -0.07336187362670898, 0.22557489573955536, 0.398066908121109]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_0e304e5f1b498745126b7fafd58df85c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.030529730021953583, -0.2831009328365326, 0.36617031693458557, -0.04184707999229431], [-0.07069867849349976, -0.2227173000574112, 0.041081346571445465, -0.03687387704849243], [-0.38079363107681274, 0.0006191432476043701, 0.3535638451576233, 0.09530767798423767], [-0.07069867849349976, -0.2227173000574112, 0.041081346571445465, -0.03687387704849243], [-0.38079363107681274, 0.0006191432476043701, 0.3535638451576233, 0.09530767798423767]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_b2b3d24c6c8d171d6fdb2f38ddb3a290(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_594472acf6bfaca02ecc9f5c6e27a31a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2b3d24c6c8d171d6fdb2f38ddb3a290
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d9fd80dab9bec9da77010680687bb05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_142eb2b99489f7f4dce93d0ce79ad2ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.027252674102783203, -0.39509862661361694, -0.02363678812980652, 0.1274213194847107], [0.20206919312477112, -0.032816849648952484, -0.32369446754455566, -0.09777773916721344], [0.2741281986236572, 0.16037379205226898, 0.44714775681495667, 0.0011175870895385742], [0.20206919312477112, -0.032816849648952484, -0.32369446754455566, -0.09777773916721344], [0.2741281986236572, 0.16037379205226898, 0.44714775681495667, 0.0011175870895385742], [-0.3455130159854889, 0.1529913991689682, -0.24695870280265808, -0.2722965180873871], [-0.3455130159854889, 0.1529913991689682, -0.24695870280265808, -0.2722965180873871]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_4c235f1a8fce3b78197645cdf9eee8e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97e63d0670ddbe187ad7252fc3851153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b18f860a95171df63edad843fa25121f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eec0447e82ab83c0d594f835a02e8a15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a875ce5a7ae4d01c00fd7006d5a73df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2679559886455536, 0.09963895380496979, 0.050239741802215576, 0.09699589759111404], [-0.4907260537147522, -0.0988239049911499, 0.043211743235588074, 0.13195684552192688], [0.22280314564704895, -0.2555414140224457, 0.20873220264911652, 0.1573859304189682], [0.0817495584487915, 0.20968347787857056, 0.3012875020503998, -0.022008880972862244], [0.0817495584487915, 0.20968347787857056, 0.3012875020503998, -0.022008880972862244], [0.22280314564704895, -0.2555414140224457, 0.20873220264911652, 0.1573859304189682]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_63dde857c551bc58454f971ce8b32e45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42347490787506104, 0.44859740138053894, -0.025512784719467163, -0.1642150580883026], [-0.38556623458862305, -0.3216448426246643, 0.11168976873159409, 0.03786981850862503], [0.0014653801918029785, 0.13820701837539673, 0.07339861989021301, -0.0377705842256546], [0.3172072768211365, 0.15113165974617004, -0.22106342017650604, -0.08921520411968231], [0.42347490787506104, 0.44859740138053894, -0.025512784719467163, -0.1642150580883026]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_fcae0dc2a2c722cb837fd38f192fe476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9cd7b0ad5c95de58b0630b1f5acb655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.03413599729537964, -0.28878089785575867, 0.3469504117965698, 0.19062592089176178], [0.2298850268125534, -0.05640196055173874, -0.03873452544212341, 0.22728106379508972], [0.02583347260951996, 0.3125740587711334, -0.10351571440696716, 0.242221400141716], [-0.33769890666007996, -0.06839853525161743, 0.11585550010204315, -0.06547404825687408]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_8e8c9c94ea323521a1f7fc50a4c07dfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2fb4a34b6b8d9f2ba3e11bd407a96a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5253b7f04a076ee70be0408169394e7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.140933558344841, -0.00036913156509399414, -0.33528417348861694, 0.08211620897054672], [0.140933558344841, -0.00036913156509399414, -0.33528417348861694, 0.08211620897054672], [-0.1097898781299591, 0.3261854350566864, 0.044446952641010284, 0.019107185304164886], [0.20964205265045166, -0.009211540222167969, -0.2750697135925293, -0.14300581812858582], [-0.06909973919391632, 0.23634669184684753, -0.23610402643680573, -0.2775801420211792], [-0.30173084139823914, -0.3142993748188019, -0.1273236721754074, -0.06189805269241333], [-0.09281511604785919, -0.13114047050476074, 0.019350245594978333, 0.03895244002342224]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_58192e70de51850ae15bd77d8e63ee35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63343e6774a43f2f65608a94804c5fb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2b3d24c6c8d171d6fdb2f38ddb3a290
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10ea1c10ee0f56ec92c29ace19a24710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f997a0b0ce6adf6e53f318211a00471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c5986b4367c2602968dc7ec718f5674(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07022564113140106, 0.24063928425312042, 0.2388477772474289, -0.0949905663728714], [0.03909817337989807, -0.15637561678886414, 0.20350569486618042, -0.2634051442146301], [0.03909817337989807, -0.15637561678886414, 0.20350569486618042, -0.2634051442146301], [-0.06787297129631042, 0.008341282606124878, 0.10706061124801636, -0.25277483463287354], [0.29549020528793335, -0.19191592931747437, -0.07872748374938965, 0.09537695348262787], [-0.10860984027385712, -0.09693586826324463, -0.16498376429080963, 0.18836523592472076]], dtype='float32').reshape([6, 4]),
            ]


    
    class PrimitiveOp_66f1bcc1df264ab7fa037bda4cfe3632(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5dd1040eb2474fd28e8585ddd238852(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66f1bcc1df264ab7fa037bda4cfe3632
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff41e6b8c9fdb98d69cafa3e9288912d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66f1bcc1df264ab7fa037bda4cfe3632
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0055529b72cffe4bb75257cabe93126c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2b3d24c6c8d171d6fdb2f38ddb3a290
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07bc05051fd7d2010a3d439e92ebc3f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ba38106e418f488ef4654fb160730a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_085ed916f8b4bf8127b6ebf3c5b294c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03e9d577f7660b9634ac9e0303dc0911(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2b3d24c6c8d171d6fdb2f38ddb3a290
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d47325b79fffba42a28e9d97997422c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af313d08cc6fdcc1637b071dc8949dc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c1f9eda0b4d1e1da82682a25a26b053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7905a1661fcba8bb1482190ab3a38963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2b3d24c6c8d171d6fdb2f38ddb3a290
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52e1b1bdb568cb1dee355bab72d23e28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06421732902526855, -0.04781962186098099, 0.16483047604560852, -0.16644680500030518], [0.10803155601024628, 0.03807184100151062, 0.24309112131595612, -0.014126971364021301], [-0.1445263922214508, 0.45514774322509766, 0.1134442538022995, -0.06528478860855103], [-0.1445263922214508, 0.45514774322509766, 0.1134442538022995, -0.06528478860855103], [-0.34093427658081055, -0.1452919840812683, -0.3238971531391144, 0.25240251421928406]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_bfbee49ba5dbc5ee76617ec1c93f217c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a598cf72e489725419cfbfd582501bcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2abcba58ac17e61530721b82429be0c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1356925368309021, -0.3836618661880493, -0.2497148960828781, -0.10594353079795837], [-0.13692587614059448, -0.18728870153427124, -0.11343744397163391, -0.3271639943122864], [-0.17433902621269226, -0.08971483260393143, -0.13315066695213318, 0.015453487634658813], [0.1356925368309021, -0.3836618661880493, -0.2497148960828781, -0.10594353079795837], [-0.11636693775653839, -0.08864608407020569, 0.22304943203926086, -0.282735139131546], [0.14897876977920532, -0.186306431889534, 0.06858530640602112, 0.24515454471111298], [-0.11636693775653839, -0.08864608407020569, 0.22304943203926086, -0.282735139131546]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_96f575ab4ee4d0900e64fc015ccd7dc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ca5a0028fa1855630f072e2b570b6f0
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()