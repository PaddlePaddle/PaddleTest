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
    class PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34d0d61f0b2de650f32636a14dfed054(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ceede36361657b888ae7b7b03c8b79c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bad79577c379511ab8173ddec292672(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28195f2ce8993321e3e9598d40f45dde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46d8e7d063f5399abfddaaae37d7ccab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eb9a1104993044efa988966d41891de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c53e011a683253e9d9a33ea12f08c97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a43ee025a5099cf85dab38705b35f408(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad4cb6a1bf60979c227887454326a313(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_bbe277936969f858de8759e5c9cb84e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_cb5691e3928bda27edb8d1af13a946e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb5691e3928bda27edb8d1af13a946e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3ab333fe664722fc5edeb88b712b522(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ad4cb6a1bf60979c227887454326a313(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3daa4e732a682590eb7c833e0a8659cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.8156734704971313], [2.0851945877075195], [2.0087790489196777], [2.175837755203247], [2.072892189025879], [2.226891279220581], [2.2403757572174072], [2.028991222381592], [1.9627472162246704], [2.1859636306762695], [2.071002960205078], [2.0156030654907227], [2.1725573539733887], [1.8901900053024292], [2.30122447013855], [2.2802863121032715]], dtype='float32').reshape([16, 1]),
            ]


    class TestPrimitiveOp_7f656f82076c06b4557f15ca1d6b77f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.8358906507492065], [2.0165863037109375], [1.986639380455017], [2.0012130737304688], [2.1091840267181396], [2.1463732719421387], [1.894654393196106], [1.9875061511993408], [2.0823562145233154], [2.1248931884765625], [2.0183653831481934], [2.0908870697021484], [2.0069997310638428], [1.9543657302856445], [1.935641884803772], [2.2911624908447266]], dtype='float32').reshape([16, 1]),
            ]


    class TestPrimitiveOp_499bf21d211d3f9edf46df523ef5e75d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_255ca5e3cc291b17143fe478b9bc1595(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2feb4d76d82e48edff7efe87c2008a3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2feb4d76d82e48edff7efe87c2008a3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf82b8fab2bc2b839a29e0ee42b93cd2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4814be0820b9c6429ce9825a1620514e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf82b8fab2bc2b839a29e0ee42b93cd2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a30372766a0e1d4f8960929ebfb6406(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_923c9dd3b385455e834e61efaeae8ad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dab385e81ef6f65d24b0368f0b8d7b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3900ff0a44f93363b78c4e6334051b5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3900ff0a44f93363b78c4e6334051b5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_625bf231e59f2b771feb49abbae529d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61de64bacc8693dabdc4aca464514d4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e6887a5a75f2427638d8782920cf454e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_69daff242f9b9f07a37d46980cbb7153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6887a5a75f2427638d8782920cf454e
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c53e011a683253e9d9a33ea12f08c97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70ae50d4f188236ca30f7b1b369c52d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fb10bb6fd711ca4ec00a15d6f12328a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fb10bb6fd711ca4ec00a15d6f12328a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42071ce7cc97c54c5685d0198336324a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42071ce7cc97c54c5685d0198336324a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f3ef97f372b21e9c4cdefc06daa6e44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24b52382e92023232a4288ca73a6b1d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f3ef97f372b21e9c4cdefc06daa6e44
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eb9a1104993044efa988966d41891de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c502cc19daae89701588c31f6fc98c8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c502cc19daae89701588c31f6fc98c8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_31d62990aa382baa2de7d58e7a8cb48a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c779cd472d842d7330b3aa7fe7545224(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4ec975991ce209a7d0546894f1ac39d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f3ef97f372b21e9c4cdefc06daa6e44
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7aea8e97ddfb84e136cd4faabd2725b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb86ba91d651a99b17fa7c0f24f63973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb86ba91d651a99b17fa7c0f24f63973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71342863785094055fd2ef24e2f76531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1343eddc8b747ca8105c6cde848fb2ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75531bea9dfbf6d6f0d7762f32636d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61de64bacc8693dabdc4aca464514d4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a855c25cd3fc25ff836cc239996a9587(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09a1a0a4046cc8ec806ce793d1129dba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.0022754669189453], [2.240567207336426], [2.111360788345337], [2.1870291233062744], [1.9671711921691895], [2.0732555389404297], [1.9710893630981445], [2.024165153503418], [2.334258794784546], [2.063455581665039], [1.9864293336868286], [2.0223824977874756], [2.1271204948425293], [2.148427963256836], [1.9668179750442505], [1.997074842453003], [2.1886327266693115], [2.188886880874634], [2.300858497619629], [1.8540812730789185], [1.9147701263427734], [1.975813627243042], [2.1764822006225586], [1.938555359840393]], dtype='float32').reshape([24, 1]),
            ]


    class TestPrimitiveOp_797c54f88d280fe1bd7a9186e4351c78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.219653606414795], [1.9027743339538574], [1.973353624343872], [2.1888537406921387], [2.2433595657348633], [2.0183637142181396], [1.9209812879562378], [1.9374973773956299], [2.027986764907837], [2.10695219039917], [2.2992305755615234], [2.2776315212249756], [2.1760663986206055], [1.852568507194519], [2.2449796199798584], [2.340653657913208], [2.0092720985412598], [2.016832113265991], [2.133084535598755], [2.299717903137207], [2.0900368690490723], [2.204993724822998], [1.8239502906799316], [1.931168556213379]], dtype='float32').reshape([24, 1]),
            ]


    class TestPrimitiveOp_f05c091995578c92f5a18ecbb98b541e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f742dcbaddb2237b98f7372e249de12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53629067ce47e88555e7fd261f39d3c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53629067ce47e88555e7fd261f39d3c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee53aa97a9eddfd458688ded3f804791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2be55af3e451a86739f26c619d33572(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efd7ea0f3dc01a9e2442baf77b6b8dab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.0273215770721436], [2.051156520843506], [1.8777961730957031], [2.1210427284240723]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_751d0b7a9a798dfcfe0e0400a2fe3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.9933178424835205], [2.198335647583008], [2.116931438446045], [2.207944393157959]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a3411d91e56fbc67d4501f8b52dff266(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf82b8fab2bc2b839a29e0ee42b93cd2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d663c3c9c27d090861b36a10493c6710(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_daeec9d8feba1c97d609ed241a2ddb2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d663c3c9c27d090861b36a10493c6710
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a12111fa416b1d9a03dbffbfe4124b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_71481db977097a0ce75bb76b21830911(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_081d32357ddef90a8cde482e109a071f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf82b8fab2bc2b839a29e0ee42b93cd2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbc6f71f5e523dcf2667019d608dcbc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f3ef97f372b21e9c4cdefc06daa6e44
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fed185e4c9b1f17970360713e553c6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6aa0ac0a40b928a836686c73105b6c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6aa0ac0a40b928a836686c73105b6c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c572df188e547a102c0dc437b5ec96c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_7334b1ce81fa7ff132701187b320a8bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65cb5fbc2dea354061397c3ff5299d89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_71342863785094055fd2ef24e2f76531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03b0873225c295ae749e0502b6a2e381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_7b0a3edfdfeda7bafee8fc79754d0d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a48e0b29bc2d7d3a1ab186ef90418a69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_e13f7c024d1e4db8179546256078f8fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e13f7c024d1e4db8179546256078f8fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29edb63ba3d7af903197bc27adc5adbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_205cbf11ed2a908a4b0acb6490558db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_205cbf11ed2a908a4b0acb6490558db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1f81fc3efed01f511431b417df56b2af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1968005bd48f878d964eb7606a611ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f81fc3efed01f511431b417df56b2af
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b05d66d3b1f61ba112f9adeac3fd086d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_edbf3948cc25d34f117fc341f5e8aa46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b05d66d3b1f61ba112f9adeac3fd086d
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0660aa029ce76264caa9a724d9381740(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f35cbe81f4390b17492167e98b35f12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0660aa029ce76264caa9a724d9381740
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b90f53a4d55a91488463361e517402ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b90f53a4d55a91488463361e517402ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34a4e14745cd452b4dd8b78c3684e2f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01943cde4d666e5cf379a436b282b69e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6887a5a75f2427638d8782920cf454e
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebfbb319d8bee43145bee4f0eee6b7ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf82b8fab2bc2b839a29e0ee42b93cd2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6135375915c87e2e8b0429d9c3eaa073(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_933069ca6057a8c094b0bb4effc0cee2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6135375915c87e2e8b0429d9c3eaa073
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef3a65123f62fdbc0bd23bd7dfc1c983(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29fda18709fb6e505551e9d2ed573916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef3a65123f62fdbc0bd23bd7dfc1c983
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b91bc2cf9d01bf59d604182a8eb16fbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e86effb9bf272438e7e996ad138b449d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbf750e048d62509aa5641a364c8091d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81b6d40d62dd3d4b3e010e432f173f57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_015034459abd759872dbcc788337c73e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_015034459abd759872dbcc788337c73e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb4be0313109c96814a2c20d75d20dc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb4be0313109c96814a2c20d75d20dc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9efa8e730502f1e9de3bfa7d86c11130(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9efa8e730502f1e9de3bfa7d86c11130(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fc03b2733f64141b1b920bfd7b66e07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fc03b2733f64141b1b920bfd7b66e07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3a83c23716a6feea67b2300b3ed951b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3a83c23716a6feea67b2300b3ed951b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4971cfc3fc9afdaf2283ecc9dcd2472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4971cfc3fc9afdaf2283ecc9dcd2472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8745c17e7dbfce0bb3a59d65d36f32ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9e5c968963fcd1bf3960452771fee08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9e5c968963fcd1bf3960452771fee08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b4c80ff7761caeb0689a0a80d0eb29
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_970fb2003ef88434d113ce99879683ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dab385e81ef6f65d24b0368f0b8d7b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea9be428f1134a7f676826f3bcab065d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a92e92409bc6a1d5a7727a7857c3cac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07bd82150876b4e8241d7d513e2a2b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f81fc3efed01f511431b417df56b2af
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49c6711789acd4148928e06d67515789(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a40879b80eab78efc9313648b67ec38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3173210845d32aac430011f2b7291e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.2074105739593506], [2.0814871788024902], [2.0127875804901123], [1.9578313827514648], [2.0946848392486572], [2.2787415981292725], [2.061972141265869], [1.993729829788208], [1.838011384010315], [2.260199546813965], [2.0789196491241455], [2.2942206859588623], [2.0726144313812256], [1.9446386098861694], [2.2542762756347656], [2.0304219722747803], [2.349889039993286], [2.2931485176086426], [2.0853161811828613], [2.083437442779541]], dtype='float32').reshape([20, 1]),
            ]


    class TestPrimitiveOp_722450b34f8aa3b33514a457b12be486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26c1a21e28f4604551145552b6b8e5a5
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.9475164413452148], [2.2945051193237305], [2.191253185272217], [2.280691146850586], [2.127012014389038], [2.0838818550109863], [2.1300153732299805], [2.147913932800293], [2.332383632659912], [1.930082082748413], [2.3858895301818848], [2.144876003265381], [2.0804991722106934], [2.0540261268615723], [2.0113155841827393], [2.045353651046753], [1.8870145082473755], [2.307952880859375], [2.000396251678467], [2.1889240741729736]], dtype='float32').reshape([20, 1]),
            ]


    class TestPrimitiveOp_03b0873225c295ae749e0502b6a2e381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_7334b1ce81fa7ff132701187b320a8bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad4cb6a1bf60979c227887454326a313(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_224e8778c4524de62a2006f376c4a99c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b05d66d3b1f61ba112f9adeac3fd086d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67a221d831b4684833093fade0d02e1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0660aa029ce76264caa9a724d9381740
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a48e0b29bc2d7d3a1ab186ef90418a69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_1cc458dd7b6e6199ecc9f89d63fd12a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cc458dd7b6e6199ecc9f89d63fd12a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0efc054aeb9f6b27aa17d0b65cd87bce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f3ef97f372b21e9c4cdefc06daa6e44
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c572df188e547a102c0dc437b5ec96c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d62794fde2b5d4025e237a8db2d0165
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9212ed066c264c0cf4d7281f9c08e500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_769940d1b473a8d8ab0d0a167d5aa878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_094f357c3efa9ef07650ac876615682c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b610ebd7d4e6acb3fb4f1c2702ef86b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b610ebd7d4e6acb3fb4f1c2702ef86b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aee82de719ddd8a9c16f0021f8a8e9df
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_092c143264b90b6d3007f514f9c562fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b9356f7ce02b28d49a0a48e6d512b5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ef5dbdc1efc371a3a258541411773af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6135375915c87e2e8b0429d9c3eaa073
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ed0b25dfeeef8e7d8ea2fc69c6e740c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef3a65123f62fdbc0bd23bd7dfc1c983
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5574e1a5649e3d399e59c19e4f5c73c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_874f5f3a526587ac2ded85a712b03d2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6fd091d777544ff7f52e25c8077d228(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d2d0ec28c0b9a32bda60405668e120e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_495d914a48728298b75d07c891070260(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9b1fba477dd7111eb898a325f1be608(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9fd84cb81cadb410fbe10d15c4157872(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8186a5fdea420f213546205f5eb1ab6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aace9d33f23e3381e5009c0fa84873aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_16f6aab0c90e1499040c8de1282f69cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_21a4cf105b57489c47efc8f0b702b490(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21a4cf105b57489c47efc8f0b702b490(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_190a3c36db39e11bf662b6bc1b478919(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_aace9d33f23e3381e5009c0fa84873aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_e578860d99708be16a2381efab5a3963(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c55d08525657f2bf2b693f78c8716c40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.8156734704971313], [2.0851945877075195], [2.0087790489196777], [2.175837755203247], [2.072892189025879], [2.226891279220581], [2.2403757572174072], [2.028991222381592], [1.9627472162246704], [2.1859636306762695], [2.071002960205078], [2.0156030654907227], [2.1725573539733887], [1.8901900053024292], [2.30122447013855], [2.2802863121032715]], dtype='float32').reshape([16, 1]),
            ]


    class TestPrimitiveOp_95b4a3407eaa090cf5d422aec4befe93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.8358906507492065], [2.0165863037109375], [1.986639380455017], [2.0012130737304688], [2.1091840267181396], [2.1463732719421387], [1.894654393196106], [1.9875061511993408], [2.0823562145233154], [2.1248931884765625], [2.0183653831481934], [2.0908870697021484], [2.0069997310638428], [1.9543657302856445], [1.935641884803772], [2.2911624908447266]], dtype='float32').reshape([16, 1]),
            ]


    class TestPrimitiveOp_d452d6de9721c257dd8fe07be7c7986a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bb1dec91ca36ac3310f6c61056eaeba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_28d911497f992ffd1a2976e48388135a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ef93de17a305bf1f111d9ea430a2767(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ef93de17a305bf1f111d9ea430a2767(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8671212247be8d629effe46aa5cad0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cc0f81eb704050da865262042911b2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8671212247be8d629effe46aa5cad0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d57959f9a56c8c891f5fef2be6f7021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_579478484a8c03d8db15d30660d6a81f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffe7dc83d98b927d088351f0c907b78a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_800ebbcf62680848172611bdd7bcbf46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_800ebbcf62680848172611bdd7bcbf46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f90829eb4d40c98efa7fec173059a69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40947cb94e296fd46b4c960b54689a9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51534adc84ecc23b33171812ee69bf66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f81fc3efed01f511431b417df56b2af
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fd84cb81cadb410fbe10d15c4157872(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_405c7b8b3ed23ea9d1379dd412e155e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a31a1cffacba8679cd423c52f424aed9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a31a1cffacba8679cd423c52f424aed9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21ac1b005b0df40a36c111cffcbb9cb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21ac1b005b0df40a36c111cffcbb9cb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b2dfae410c2e169a09040f2b0b29ab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9b1fba477dd7111eb898a325f1be608(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d0a412a59f1620f387984c4840e3e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9d0a412a59f1620f387984c4840e3e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ba95568b6b9d8d74603eba43159b16c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cea6e39731ced0d59936e2e6d651e756(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74e44cb74f53dae343b8df9d9e08c752(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbb53077b23e5c5ffe9c26c76aec173f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9784ad6c47555b4aaa2d118149fd284e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9784ad6c47555b4aaa2d118149fd284e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89cd79684dbbf8d6692869369bf5a271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a708792bd0bfe16c2f8d2e80502e808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3172c18fa64a5b034c19b59f06ccf76e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40947cb94e296fd46b4c960b54689a9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2894c51e5dab2000c60638b6131ce514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17194e69aa6bbf0051e62957fc69ca32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.0022754669189453], [2.240567207336426], [2.111360788345337], [2.1870291233062744], [1.9671711921691895], [2.0732555389404297], [1.9710893630981445], [2.024165153503418], [2.334258794784546], [2.063455581665039], [1.9864293336868286], [2.0223824977874756], [2.1271204948425293], [2.148427963256836], [1.9668179750442505], [1.997074842453003], [2.1886327266693115], [2.188886880874634], [2.300858497619629], [1.8540812730789185], [1.9147701263427734], [1.975813627243042], [2.1764822006225586], [1.938555359840393]], dtype='float32').reshape([24, 1]),
            ]


    class TestPrimitiveOp_bfb82b31d04d2cceca912da7d7d60df7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.219653606414795], [1.9027743339538574], [1.973353624343872], [2.1888537406921387], [2.2433595657348633], [2.0183637142181396], [1.9209812879562378], [1.9374973773956299], [2.027986764907837], [2.10695219039917], [2.2992305755615234], [2.2776315212249756], [2.1760663986206055], [1.852568507194519], [2.2449796199798584], [2.340653657913208], [2.0092720985412598], [2.016832113265991], [2.133084535598755], [2.299717903137207], [2.0900368690490723], [2.204993724822998], [1.8239502906799316], [1.931168556213379]], dtype='float32').reshape([24, 1]),
            ]


    class TestPrimitiveOp_3790d01f2f8c2439e06c846207581e96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ef3b89e910f14f8cfc659e3c0264497(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8d3a07bdb42dd8ee3547eae93c1c712(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8d3a07bdb42dd8ee3547eae93c1c712(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_460b577b9873010c53e3648072ed983d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b352ae4014713bd1e3fcd27d88ad7a46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26ed74b7fb33605e33dca9940fbbba41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.0273215770721436], [2.051156520843506], [1.8777961730957031], [2.1210427284240723]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d2621fbd503f4d064a919e8beea91886(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.9933178424835205], [2.198335647583008], [2.116931438446045], [2.207944393157959]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d63d0502835c97ce05066f47a59de7b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8671212247be8d629effe46aa5cad0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e570f93f82f51d0f37b62a9901358a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8671212247be8d629effe46aa5cad0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1b24e64e7d061a24b326453d32b895e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_e635197811d58c100c4c67b18942513d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bd856618679bd4782fd204381d1cfa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8671212247be8d629effe46aa5cad0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26647d210bd36912204e58e95f875427(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05f55fac9e4f3f834f25a0494527e0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ebcabef48bdfec6bc370f694b8fc59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ebcabef48bdfec6bc370f694b8fc59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ca760ac14015e722b8c6e3115074ac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4137946717a0d549650becdc0368714f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdd5c77a1bccac97fb9bcd941be55b0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_89cd79684dbbf8d6692869369bf5a271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d5805bbb182cb2d31b437d9b7029522(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_11b835dbdeff832a672b95134cc36d4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec344dab8a3adb5edb327d9676fb8802(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_7b4351c884bf97a751520b83b49c471a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b4351c884bf97a751520b83b49c471a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_207e09a2c298c303325e10d3aa15abda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8e7220e97ae035e2b237e229452a1fce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e7220e97ae035e2b237e229452a1fce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1968005bd48f878d964eb7606a611ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f81fc3efed01f511431b417df56b2af
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6f5382b92120fde8e753ff810bc1910(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bafaf47b48e62cdf13e1bf987fb1aed5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce0e3e749e40fb9ca2d2933a802f9e77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bafaf47b48e62cdf13e1bf987fb1aed5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_fce7bd83b38e797086b31748cfdfc0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fce7bd83b38e797086b31748cfdfc0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_993e04e71fc0afc29fffd323dd66b7bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bca79e184cd009e4f6f17e8b24f31171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f81fc3efed01f511431b417df56b2af
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6f5863320e0b28d757d719f44784314(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8671212247be8d629effe46aa5cad0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e3af6f5ec05b11660844ebe9a0134770(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_957151710fea28bae632e3b095b6d36b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3af6f5ec05b11660844ebe9a0134770
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dc2270d63c9f1934d703e56317a9165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24d6b2c877339bdd0b87c2599fdca484(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ec9b1d1d50bf5421efd6c8da02550d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0a1de43c059fd5ef5fc85de6794d00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b30ddd43403d6fca1d269c07b648478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14a6a60ee4743cbd4e68f1958bc02842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14a6a60ee4743cbd4e68f1958bc02842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c80296a61808577a69c360e7716fa09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c80296a61808577a69c360e7716fa09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d06f209b4d255672b2ca08d72928cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d06f209b4d255672b2ca08d72928cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60c09e6834477532ad92de97c8a0c7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60c09e6834477532ad92de97c8a0c7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dca967215480e8e8a7ae145749c5c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dca967215480e8e8a7ae145749c5c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587aa7312e395a179b53e68f6ba529b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587aa7312e395a179b53e68f6ba529b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0950103cc239c30cc8bd48cfd542f8ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b282902c491e8cb11e87a4b3dff47358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b282902c491e8cb11e87a4b3dff47358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d911497f992ffd1a2976e48388135a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6eab1dfbb55712fab34eaf5fc07613c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffe7dc83d98b927d088351f0c907b78a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbfa2a3ae357c9676008ad25ac4757f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a77f7e84e0bfeb55a436eece61054538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07bd82150876b4e8241d7d513e2a2b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f81fc3efed01f511431b417df56b2af
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab9f30d64115bf2e78d0847c9624ef89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5c2a767c26d1d328e47c1af7180f540(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b136e177ba2e57b67fc2acf51c5cc87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.2074105739593506], [2.0814871788024902], [2.0127875804901123], [1.9578313827514648], [2.0946848392486572], [2.2787415981292725], [2.061972141265869], [1.993729829788208], [1.838011384010315], [2.260199546813965], [2.0789196491241455], [2.2942206859588623], [2.0726144313812256], [1.9446386098861694], [2.2542762756347656], [2.0304219722747803], [2.349889039993286], [2.2931485176086426], [2.0853161811828613], [2.083437442779541]], dtype='float32').reshape([20, 1]),
            ]


    class TestPrimitiveOp_58cecb177c8e73bb56890e7d1b62cdb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e578860d99708be16a2381efab5a3963
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.9475164413452148], [2.2945051193237305], [2.191253185272217], [2.280691146850586], [2.127012014389038], [2.0838818550109863], [2.1300153732299805], [2.147913932800293], [2.332383632659912], [1.930082082748413], [2.3858895301818848], [2.144876003265381], [2.0804991722106934], [2.0540261268615723], [2.0113155841827393], [2.045353651046753], [1.8870145082473755], [2.307952880859375], [2.000396251678467], [2.1889240741729736]], dtype='float32').reshape([20, 1]),
            ]


    class TestPrimitiveOp_2d5805bbb182cb2d31b437d9b7029522(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4137946717a0d549650becdc0368714f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aace9d33f23e3381e5009c0fa84873aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_08e1bca00477229b6287ed0d6816f3e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccb0917d8abbbfc32f9f3bb2d6f859f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bafaf47b48e62cdf13e1bf987fb1aed5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ec344dab8a3adb5edb327d9676fb8802(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5d66dc7816ca5af1c5d525867416ebbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d66dc7816ca5af1c5d525867416ebbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b564adcf634a601c6479de15243c92e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ca760ac14015e722b8c6e3115074ac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4581f7e6434d4539bfb21f1d39234f86
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f354b0bdba59a67cc2ff41af02c40bc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35ab90e99d08a4aff0cdb884689a8ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c65875fde78c9907aabbd18cc197a83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d40cb4ab4c021b14ab7fee2a6a72c37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d40cb4ab4c021b14ab7fee2a6a72c37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed015de1cd4c3df3ac7fb85af1eb4c8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15dabb18b0ea2ffce55edd26fbd8e41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a22b0b42350f58480d396eb1ba23102(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3af6f5ec05b11660844ebe9a0134770
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da390d3f9f6226c327e13d2f9b1837b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6189a1af576559c3c4ddbdf8fcbbc686
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()