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
        return True, f"last stage failed."
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

def NumOperationsInBlock(block_idx):
    return [1389][block_idx] - 1 # number-of-ops-in-block

def GetPaddleDebugNumAllowedOps():
    try:
        return int(os.getenv('PADDLE_DEBUG_NUM_ALLOWED_OPS'))
    except:
        return None

paddle_debug_num_allowed_ops = GetPaddleDebugNumAllowedOps()


if type(paddle_debug_num_allowed_ops) is not int:
    def EarlyReturn(block_idx, op_idx):
        return False      
else:
    def EarlyReturn(block_idx, op_idx):
        return op_idx >= paddle_debug_num_allowed_ops

class BlockEntries:
    def builtin_module_4575_0_0(self, parameter_693, parameter_677, parameter_661, parameter_645, parameter_629, parameter_613, constant_17, constant_16, parameter_597, constant_15, constant_14, parameter_593, parameter_577, parameter_561, parameter_545, parameter_529, parameter_513, parameter_497, parameter_481, parameter_465, parameter_449, parameter_433, parameter_417, parameter_401, parameter_385, parameter_369, parameter_353, parameter_337, parameter_321, parameter_305, parameter_289, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, parameter_273, constant_7, constant_6, parameter_269, parameter_267, parameter_265, parameter_259, parameter_257, parameter_255, parameter_249, parameter_247, parameter_245, parameter_239, parameter_237, parameter_235, parameter_229, parameter_227, parameter_225, parameter_219, parameter_217, parameter_215, parameter_209, parameter_207, parameter_205, parameter_199, parameter_197, parameter_195, parameter_189, parameter_187, parameter_185, parameter_179, parameter_177, parameter_175, parameter_169, parameter_167, parameter_165, parameter_159, parameter_157, parameter_155, parameter_149, parameter_147, parameter_145, parameter_139, parameter_137, parameter_135, parameter_129, parameter_127, parameter_125, parameter_119, parameter_117, parameter_115, parameter_109, constant_5, constant_4, parameter_105, parameter_103, parameter_101, parameter_95, parameter_93, parameter_91, parameter_85, parameter_83, parameter_81, parameter_75, parameter_73, parameter_71, parameter_65, parameter_63, parameter_61, parameter_55, parameter_53, parameter_51, parameter_45, parameter_43, parameter_41, parameter_35, parameter_33, parameter_31, parameter_25, parameter_23, parameter_21, parameter_15, parameter_13, parameter_11, parameter_5, constant_3, constant_2, constant_1, constant_0, parameter_1, parameter_0, parameter_3, parameter_2, parameter_4, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_12, parameter_14, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_22, parameter_24, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_32, parameter_34, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_42, parameter_44, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_52, parameter_54, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_62, parameter_64, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_72, parameter_74, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_82, parameter_84, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_92, parameter_94, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_102, parameter_104, parameter_107, parameter_106, parameter_108, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_116, parameter_118, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_126, parameter_128, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_136, parameter_138, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_146, parameter_148, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_156, parameter_158, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_166, parameter_168, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_176, parameter_178, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_186, parameter_188, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_196, parameter_198, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_206, parameter_208, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_216, parameter_218, parameter_223, parameter_220, parameter_222, parameter_221, parameter_224, parameter_226, parameter_228, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_236, parameter_238, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_246, parameter_248, parameter_253, parameter_250, parameter_252, parameter_251, parameter_254, parameter_256, parameter_258, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_266, parameter_268, parameter_271, parameter_270, parameter_272, parameter_275, parameter_274, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_282, parameter_281, parameter_283, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_291, parameter_290, parameter_292, parameter_293, parameter_294, parameter_295, parameter_296, parameter_298, parameter_297, parameter_299, parameter_300, parameter_301, parameter_302, parameter_303, parameter_304, parameter_307, parameter_306, parameter_308, parameter_309, parameter_310, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_323, parameter_322, parameter_324, parameter_325, parameter_326, parameter_327, parameter_328, parameter_330, parameter_329, parameter_331, parameter_332, parameter_333, parameter_334, parameter_335, parameter_336, parameter_339, parameter_338, parameter_340, parameter_341, parameter_342, parameter_343, parameter_344, parameter_346, parameter_345, parameter_347, parameter_348, parameter_349, parameter_350, parameter_351, parameter_352, parameter_355, parameter_354, parameter_356, parameter_357, parameter_358, parameter_359, parameter_360, parameter_362, parameter_361, parameter_363, parameter_364, parameter_365, parameter_366, parameter_367, parameter_368, parameter_371, parameter_370, parameter_372, parameter_373, parameter_374, parameter_375, parameter_376, parameter_378, parameter_377, parameter_379, parameter_380, parameter_381, parameter_382, parameter_383, parameter_384, parameter_387, parameter_386, parameter_388, parameter_389, parameter_390, parameter_391, parameter_392, parameter_394, parameter_393, parameter_395, parameter_396, parameter_397, parameter_398, parameter_399, parameter_400, parameter_403, parameter_402, parameter_404, parameter_405, parameter_406, parameter_407, parameter_408, parameter_410, parameter_409, parameter_411, parameter_412, parameter_413, parameter_414, parameter_415, parameter_416, parameter_419, parameter_418, parameter_420, parameter_421, parameter_422, parameter_423, parameter_424, parameter_426, parameter_425, parameter_427, parameter_428, parameter_429, parameter_430, parameter_431, parameter_432, parameter_435, parameter_434, parameter_436, parameter_437, parameter_438, parameter_439, parameter_440, parameter_442, parameter_441, parameter_443, parameter_444, parameter_445, parameter_446, parameter_447, parameter_448, parameter_451, parameter_450, parameter_452, parameter_453, parameter_454, parameter_455, parameter_456, parameter_458, parameter_457, parameter_459, parameter_460, parameter_461, parameter_462, parameter_463, parameter_464, parameter_467, parameter_466, parameter_468, parameter_469, parameter_470, parameter_471, parameter_472, parameter_474, parameter_473, parameter_475, parameter_476, parameter_477, parameter_478, parameter_479, parameter_480, parameter_483, parameter_482, parameter_484, parameter_485, parameter_486, parameter_487, parameter_488, parameter_490, parameter_489, parameter_491, parameter_492, parameter_493, parameter_494, parameter_495, parameter_496, parameter_499, parameter_498, parameter_500, parameter_501, parameter_502, parameter_503, parameter_504, parameter_506, parameter_505, parameter_507, parameter_508, parameter_509, parameter_510, parameter_511, parameter_512, parameter_515, parameter_514, parameter_516, parameter_517, parameter_518, parameter_519, parameter_520, parameter_522, parameter_521, parameter_523, parameter_524, parameter_525, parameter_526, parameter_527, parameter_528, parameter_531, parameter_530, parameter_532, parameter_533, parameter_534, parameter_535, parameter_536, parameter_538, parameter_537, parameter_539, parameter_540, parameter_541, parameter_542, parameter_543, parameter_544, parameter_547, parameter_546, parameter_548, parameter_549, parameter_550, parameter_551, parameter_552, parameter_554, parameter_553, parameter_555, parameter_556, parameter_557, parameter_558, parameter_559, parameter_560, parameter_563, parameter_562, parameter_564, parameter_565, parameter_566, parameter_567, parameter_568, parameter_570, parameter_569, parameter_571, parameter_572, parameter_573, parameter_574, parameter_575, parameter_576, parameter_579, parameter_578, parameter_580, parameter_581, parameter_582, parameter_583, parameter_584, parameter_586, parameter_585, parameter_587, parameter_588, parameter_589, parameter_590, parameter_591, parameter_592, parameter_595, parameter_594, parameter_596, parameter_599, parameter_598, parameter_600, parameter_601, parameter_602, parameter_603, parameter_604, parameter_606, parameter_605, parameter_607, parameter_608, parameter_609, parameter_610, parameter_611, parameter_612, parameter_615, parameter_614, parameter_616, parameter_617, parameter_618, parameter_619, parameter_620, parameter_622, parameter_621, parameter_623, parameter_624, parameter_625, parameter_626, parameter_627, parameter_628, parameter_631, parameter_630, parameter_632, parameter_633, parameter_634, parameter_635, parameter_636, parameter_638, parameter_637, parameter_639, parameter_640, parameter_641, parameter_642, parameter_643, parameter_644, parameter_647, parameter_646, parameter_648, parameter_649, parameter_650, parameter_651, parameter_652, parameter_654, parameter_653, parameter_655, parameter_656, parameter_657, parameter_658, parameter_659, parameter_660, parameter_663, parameter_662, parameter_664, parameter_665, parameter_666, parameter_667, parameter_668, parameter_670, parameter_669, parameter_671, parameter_672, parameter_673, parameter_674, parameter_675, parameter_676, parameter_679, parameter_678, parameter_680, parameter_681, parameter_682, parameter_683, parameter_684, parameter_686, parameter_685, parameter_687, parameter_688, parameter_689, parameter_690, parameter_691, parameter_692, parameter_695, parameter_694, parameter_696, parameter_697, parameter_698, parameter_699, parameter_700, parameter_702, parameter_701, parameter_703, parameter_704, parameter_705, parameter_706, parameter_707, parameter_711, parameter_708, parameter_710, parameter_709, parameter_712, parameter_713, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x3x224x224xf16, 64x3x4x4xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__0 = paddle._C_ops.add(conv2d_0, parameter_1)

        # pd_op.shape: (4xi32) <- (-1x64x56x56xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(add__0, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x64x3136xf16, None) <- (-1x64x56x56xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x64xf16) <- (-1x64x3136xf16)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.layer_norm: (-1x3136x64xf16, -3136xf32, -3136xf32) <- (-1x3136x64xf16, 64xf32, 64xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_0, parameter_2, parameter_3, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, constant_2, constant_2, constant_3]

        # pd_op.reshape_: (-1x56x56x64xf16, 0x-1x3136x64xf16) <- (-1x3136x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(layer_norm_0, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x56x56xf16) <- (-1x56x56x64xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [0, 3, 1, 2])

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(transpose_1, parameter_4, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__1 = paddle._C_ops.add(depthwise_conv2d_0, parameter_5)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__2 = paddle._C_ops.add(transpose_1, add__1)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__2, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(batch_norm__0, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__3 = paddle._C_ops.add(conv2d_1, parameter_11)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x5x5xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(add__3, parameter_12, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__4 = paddle._C_ops.add(depthwise_conv2d_1, parameter_13)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(add__4, parameter_14, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__5 = paddle._C_ops.add(conv2d_2, parameter_15)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__6 = paddle._C_ops.add(add__2, add__5)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__6, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(batch_norm__6, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__7 = paddle._C_ops.add(conv2d_3, parameter_21)

        # pd_op.gelu: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        gelu_0 = paddle._C_ops.gelu(add__7, False)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x256x56x56xf16, 64x256x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(gelu_0, parameter_22, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__8 = paddle._C_ops.add(conv2d_4, parameter_23)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__9 = paddle._C_ops.add(add__6, add__8)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(add__9, parameter_24, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__10 = paddle._C_ops.add(depthwise_conv2d_2, parameter_25)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__11 = paddle._C_ops.add(add__9, add__10)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__11, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(batch_norm__12, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__12 = paddle._C_ops.add(conv2d_5, parameter_31)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x5x5xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(add__12, parameter_32, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__13 = paddle._C_ops.add(depthwise_conv2d_3, parameter_33)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(add__13, parameter_34, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__14 = paddle._C_ops.add(conv2d_6, parameter_35)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__15 = paddle._C_ops.add(add__11, add__14)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__15, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(batch_norm__18, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__16 = paddle._C_ops.add(conv2d_7, parameter_41)

        # pd_op.gelu: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        gelu_1 = paddle._C_ops.gelu(add__16, False)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x256x56x56xf16, 64x256x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(gelu_1, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__17 = paddle._C_ops.add(conv2d_8, parameter_43)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__18 = paddle._C_ops.add(add__15, add__17)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x3x3xf16)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(add__18, parameter_44, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__19 = paddle._C_ops.add(depthwise_conv2d_4, parameter_45)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__20 = paddle._C_ops.add(add__18, add__19)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__20, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(batch_norm__24, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__21 = paddle._C_ops.add(conv2d_9, parameter_51)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x5x5xf16)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(add__21, parameter_52, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__22 = paddle._C_ops.add(depthwise_conv2d_5, parameter_53)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(add__22, parameter_54, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__23 = paddle._C_ops.add(conv2d_10, parameter_55)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__24 = paddle._C_ops.add(add__20, add__23)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__24, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(batch_norm__30, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__25 = paddle._C_ops.add(conv2d_11, parameter_61)

        # pd_op.gelu: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        gelu_2 = paddle._C_ops.gelu(add__25, False)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x256x56x56xf16, 64x256x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(gelu_2, parameter_62, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__26 = paddle._C_ops.add(conv2d_12, parameter_63)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__27 = paddle._C_ops.add(add__24, add__26)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x3x3xf16)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(add__27, parameter_64, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__28 = paddle._C_ops.add(depthwise_conv2d_6, parameter_65)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__29 = paddle._C_ops.add(add__27, add__28)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__29, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(batch_norm__36, parameter_70, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__30 = paddle._C_ops.add(conv2d_13, parameter_71)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x5x5xf16)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(add__30, parameter_72, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__31 = paddle._C_ops.add(depthwise_conv2d_7, parameter_73)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(add__31, parameter_74, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__32 = paddle._C_ops.add(conv2d_14, parameter_75)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__33 = paddle._C_ops.add(add__29, add__32)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__33, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(batch_norm__42, parameter_80, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__34 = paddle._C_ops.add(conv2d_15, parameter_81)

        # pd_op.gelu: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        gelu_3 = paddle._C_ops.gelu(add__34, False)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x256x56x56xf16, 64x256x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(gelu_3, parameter_82, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__35 = paddle._C_ops.add(conv2d_16, parameter_83)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__36 = paddle._C_ops.add(add__33, add__35)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x3x3xf16)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(add__36, parameter_84, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__37 = paddle._C_ops.add(depthwise_conv2d_8, parameter_85)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__38 = paddle._C_ops.add(add__36, add__37)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__38, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(batch_norm__48, parameter_90, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__39 = paddle._C_ops.add(conv2d_17, parameter_91)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x5x5xf16)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(add__39, parameter_92, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__40 = paddle._C_ops.add(depthwise_conv2d_9, parameter_93)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(add__40, parameter_94, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__41 = paddle._C_ops.add(conv2d_18, parameter_95)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__42 = paddle._C_ops.add(add__38, add__41)

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__42, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(batch_norm__54, parameter_100, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__43 = paddle._C_ops.add(conv2d_19, parameter_101)

        # pd_op.gelu: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        gelu_4 = paddle._C_ops.gelu(add__43, False)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x256x56x56xf16, 64x256x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(gelu_4, parameter_102, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__44 = paddle._C_ops.add(conv2d_20, parameter_103)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__45 = paddle._C_ops.add(add__42, add__44)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x64x56x56xf16, 128x64x2x2xf16)
        conv2d_21 = paddle._C_ops.conv2d(add__45, parameter_104, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__46 = paddle._C_ops.add(conv2d_21, parameter_105)

        # pd_op.shape: (4xi32) <- (-1x128x28x28xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(add__46, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x128x784xf16, None) <- (-1x128x28x28xf16)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__46, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x128xf16) <- (-1x128x784xf16)
        transpose_2 = paddle._C_ops.transpose(flatten__2, [0, 2, 1])

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_2, parameter_106, parameter_107, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, constant_4, constant_4, constant_5]

        # pd_op.reshape_: (-1x28x28x128xf16, 0x-1x784x128xf16) <- (-1x784x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(layer_norm_3, combine_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x128x28x28xf16) <- (-1x28x28x128xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__2, [0, 3, 1, 2])

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x3x3xf16)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(transpose_3, parameter_108, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__47 = paddle._C_ops.add(depthwise_conv2d_10, parameter_109)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__48 = paddle._C_ops.add(transpose_3, add__47)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__48, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(batch_norm__60, parameter_114, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__49 = paddle._C_ops.add(conv2d_22, parameter_115)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x5x5xf16)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(add__49, parameter_116, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__50 = paddle._C_ops.add(depthwise_conv2d_11, parameter_117)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(add__50, parameter_118, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__51 = paddle._C_ops.add(conv2d_23, parameter_119)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__52 = paddle._C_ops.add(add__48, add__51)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__52, parameter_120, parameter_121, parameter_122, parameter_123, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(batch_norm__66, parameter_124, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__53 = paddle._C_ops.add(conv2d_24, parameter_125)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_5 = paddle._C_ops.gelu(add__53, False)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(gelu_5, parameter_126, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__54 = paddle._C_ops.add(conv2d_25, parameter_127)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__55 = paddle._C_ops.add(add__52, add__54)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x3x3xf16)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(add__55, parameter_128, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__56 = paddle._C_ops.add(depthwise_conv2d_12, parameter_129)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__57 = paddle._C_ops.add(add__55, add__56)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__57, parameter_130, parameter_131, parameter_132, parameter_133, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(batch_norm__72, parameter_134, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__58 = paddle._C_ops.add(conv2d_26, parameter_135)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x5x5xf16)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(add__58, parameter_136, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__59 = paddle._C_ops.add(depthwise_conv2d_13, parameter_137)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(add__59, parameter_138, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__60 = paddle._C_ops.add(conv2d_27, parameter_139)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__61 = paddle._C_ops.add(add__57, add__60)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__61, parameter_140, parameter_141, parameter_142, parameter_143, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(batch_norm__78, parameter_144, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__62 = paddle._C_ops.add(conv2d_28, parameter_145)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_6 = paddle._C_ops.gelu(add__62, False)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(gelu_6, parameter_146, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__63 = paddle._C_ops.add(conv2d_29, parameter_147)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__64 = paddle._C_ops.add(add__61, add__63)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x3x3xf16)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(add__64, parameter_148, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__65 = paddle._C_ops.add(depthwise_conv2d_14, parameter_149)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__66 = paddle._C_ops.add(add__64, add__65)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__66, parameter_150, parameter_151, parameter_152, parameter_153, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(batch_norm__84, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__67 = paddle._C_ops.add(conv2d_30, parameter_155)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x5x5xf16)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(add__67, parameter_156, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__68 = paddle._C_ops.add(depthwise_conv2d_15, parameter_157)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(add__68, parameter_158, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__69 = paddle._C_ops.add(conv2d_31, parameter_159)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__70 = paddle._C_ops.add(add__66, add__69)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__70, parameter_160, parameter_161, parameter_162, parameter_163, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(batch_norm__90, parameter_164, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__71 = paddle._C_ops.add(conv2d_32, parameter_165)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_7 = paddle._C_ops.gelu(add__71, False)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(gelu_7, parameter_166, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__72 = paddle._C_ops.add(conv2d_33, parameter_167)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__73 = paddle._C_ops.add(add__70, add__72)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x3x3xf16)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(add__73, parameter_168, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__74 = paddle._C_ops.add(depthwise_conv2d_16, parameter_169)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__75 = paddle._C_ops.add(add__73, add__74)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__75, parameter_170, parameter_171, parameter_172, parameter_173, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(batch_norm__96, parameter_174, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__76 = paddle._C_ops.add(conv2d_34, parameter_175)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x5x5xf16)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(add__76, parameter_176, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__77 = paddle._C_ops.add(depthwise_conv2d_17, parameter_177)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(add__77, parameter_178, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__78 = paddle._C_ops.add(conv2d_35, parameter_179)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__79 = paddle._C_ops.add(add__75, add__78)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__79, parameter_180, parameter_181, parameter_182, parameter_183, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(batch_norm__102, parameter_184, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__80 = paddle._C_ops.add(conv2d_36, parameter_185)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_8 = paddle._C_ops.gelu(add__80, False)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(gelu_8, parameter_186, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__81 = paddle._C_ops.add(conv2d_37, parameter_187)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__82 = paddle._C_ops.add(add__79, add__81)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x3x3xf16)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(add__82, parameter_188, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__83 = paddle._C_ops.add(depthwise_conv2d_18, parameter_189)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__84 = paddle._C_ops.add(add__82, add__83)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__84, parameter_190, parameter_191, parameter_192, parameter_193, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(batch_norm__108, parameter_194, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__85 = paddle._C_ops.add(conv2d_38, parameter_195)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x5x5xf16)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(add__85, parameter_196, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__86 = paddle._C_ops.add(depthwise_conv2d_19, parameter_197)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(add__86, parameter_198, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__87 = paddle._C_ops.add(conv2d_39, parameter_199)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__88 = paddle._C_ops.add(add__84, add__87)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__88, parameter_200, parameter_201, parameter_202, parameter_203, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(batch_norm__114, parameter_204, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__89 = paddle._C_ops.add(conv2d_40, parameter_205)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_9 = paddle._C_ops.gelu(add__89, False)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(gelu_9, parameter_206, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__90 = paddle._C_ops.add(conv2d_41, parameter_207)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__91 = paddle._C_ops.add(add__88, add__90)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x3x3xf16)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(add__91, parameter_208, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__92 = paddle._C_ops.add(depthwise_conv2d_20, parameter_209)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__93 = paddle._C_ops.add(add__91, add__92)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__93, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(batch_norm__120, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__94 = paddle._C_ops.add(conv2d_42, parameter_215)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x5x5xf16)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(add__94, parameter_216, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__95 = paddle._C_ops.add(depthwise_conv2d_21, parameter_217)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(add__95, parameter_218, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__96 = paddle._C_ops.add(conv2d_43, parameter_219)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__97 = paddle._C_ops.add(add__93, add__96)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__97, parameter_220, parameter_221, parameter_222, parameter_223, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(batch_norm__126, parameter_224, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__98 = paddle._C_ops.add(conv2d_44, parameter_225)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_10 = paddle._C_ops.gelu(add__98, False)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(gelu_10, parameter_226, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__99 = paddle._C_ops.add(conv2d_45, parameter_227)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__100 = paddle._C_ops.add(add__97, add__99)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x3x3xf16)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(add__100, parameter_228, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__101 = paddle._C_ops.add(depthwise_conv2d_22, parameter_229)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__102 = paddle._C_ops.add(add__100, add__101)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__102, parameter_230, parameter_231, parameter_232, parameter_233, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(batch_norm__132, parameter_234, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__103 = paddle._C_ops.add(conv2d_46, parameter_235)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x5x5xf16)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(add__103, parameter_236, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__104 = paddle._C_ops.add(depthwise_conv2d_23, parameter_237)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(add__104, parameter_238, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__105 = paddle._C_ops.add(conv2d_47, parameter_239)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__106 = paddle._C_ops.add(add__102, add__105)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__106, parameter_240, parameter_241, parameter_242, parameter_243, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(batch_norm__138, parameter_244, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__107 = paddle._C_ops.add(conv2d_48, parameter_245)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_11 = paddle._C_ops.gelu(add__107, False)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(gelu_11, parameter_246, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__108 = paddle._C_ops.add(conv2d_49, parameter_247)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__109 = paddle._C_ops.add(add__106, add__108)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x3x3xf16)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(add__109, parameter_248, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__110 = paddle._C_ops.add(depthwise_conv2d_24, parameter_249)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__111 = paddle._C_ops.add(add__109, add__110)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__111, parameter_250, parameter_251, parameter_252, parameter_253, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(batch_norm__144, parameter_254, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__112 = paddle._C_ops.add(conv2d_50, parameter_255)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x5x5xf16)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(add__112, parameter_256, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__113 = paddle._C_ops.add(depthwise_conv2d_25, parameter_257)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x128x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(add__113, parameter_258, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__114 = paddle._C_ops.add(conv2d_51, parameter_259)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__115 = paddle._C_ops.add(add__111, add__114)

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__115, parameter_260, parameter_261, parameter_262, parameter_263, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(batch_norm__150, parameter_264, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__116 = paddle._C_ops.add(conv2d_52, parameter_265)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_12 = paddle._C_ops.gelu(add__116, False)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_53 = paddle._C_ops.conv2d(gelu_12, parameter_266, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__117 = paddle._C_ops.add(conv2d_53, parameter_267)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__118 = paddle._C_ops.add(add__115, add__117)

        # pd_op.conv2d: (-1x320x14x14xf16) <- (-1x128x28x28xf16, 320x128x2x2xf16)
        conv2d_54 = paddle._C_ops.conv2d(add__118, parameter_268, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__119 = paddle._C_ops.add(conv2d_54, parameter_269)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(add__119, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__119, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_4 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_4, parameter_270, parameter_271, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_2, constant_6, constant_6, constant_7]

        # pd_op.reshape_: (-1x14x14x320xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(layer_norm_6, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x320x14x14xf16) <- (-1x14x14x320xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__4, [0, 3, 1, 2])

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(transpose_5, parameter_272, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__120 = paddle._C_ops.add(depthwise_conv2d_26, parameter_273)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__121 = paddle._C_ops.add(transpose_5, add__120)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(add__121, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_3, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__6, flatten__7 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__121, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_6 = paddle._C_ops.transpose(flatten__6, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_6, parameter_274, parameter_275, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(layer_norm_9, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_4, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_0 = paddle.matmul(layer_norm_9, parameter_276, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__122 = paddle._C_ops.add(matmul_0, parameter_277)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_4, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__122, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_7 = paddle._C_ops.transpose(reshape__6, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(transpose_7, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_7, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_7, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_8 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_1 = paddle.matmul(slice_5, transpose_8, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__0 = paddle._C_ops.scale(matmul_1, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__0 = paddle._C_ops.softmax(scale__0, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_2 = paddle.matmul(softmax__0, slice_7, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_9 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_4, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_9, combine_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_3 = paddle.matmul(reshape__8, parameter_278, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__123 = paddle._C_ops.add(matmul_3, parameter_279)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_0 = parameter_280 * add__123

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__124 = paddle._C_ops.add(transpose_6, multiply_0)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__124, parameter_281, parameter_282, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_4 = paddle.matmul(layer_norm_12, parameter_283, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__125 = paddle._C_ops.add(matmul_4, parameter_284)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_13 = paddle._C_ops.gelu(add__125, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_5 = paddle.matmul(gelu_13, parameter_285, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__126 = paddle._C_ops.add(matmul_5, parameter_286)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_1 = parameter_287 * add__126

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__127 = paddle._C_ops.add(add__124, multiply_1)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_10 = paddle._C_ops.transpose(add__127, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_3, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_10, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_27 = paddle._C_ops.depthwise_conv2d(reshape__10, parameter_288, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__128 = paddle._C_ops.add(depthwise_conv2d_27, parameter_289)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__129 = paddle._C_ops.add(reshape__10, add__128)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(add__129, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_5, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__8, flatten__9 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__129, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_11 = paddle._C_ops.transpose(flatten__8, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_11, parameter_290, parameter_291, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(layer_norm_15, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_6, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_6 = paddle.matmul(layer_norm_15, parameter_292, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__130 = paddle._C_ops.add(matmul_6, parameter_293)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_9, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__130, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_12 = paddle._C_ops.transpose(reshape__12, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_12, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_12, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(transpose_12, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_13 = paddle._C_ops.transpose(slice_11, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_7 = paddle.matmul(slice_10, transpose_13, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__1 = paddle._C_ops.scale(matmul_7, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__1 = paddle._C_ops.softmax(scale__1, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_8 = paddle.matmul(softmax__1, slice_12, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_14 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_7 = [slice_9, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_14, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_9 = paddle.matmul(reshape__14, parameter_294, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__131 = paddle._C_ops.add(matmul_9, parameter_295)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_2 = parameter_296 * add__131

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__132 = paddle._C_ops.add(transpose_11, multiply_2)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__132, parameter_297, parameter_298, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_10 = paddle.matmul(layer_norm_18, parameter_299, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__133 = paddle._C_ops.add(matmul_10, parameter_300)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_14 = paddle._C_ops.gelu(add__133, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_11 = paddle.matmul(gelu_14, parameter_301, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__134 = paddle._C_ops.add(matmul_11, parameter_302)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_3 = parameter_303 * add__134

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__135 = paddle._C_ops.add(add__132, multiply_3)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_15 = paddle._C_ops.transpose(add__135, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_8, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_15, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_28 = paddle._C_ops.depthwise_conv2d(reshape__16, parameter_304, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__136 = paddle._C_ops.add(depthwise_conv2d_28, parameter_305)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__137 = paddle._C_ops.add(reshape__16, add__136)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(add__137, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_7, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__10, flatten__11 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__137, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_16 = paddle._C_ops.transpose(flatten__10, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_16, parameter_306, parameter_307, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(layer_norm_21, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_8, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_12 = paddle.matmul(layer_norm_21, parameter_308, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__138 = paddle._C_ops.add(matmul_12, parameter_309)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_14, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__138, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_17 = paddle._C_ops.transpose(reshape__18, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_17, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_17, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_17, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_18 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_13 = paddle.matmul(slice_15, transpose_18, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__2 = paddle._C_ops.scale(matmul_13, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__2 = paddle._C_ops.softmax(scale__2, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_14 = paddle.matmul(softmax__2, slice_17, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_19 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_14, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_19, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_15 = paddle.matmul(reshape__20, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__139 = paddle._C_ops.add(matmul_15, parameter_311)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_4 = parameter_312 * add__139

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__140 = paddle._C_ops.add(transpose_16, multiply_4)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__140, parameter_313, parameter_314, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_16 = paddle.matmul(layer_norm_24, parameter_315, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__141 = paddle._C_ops.add(matmul_16, parameter_316)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_15 = paddle._C_ops.gelu(add__141, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_17 = paddle.matmul(gelu_15, parameter_317, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__142 = paddle._C_ops.add(matmul_17, parameter_318)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_5 = parameter_319 * add__142

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__143 = paddle._C_ops.add(add__140, multiply_5)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_20 = paddle._C_ops.transpose(add__143, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_13, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_20, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_29 = paddle._C_ops.depthwise_conv2d(reshape__22, parameter_320, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__144 = paddle._C_ops.add(depthwise_conv2d_29, parameter_321)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__145 = paddle._C_ops.add(reshape__22, add__144)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(add__145, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(shape_9, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__12, flatten__13 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__145, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_21 = paddle._C_ops.transpose(flatten__12, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_21, parameter_322, parameter_323, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_10 = paddle._C_ops.shape(paddle.cast(layer_norm_27, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(shape_10, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_18 = paddle.matmul(layer_norm_27, parameter_324, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__146 = paddle._C_ops.add(matmul_18, parameter_325)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_19, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__146, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_22 = paddle._C_ops.transpose(reshape__24, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(transpose_22, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(transpose_22, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_22, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_23 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_19 = paddle.matmul(slice_20, transpose_23, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__3 = paddle._C_ops.scale(matmul_19, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__3 = paddle._C_ops.softmax(scale__3, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_20 = paddle.matmul(softmax__3, slice_22, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_24 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_13 = [slice_19, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_24, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_21 = paddle.matmul(reshape__26, parameter_326, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__147 = paddle._C_ops.add(matmul_21, parameter_327)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_6 = parameter_328 * add__147

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__148 = paddle._C_ops.add(transpose_21, multiply_6)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__148, parameter_329, parameter_330, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_22 = paddle.matmul(layer_norm_30, parameter_331, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__149 = paddle._C_ops.add(matmul_22, parameter_332)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_16 = paddle._C_ops.gelu(add__149, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_23 = paddle.matmul(gelu_16, parameter_333, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__150 = paddle._C_ops.add(matmul_23, parameter_334)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_7 = parameter_335 * add__150

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__151 = paddle._C_ops.add(add__148, multiply_7)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_25 = paddle._C_ops.transpose(add__151, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_18, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_25, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_30 = paddle._C_ops.depthwise_conv2d(reshape__28, parameter_336, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__152 = paddle._C_ops.add(depthwise_conv2d_30, parameter_337)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__153 = paddle._C_ops.add(reshape__28, add__152)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(add__153, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(shape_11, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__14, flatten__15 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__153, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_26 = paddle._C_ops.transpose(flatten__14, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_26, parameter_338, parameter_339, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_12 = paddle._C_ops.shape(paddle.cast(layer_norm_33, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_12, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_24 = paddle.matmul(layer_norm_33, parameter_340, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__154 = paddle._C_ops.add(matmul_24, parameter_341)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_15 = [slice_24, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__154, combine_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_27 = paddle._C_ops.transpose(reshape__30, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_27, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_27, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_27, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_28 = paddle._C_ops.transpose(slice_26, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_25 = paddle.matmul(slice_25, transpose_28, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__4 = paddle._C_ops.scale(matmul_25, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__4 = paddle._C_ops.softmax(scale__4, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_26 = paddle.matmul(softmax__4, slice_27, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_29 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_16 = [slice_24, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_29, combine_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_27 = paddle.matmul(reshape__32, parameter_342, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__155 = paddle._C_ops.add(matmul_27, parameter_343)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_8 = parameter_344 * add__155

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__156 = paddle._C_ops.add(transpose_26, multiply_8)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__156, parameter_345, parameter_346, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_28 = paddle.matmul(layer_norm_36, parameter_347, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__157 = paddle._C_ops.add(matmul_28, parameter_348)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_17 = paddle._C_ops.gelu(add__157, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_29 = paddle.matmul(gelu_17, parameter_349, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__158 = paddle._C_ops.add(matmul_29, parameter_350)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_9 = parameter_351 * add__158

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__159 = paddle._C_ops.add(add__156, multiply_9)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_30 = paddle._C_ops.transpose(add__159, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_23, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_30, combine_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_31 = paddle._C_ops.depthwise_conv2d(reshape__34, parameter_352, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__160 = paddle._C_ops.add(depthwise_conv2d_31, parameter_353)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__161 = paddle._C_ops.add(reshape__34, add__160)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_13 = paddle._C_ops.shape(paddle.cast(add__161, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_13, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__16, flatten__17 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__161, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_31 = paddle._C_ops.transpose(flatten__16, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_31, parameter_354, parameter_355, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_14 = paddle._C_ops.shape(paddle.cast(layer_norm_39, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_14, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_30 = paddle.matmul(layer_norm_39, parameter_356, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__162 = paddle._C_ops.add(matmul_30, parameter_357)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_29, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__162, combine_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_32 = paddle._C_ops.transpose(reshape__36, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_32, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_32, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(transpose_32, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_33 = paddle._C_ops.transpose(slice_31, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_31 = paddle.matmul(slice_30, transpose_33, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__5 = paddle._C_ops.scale(matmul_31, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__5 = paddle._C_ops.softmax(scale__5, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_32 = paddle.matmul(softmax__5, slice_32, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_34 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_19 = [slice_29, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_34, combine_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_33 = paddle.matmul(reshape__38, parameter_358, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__163 = paddle._C_ops.add(matmul_33, parameter_359)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_10 = parameter_360 * add__163

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__164 = paddle._C_ops.add(transpose_31, multiply_10)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__164, parameter_361, parameter_362, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_34 = paddle.matmul(layer_norm_42, parameter_363, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__165 = paddle._C_ops.add(matmul_34, parameter_364)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_18 = paddle._C_ops.gelu(add__165, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_35 = paddle.matmul(gelu_18, parameter_365, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__166 = paddle._C_ops.add(matmul_35, parameter_366)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_11 = parameter_367 * add__166

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__167 = paddle._C_ops.add(add__164, multiply_11)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_35 = paddle._C_ops.transpose(add__167, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_28, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_35, combine_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_32 = paddle._C_ops.depthwise_conv2d(reshape__40, parameter_368, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__168 = paddle._C_ops.add(depthwise_conv2d_32, parameter_369)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__169 = paddle._C_ops.add(reshape__40, add__168)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_15 = paddle._C_ops.shape(paddle.cast(add__169, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(shape_15, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__18, flatten__19 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__169, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_36 = paddle._C_ops.transpose(flatten__18, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_36, parameter_370, parameter_371, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_16 = paddle._C_ops.shape(paddle.cast(layer_norm_45, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(shape_16, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_36 = paddle.matmul(layer_norm_45, parameter_372, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__170 = paddle._C_ops.add(matmul_36, parameter_373)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_34, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__170, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_37 = paddle._C_ops.transpose(reshape__42, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_37, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(transpose_37, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(transpose_37, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_38 = paddle._C_ops.transpose(slice_36, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_37 = paddle.matmul(slice_35, transpose_38, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__6 = paddle._C_ops.scale(matmul_37, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__6 = paddle._C_ops.softmax(scale__6, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_38 = paddle.matmul(softmax__6, slice_37, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_39 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_22 = [slice_34, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_39, combine_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_39 = paddle.matmul(reshape__44, parameter_374, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__171 = paddle._C_ops.add(matmul_39, parameter_375)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_12 = parameter_376 * add__171

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__172 = paddle._C_ops.add(transpose_36, multiply_12)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__172, parameter_377, parameter_378, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_40 = paddle.matmul(layer_norm_48, parameter_379, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__173 = paddle._C_ops.add(matmul_40, parameter_380)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_19 = paddle._C_ops.gelu(add__173, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_41 = paddle.matmul(gelu_19, parameter_381, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__174 = paddle._C_ops.add(matmul_41, parameter_382)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_13 = parameter_383 * add__174

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__175 = paddle._C_ops.add(add__172, multiply_13)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_40 = paddle._C_ops.transpose(add__175, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_33, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_40, combine_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_33 = paddle._C_ops.depthwise_conv2d(reshape__46, parameter_384, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__176 = paddle._C_ops.add(depthwise_conv2d_33, parameter_385)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__177 = paddle._C_ops.add(reshape__46, add__176)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_17 = paddle._C_ops.shape(paddle.cast(add__177, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(shape_17, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__20, flatten__21 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__177, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_41 = paddle._C_ops.transpose(flatten__20, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_41, parameter_386, parameter_387, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_18 = paddle._C_ops.shape(paddle.cast(layer_norm_51, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(shape_18, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_42 = paddle.matmul(layer_norm_51, parameter_388, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__178 = paddle._C_ops.add(matmul_42, parameter_389)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_39, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__178, combine_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_42 = paddle._C_ops.transpose(reshape__48, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(transpose_42, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(transpose_42, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(transpose_42, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_43 = paddle._C_ops.transpose(slice_41, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_43 = paddle.matmul(slice_40, transpose_43, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__7 = paddle._C_ops.scale(matmul_43, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__7 = paddle._C_ops.softmax(scale__7, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_44 = paddle.matmul(softmax__7, slice_42, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_44 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_39, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_44, combine_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_45 = paddle.matmul(reshape__50, parameter_390, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__179 = paddle._C_ops.add(matmul_45, parameter_391)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_14 = parameter_392 * add__179

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__180 = paddle._C_ops.add(transpose_41, multiply_14)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__180, parameter_393, parameter_394, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_46 = paddle.matmul(layer_norm_54, parameter_395, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__181 = paddle._C_ops.add(matmul_46, parameter_396)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_20 = paddle._C_ops.gelu(add__181, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_47 = paddle.matmul(gelu_20, parameter_397, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__182 = paddle._C_ops.add(matmul_47, parameter_398)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_15 = parameter_399 * add__182

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__183 = paddle._C_ops.add(add__180, multiply_15)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_45 = paddle._C_ops.transpose(add__183, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_38, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_45, combine_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_34 = paddle._C_ops.depthwise_conv2d(reshape__52, parameter_400, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__184 = paddle._C_ops.add(depthwise_conv2d_34, parameter_401)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__185 = paddle._C_ops.add(reshape__52, add__184)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_19 = paddle._C_ops.shape(paddle.cast(add__185, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(shape_19, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__22, flatten__23 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__185, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_46 = paddle._C_ops.transpose(flatten__22, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_46, parameter_402, parameter_403, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_20 = paddle._C_ops.shape(paddle.cast(layer_norm_57, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(shape_20, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_48 = paddle.matmul(layer_norm_57, parameter_404, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__186 = paddle._C_ops.add(matmul_48, parameter_405)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_27 = [slice_44, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__186, combine_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_47 = paddle._C_ops.transpose(reshape__54, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(transpose_47, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(transpose_47, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(transpose_47, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_48 = paddle._C_ops.transpose(slice_46, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_49 = paddle.matmul(slice_45, transpose_48, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__8 = paddle._C_ops.scale(matmul_49, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__8 = paddle._C_ops.softmax(scale__8, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_50 = paddle.matmul(softmax__8, slice_47, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_49 = paddle._C_ops.transpose(matmul_50, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_28 = [slice_44, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_49, combine_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_51 = paddle.matmul(reshape__56, parameter_406, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__187 = paddle._C_ops.add(matmul_51, parameter_407)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_16 = parameter_408 * add__187

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__188 = paddle._C_ops.add(transpose_46, multiply_16)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__188, parameter_409, parameter_410, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_52 = paddle.matmul(layer_norm_60, parameter_411, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__189 = paddle._C_ops.add(matmul_52, parameter_412)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_21 = paddle._C_ops.gelu(add__189, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_53 = paddle.matmul(gelu_21, parameter_413, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__190 = paddle._C_ops.add(matmul_53, parameter_414)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_17 = parameter_415 * add__190

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__191 = paddle._C_ops.add(add__188, multiply_17)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_50 = paddle._C_ops.transpose(add__191, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_43, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_50, combine_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_35 = paddle._C_ops.depthwise_conv2d(reshape__58, parameter_416, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__192 = paddle._C_ops.add(depthwise_conv2d_35, parameter_417)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__193 = paddle._C_ops.add(reshape__58, add__192)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_21 = paddle._C_ops.shape(paddle.cast(add__193, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(shape_21, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__24, flatten__25 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__193, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_51 = paddle._C_ops.transpose(flatten__24, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_51, parameter_418, parameter_419, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_22 = paddle._C_ops.shape(paddle.cast(layer_norm_63, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(shape_22, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_54 = paddle.matmul(layer_norm_63, parameter_420, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__194 = paddle._C_ops.add(matmul_54, parameter_421)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_30 = [slice_49, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__194, combine_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_52 = paddle._C_ops.transpose(reshape__60, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(transpose_52, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(transpose_52, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(transpose_52, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_53 = paddle._C_ops.transpose(slice_51, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_55 = paddle.matmul(slice_50, transpose_53, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__9 = paddle._C_ops.scale(matmul_55, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__9 = paddle._C_ops.softmax(scale__9, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_56 = paddle.matmul(softmax__9, slice_52, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_54 = paddle._C_ops.transpose(matmul_56, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_31 = [slice_49, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_54, combine_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_57 = paddle.matmul(reshape__62, parameter_422, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__195 = paddle._C_ops.add(matmul_57, parameter_423)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_18 = parameter_424 * add__195

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__196 = paddle._C_ops.add(transpose_51, multiply_18)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__196, parameter_425, parameter_426, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_58 = paddle.matmul(layer_norm_66, parameter_427, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__197 = paddle._C_ops.add(matmul_58, parameter_428)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_22 = paddle._C_ops.gelu(add__197, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_59 = paddle.matmul(gelu_22, parameter_429, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__198 = paddle._C_ops.add(matmul_59, parameter_430)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_19 = parameter_431 * add__198

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__199 = paddle._C_ops.add(add__196, multiply_19)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_55 = paddle._C_ops.transpose(add__199, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_48, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_55, combine_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_36 = paddle._C_ops.depthwise_conv2d(reshape__64, parameter_432, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__200 = paddle._C_ops.add(depthwise_conv2d_36, parameter_433)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__201 = paddle._C_ops.add(reshape__64, add__200)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_23 = paddle._C_ops.shape(paddle.cast(add__201, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(shape_23, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__26, flatten__27 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__201, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_56 = paddle._C_ops.transpose(flatten__26, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_56, parameter_434, parameter_435, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_24 = paddle._C_ops.shape(paddle.cast(layer_norm_69, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(shape_24, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_60 = paddle.matmul(layer_norm_69, parameter_436, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__202 = paddle._C_ops.add(matmul_60, parameter_437)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_54, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__202, combine_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_57 = paddle._C_ops.transpose(reshape__66, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(transpose_57, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(transpose_57, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(transpose_57, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_58 = paddle._C_ops.transpose(slice_56, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_61 = paddle.matmul(slice_55, transpose_58, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__10 = paddle._C_ops.scale(matmul_61, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__10 = paddle._C_ops.softmax(scale__10, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_62 = paddle.matmul(softmax__10, slice_57, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_59 = paddle._C_ops.transpose(matmul_62, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_34 = [slice_54, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_59, combine_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_63 = paddle.matmul(reshape__68, parameter_438, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__203 = paddle._C_ops.add(matmul_63, parameter_439)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_20 = parameter_440 * add__203

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__204 = paddle._C_ops.add(transpose_56, multiply_20)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__204, parameter_441, parameter_442, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_64 = paddle.matmul(layer_norm_72, parameter_443, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__205 = paddle._C_ops.add(matmul_64, parameter_444)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_23 = paddle._C_ops.gelu(add__205, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_65 = paddle.matmul(gelu_23, parameter_445, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__206 = paddle._C_ops.add(matmul_65, parameter_446)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_21 = parameter_447 * add__206

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__207 = paddle._C_ops.add(add__204, multiply_21)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_60 = paddle._C_ops.transpose(add__207, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_53, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_60, combine_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_37 = paddle._C_ops.depthwise_conv2d(reshape__70, parameter_448, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__208 = paddle._C_ops.add(depthwise_conv2d_37, parameter_449)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__209 = paddle._C_ops.add(reshape__70, add__208)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_25 = paddle._C_ops.shape(paddle.cast(add__209, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(shape_25, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__28, flatten__29 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__209, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_61 = paddle._C_ops.transpose(flatten__28, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_61, parameter_450, parameter_451, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_26 = paddle._C_ops.shape(paddle.cast(layer_norm_75, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(shape_26, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_66 = paddle.matmul(layer_norm_75, parameter_452, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__210 = paddle._C_ops.add(matmul_66, parameter_453)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_36 = [slice_59, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__210, combine_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_62 = paddle._C_ops.transpose(reshape__72, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(transpose_62, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(transpose_62, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(transpose_62, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_63 = paddle._C_ops.transpose(slice_61, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_67 = paddle.matmul(slice_60, transpose_63, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__11 = paddle._C_ops.scale(matmul_67, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__11 = paddle._C_ops.softmax(scale__11, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_68 = paddle.matmul(softmax__11, slice_62, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_64 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_37 = [slice_59, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_64, combine_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_69 = paddle.matmul(reshape__74, parameter_454, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__211 = paddle._C_ops.add(matmul_69, parameter_455)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_22 = parameter_456 * add__211

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__212 = paddle._C_ops.add(transpose_61, multiply_22)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__212, parameter_457, parameter_458, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_70 = paddle.matmul(layer_norm_78, parameter_459, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__213 = paddle._C_ops.add(matmul_70, parameter_460)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_24 = paddle._C_ops.gelu(add__213, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_71 = paddle.matmul(gelu_24, parameter_461, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__214 = paddle._C_ops.add(matmul_71, parameter_462)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_23 = parameter_463 * add__214

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__215 = paddle._C_ops.add(add__212, multiply_23)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_65 = paddle._C_ops.transpose(add__215, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_58, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_65, combine_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_38 = paddle._C_ops.depthwise_conv2d(reshape__76, parameter_464, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__216 = paddle._C_ops.add(depthwise_conv2d_38, parameter_465)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__217 = paddle._C_ops.add(reshape__76, add__216)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_27 = paddle._C_ops.shape(paddle.cast(add__217, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(shape_27, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__30, flatten__31 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__217, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_66 = paddle._C_ops.transpose(flatten__30, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_66, parameter_466, parameter_467, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_28 = paddle._C_ops.shape(paddle.cast(layer_norm_81, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(shape_28, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_72 = paddle.matmul(layer_norm_81, parameter_468, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__218 = paddle._C_ops.add(matmul_72, parameter_469)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_39 = [slice_64, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__218, combine_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_67 = paddle._C_ops.transpose(reshape__78, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(transpose_67, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(transpose_67, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(transpose_67, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_68 = paddle._C_ops.transpose(slice_66, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_73 = paddle.matmul(slice_65, transpose_68, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__12 = paddle._C_ops.scale(matmul_73, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__12 = paddle._C_ops.softmax(scale__12, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_74 = paddle.matmul(softmax__12, slice_67, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_69 = paddle._C_ops.transpose(matmul_74, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_40 = [slice_64, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_69, combine_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_75 = paddle.matmul(reshape__80, parameter_470, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__219 = paddle._C_ops.add(matmul_75, parameter_471)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_24 = parameter_472 * add__219

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__220 = paddle._C_ops.add(transpose_66, multiply_24)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__220, parameter_473, parameter_474, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_76 = paddle.matmul(layer_norm_84, parameter_475, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__221 = paddle._C_ops.add(matmul_76, parameter_476)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_25 = paddle._C_ops.gelu(add__221, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_77 = paddle.matmul(gelu_25, parameter_477, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__222 = paddle._C_ops.add(matmul_77, parameter_478)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_25 = parameter_479 * add__222

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__223 = paddle._C_ops.add(add__220, multiply_25)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_70 = paddle._C_ops.transpose(add__223, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_63, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_70, combine_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_39 = paddle._C_ops.depthwise_conv2d(reshape__82, parameter_480, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__224 = paddle._C_ops.add(depthwise_conv2d_39, parameter_481)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__225 = paddle._C_ops.add(reshape__82, add__224)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_29 = paddle._C_ops.shape(paddle.cast(add__225, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(shape_29, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__32, flatten__33 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__225, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_71 = paddle._C_ops.transpose(flatten__32, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_71, parameter_482, parameter_483, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_30 = paddle._C_ops.shape(paddle.cast(layer_norm_87, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(shape_30, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_78 = paddle.matmul(layer_norm_87, parameter_484, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__226 = paddle._C_ops.add(matmul_78, parameter_485)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_69, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__226, combine_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_72 = paddle._C_ops.transpose(reshape__84, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(transpose_72, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(transpose_72, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(transpose_72, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_73 = paddle._C_ops.transpose(slice_71, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_79 = paddle.matmul(slice_70, transpose_73, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__13 = paddle._C_ops.scale(matmul_79, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__13 = paddle._C_ops.softmax(scale__13, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_80 = paddle.matmul(softmax__13, slice_72, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_74 = paddle._C_ops.transpose(matmul_80, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_43 = [slice_69, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_74, combine_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_81 = paddle.matmul(reshape__86, parameter_486, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__227 = paddle._C_ops.add(matmul_81, parameter_487)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_26 = parameter_488 * add__227

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__228 = paddle._C_ops.add(transpose_71, multiply_26)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__228, parameter_489, parameter_490, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_82 = paddle.matmul(layer_norm_90, parameter_491, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__229 = paddle._C_ops.add(matmul_82, parameter_492)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_26 = paddle._C_ops.gelu(add__229, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_83 = paddle.matmul(gelu_26, parameter_493, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__230 = paddle._C_ops.add(matmul_83, parameter_494)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_27 = parameter_495 * add__230

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__231 = paddle._C_ops.add(add__228, multiply_27)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_75 = paddle._C_ops.transpose(add__231, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_68, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_75, combine_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_40 = paddle._C_ops.depthwise_conv2d(reshape__88, parameter_496, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__232 = paddle._C_ops.add(depthwise_conv2d_40, parameter_497)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__233 = paddle._C_ops.add(reshape__88, add__232)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_31 = paddle._C_ops.shape(paddle.cast(add__233, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(shape_31, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__34, flatten__35 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__233, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_76 = paddle._C_ops.transpose(flatten__34, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_76, parameter_498, parameter_499, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_32 = paddle._C_ops.shape(paddle.cast(layer_norm_93, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(shape_32, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_84 = paddle.matmul(layer_norm_93, parameter_500, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__234 = paddle._C_ops.add(matmul_84, parameter_501)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [slice_74, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__234, combine_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_77 = paddle._C_ops.transpose(reshape__90, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(transpose_77, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(transpose_77, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(transpose_77, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_78 = paddle._C_ops.transpose(slice_76, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_85 = paddle.matmul(slice_75, transpose_78, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__14 = paddle._C_ops.scale(matmul_85, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__14 = paddle._C_ops.softmax(scale__14, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_86 = paddle.matmul(softmax__14, slice_77, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_79 = paddle._C_ops.transpose(matmul_86, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_46 = [slice_74, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_79, combine_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_87 = paddle.matmul(reshape__92, parameter_502, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__235 = paddle._C_ops.add(matmul_87, parameter_503)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_28 = parameter_504 * add__235

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__236 = paddle._C_ops.add(transpose_76, multiply_28)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__236, parameter_505, parameter_506, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_88 = paddle.matmul(layer_norm_96, parameter_507, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__237 = paddle._C_ops.add(matmul_88, parameter_508)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_27 = paddle._C_ops.gelu(add__237, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_89 = paddle.matmul(gelu_27, parameter_509, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__238 = paddle._C_ops.add(matmul_89, parameter_510)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_29 = parameter_511 * add__238

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__239 = paddle._C_ops.add(add__236, multiply_29)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_80 = paddle._C_ops.transpose(add__239, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_47 = [slice_73, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_80, combine_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_41 = paddle._C_ops.depthwise_conv2d(reshape__94, parameter_512, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__240 = paddle._C_ops.add(depthwise_conv2d_41, parameter_513)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__241 = paddle._C_ops.add(reshape__94, add__240)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_33 = paddle._C_ops.shape(paddle.cast(add__241, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(shape_33, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__36, flatten__37 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__241, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_81 = paddle._C_ops.transpose(flatten__36, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_81, parameter_514, parameter_515, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_34 = paddle._C_ops.shape(paddle.cast(layer_norm_99, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(shape_34, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_90 = paddle.matmul(layer_norm_99, parameter_516, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__242 = paddle._C_ops.add(matmul_90, parameter_517)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_48 = [slice_79, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__242, combine_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_82 = paddle._C_ops.transpose(reshape__96, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(transpose_82, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(transpose_82, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(transpose_82, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_83 = paddle._C_ops.transpose(slice_81, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_91 = paddle.matmul(slice_80, transpose_83, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__15 = paddle._C_ops.scale(matmul_91, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__15 = paddle._C_ops.softmax(scale__15, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_92 = paddle.matmul(softmax__15, slice_82, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_84 = paddle._C_ops.transpose(matmul_92, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_49 = [slice_79, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_84, combine_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_93 = paddle.matmul(reshape__98, parameter_518, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__243 = paddle._C_ops.add(matmul_93, parameter_519)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_30 = parameter_520 * add__243

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__244 = paddle._C_ops.add(transpose_81, multiply_30)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__244, parameter_521, parameter_522, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_94 = paddle.matmul(layer_norm_102, parameter_523, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__245 = paddle._C_ops.add(matmul_94, parameter_524)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_28 = paddle._C_ops.gelu(add__245, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_95 = paddle.matmul(gelu_28, parameter_525, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__246 = paddle._C_ops.add(matmul_95, parameter_526)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_31 = parameter_527 * add__246

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__247 = paddle._C_ops.add(add__244, multiply_31)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_85 = paddle._C_ops.transpose(add__247, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_50 = [slice_78, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_85, combine_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_42 = paddle._C_ops.depthwise_conv2d(reshape__100, parameter_528, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__248 = paddle._C_ops.add(depthwise_conv2d_42, parameter_529)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__249 = paddle._C_ops.add(reshape__100, add__248)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_35 = paddle._C_ops.shape(paddle.cast(add__249, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(shape_35, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__38, flatten__39 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__249, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_86 = paddle._C_ops.transpose(flatten__38, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_86, parameter_530, parameter_531, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_36 = paddle._C_ops.shape(paddle.cast(layer_norm_105, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(shape_36, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_96 = paddle.matmul(layer_norm_105, parameter_532, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__250 = paddle._C_ops.add(matmul_96, parameter_533)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_51 = [slice_84, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__250, combine_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_87 = paddle._C_ops.transpose(reshape__102, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(transpose_87, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(transpose_87, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(transpose_87, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_88 = paddle._C_ops.transpose(slice_86, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_97 = paddle.matmul(slice_85, transpose_88, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__16 = paddle._C_ops.scale(matmul_97, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__16 = paddle._C_ops.softmax(scale__16, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_98 = paddle.matmul(softmax__16, slice_87, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_89 = paddle._C_ops.transpose(matmul_98, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_52 = [slice_84, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_89, combine_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_99 = paddle.matmul(reshape__104, parameter_534, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__251 = paddle._C_ops.add(matmul_99, parameter_535)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_32 = parameter_536 * add__251

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__252 = paddle._C_ops.add(transpose_86, multiply_32)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__252, parameter_537, parameter_538, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_100 = paddle.matmul(layer_norm_108, parameter_539, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__253 = paddle._C_ops.add(matmul_100, parameter_540)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_29 = paddle._C_ops.gelu(add__253, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_101 = paddle.matmul(gelu_29, parameter_541, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__254 = paddle._C_ops.add(matmul_101, parameter_542)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_33 = parameter_543 * add__254

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__255 = paddle._C_ops.add(add__252, multiply_33)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_90 = paddle._C_ops.transpose(add__255, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_53 = [slice_83, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_90, combine_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_43 = paddle._C_ops.depthwise_conv2d(reshape__106, parameter_544, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__256 = paddle._C_ops.add(depthwise_conv2d_43, parameter_545)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__257 = paddle._C_ops.add(reshape__106, add__256)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_37 = paddle._C_ops.shape(paddle.cast(add__257, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(shape_37, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__40, flatten__41 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__257, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_91 = paddle._C_ops.transpose(flatten__40, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_91, parameter_546, parameter_547, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_38 = paddle._C_ops.shape(paddle.cast(layer_norm_111, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(shape_38, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_102 = paddle.matmul(layer_norm_111, parameter_548, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__258 = paddle._C_ops.add(matmul_102, parameter_549)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_54 = [slice_89, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__258, combine_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_92 = paddle._C_ops.transpose(reshape__108, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(transpose_92, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(transpose_92, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(transpose_92, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_93 = paddle._C_ops.transpose(slice_91, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_103 = paddle.matmul(slice_90, transpose_93, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__17 = paddle._C_ops.scale(matmul_103, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__17 = paddle._C_ops.softmax(scale__17, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_104 = paddle.matmul(softmax__17, slice_92, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_94 = paddle._C_ops.transpose(matmul_104, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_55 = [slice_89, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_94, combine_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_105 = paddle.matmul(reshape__110, parameter_550, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__259 = paddle._C_ops.add(matmul_105, parameter_551)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_34 = parameter_552 * add__259

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__260 = paddle._C_ops.add(transpose_91, multiply_34)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__260, parameter_553, parameter_554, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_106 = paddle.matmul(layer_norm_114, parameter_555, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__261 = paddle._C_ops.add(matmul_106, parameter_556)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_30 = paddle._C_ops.gelu(add__261, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_107 = paddle.matmul(gelu_30, parameter_557, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__262 = paddle._C_ops.add(matmul_107, parameter_558)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_35 = parameter_559 * add__262

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__263 = paddle._C_ops.add(add__260, multiply_35)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_95 = paddle._C_ops.transpose(add__263, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_56 = [slice_88, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_95, combine_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_44 = paddle._C_ops.depthwise_conv2d(reshape__112, parameter_560, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__264 = paddle._C_ops.add(depthwise_conv2d_44, parameter_561)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__265 = paddle._C_ops.add(reshape__112, add__264)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_39 = paddle._C_ops.shape(paddle.cast(add__265, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(shape_39, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__42, flatten__43 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__265, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_96 = paddle._C_ops.transpose(flatten__42, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_96, parameter_562, parameter_563, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_40 = paddle._C_ops.shape(paddle.cast(layer_norm_117, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(shape_40, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_108 = paddle.matmul(layer_norm_117, parameter_564, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__266 = paddle._C_ops.add(matmul_108, parameter_565)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_57 = [slice_94, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__266, combine_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_97 = paddle._C_ops.transpose(reshape__114, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(transpose_97, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(transpose_97, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(transpose_97, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_98 = paddle._C_ops.transpose(slice_96, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_109 = paddle.matmul(slice_95, transpose_98, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__18 = paddle._C_ops.scale(matmul_109, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__18 = paddle._C_ops.softmax(scale__18, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_110 = paddle.matmul(softmax__18, slice_97, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_99 = paddle._C_ops.transpose(matmul_110, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_58 = [slice_94, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_99, combine_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_111 = paddle.matmul(reshape__116, parameter_566, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__267 = paddle._C_ops.add(matmul_111, parameter_567)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_36 = parameter_568 * add__267

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__268 = paddle._C_ops.add(transpose_96, multiply_36)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__268, parameter_569, parameter_570, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_112 = paddle.matmul(layer_norm_120, parameter_571, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__269 = paddle._C_ops.add(matmul_112, parameter_572)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_31 = paddle._C_ops.gelu(add__269, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_113 = paddle.matmul(gelu_31, parameter_573, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__270 = paddle._C_ops.add(matmul_113, parameter_574)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_37 = parameter_575 * add__270

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__271 = paddle._C_ops.add(add__268, multiply_37)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_100 = paddle._C_ops.transpose(add__271, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_59 = [slice_93, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_100, combine_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_45 = paddle._C_ops.depthwise_conv2d(reshape__118, parameter_576, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__272 = paddle._C_ops.add(depthwise_conv2d_45, parameter_577)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__273 = paddle._C_ops.add(reshape__118, add__272)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf16)
        shape_41 = paddle._C_ops.shape(paddle.cast(add__273, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(shape_41, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__44, flatten__45 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__273, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_101 = paddle._C_ops.transpose(flatten__44, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_101, parameter_578, parameter_579, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_42 = paddle._C_ops.shape(paddle.cast(layer_norm_123, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(shape_42, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf16) <- (-1x196x320xf16, 320x960xf16)
        matmul_114 = paddle.matmul(layer_norm_123, parameter_580, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf16) <- (-1x196x960xf16, 960xf16)
        add__274 = paddle._C_ops.add(matmul_114, parameter_581)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_60 = [slice_99, constant_8, constant_9, constant_10, constant_3]

        # pd_op.reshape_: (-1x196x3x5x64xf16, 0x-1x196x960xf16) <- (-1x196x960xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__274, combine_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf16) <- (-1x196x3x5x64xf16)
        transpose_102 = paddle._C_ops.transpose(reshape__120, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_100 = paddle._C_ops.slice(transpose_102, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_101 = paddle._C_ops.slice(transpose_102, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x5x196x64xf16) <- (3x-1x5x196x64xf16, 1xi64, 1xi64)
        slice_102 = paddle._C_ops.slice(transpose_102, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf16) <- (-1x5x196x64xf16)
        transpose_103 = paddle._C_ops.transpose(slice_101, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf16) <- (-1x5x196x64xf16, -1x5x64x196xf16)
        matmul_115 = paddle.matmul(slice_100, transpose_103, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf16) <- (-1x5x196x196xf16, 1xf32)
        scale__19 = paddle._C_ops.scale(matmul_115, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf16) <- (-1x5x196x196xf16)
        softmax__19 = paddle._C_ops.softmax(scale__19, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x196xf16, -1x5x196x64xf16)
        matmul_116 = paddle.matmul(softmax__19, slice_102, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_104 = paddle._C_ops.transpose(matmul_116, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_61 = [slice_99, constant_8, constant_7]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_104, combine_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_117 = paddle.matmul(reshape__122, parameter_582, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__275 = paddle._C_ops.add(matmul_117, parameter_583)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_38 = parameter_584 * add__275

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__276 = paddle._C_ops.add(transpose_101, multiply_38)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__276, parameter_585, parameter_586, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_118 = paddle.matmul(layer_norm_126, parameter_587, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__277 = paddle._C_ops.add(matmul_118, parameter_588)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_32 = paddle._C_ops.gelu(add__277, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_119 = paddle.matmul(gelu_32, parameter_589, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__278 = paddle._C_ops.add(matmul_119, parameter_590)

        # pd_op.multiply: (-1x196x320xf16) <- (320xf16, -1x196x320xf16)
        multiply_39 = parameter_591 * add__278

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__279 = paddle._C_ops.add(add__276, multiply_39)

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_105 = paddle._C_ops.transpose(add__279, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_62 = [slice_98, constant_7, constant_6, constant_6]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_105, combine_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x512x7x7xf16) <- (-1x320x14x14xf16, 512x320x2x2xf16)
        conv2d_55 = paddle._C_ops.conv2d(reshape__124, parameter_592, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__280 = paddle._C_ops.add(conv2d_55, parameter_593)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf16)
        shape_43 = paddle._C_ops.shape(paddle.cast(add__280, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_103 = paddle._C_ops.slice(shape_43, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__46, flatten__47 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__280, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_106 = paddle._C_ops.transpose(flatten__46, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_106, parameter_594, parameter_595, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_63 = [slice_103, constant_14, constant_14, constant_15]

        # pd_op.reshape_: (-1x7x7x512xf16, 0x-1x49x512xf16) <- (-1x49x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape(layer_norm_129, combine_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x512x7x7xf16) <- (-1x7x7x512xf16)
        transpose_107 = paddle._C_ops.transpose(reshape__126, [0, 3, 1, 2])

        # pd_op.depthwise_conv2d: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 512x1x3x3xf16)
        depthwise_conv2d_46 = paddle._C_ops.depthwise_conv2d(transpose_107, parameter_596, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__281 = paddle._C_ops.add(depthwise_conv2d_46, parameter_597)

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__282 = paddle._C_ops.add(transpose_107, add__281)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf16)
        shape_44 = paddle._C_ops.shape(paddle.cast(add__282, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(shape_44, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__48, flatten__49 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__282, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_108 = paddle._C_ops.transpose(flatten__48, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_108, parameter_598, parameter_599, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_45 = paddle._C_ops.shape(paddle.cast(layer_norm_132, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(shape_45, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf16) <- (-1x49x512xf16, 512x1536xf16)
        matmul_120 = paddle.matmul(layer_norm_132, parameter_600, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf16) <- (-1x49x1536xf16, 1536xf16)
        add__283 = paddle._C_ops.add(matmul_120, parameter_601)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_64 = [slice_105, constant_16, constant_9, constant_17, constant_3]

        # pd_op.reshape_: (-1x49x3x8x64xf16, 0x-1x49x1536xf16) <- (-1x49x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__283, combine_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf16) <- (-1x49x3x8x64xf16)
        transpose_109 = paddle._C_ops.transpose(reshape__128, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(transpose_109, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(transpose_109, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_108 = paddle._C_ops.slice(transpose_109, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_110 = paddle._C_ops.transpose(slice_107, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_121 = paddle.matmul(slice_106, transpose_110, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__20 = paddle._C_ops.scale(matmul_121, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__20 = paddle._C_ops.softmax(scale__20, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_122 = paddle.matmul(softmax__20, slice_108, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_111 = paddle._C_ops.transpose(matmul_122, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_65 = [slice_105, constant_16, constant_15]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_111, combine_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_123 = paddle.matmul(reshape__130, parameter_602, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__284 = paddle._C_ops.add(matmul_123, parameter_603)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_40 = parameter_604 * add__284

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__285 = paddle._C_ops.add(transpose_108, multiply_40)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__285, parameter_605, parameter_606, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_124 = paddle.matmul(layer_norm_135, parameter_607, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__286 = paddle._C_ops.add(matmul_124, parameter_608)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_33 = paddle._C_ops.gelu(add__286, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_125 = paddle.matmul(gelu_33, parameter_609, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__287 = paddle._C_ops.add(matmul_125, parameter_610)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_41 = parameter_611 * add__287

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__288 = paddle._C_ops.add(add__285, multiply_41)

        # pd_op.transpose: (-1x512x49xf16) <- (-1x49x512xf16)
        transpose_112 = paddle._C_ops.transpose(add__288, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_66 = [slice_104, constant_15, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf16, 0x-1x512x49xf16) <- (-1x512x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_112, combine_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 512x1x3x3xf16)
        depthwise_conv2d_47 = paddle._C_ops.depthwise_conv2d(reshape__132, parameter_612, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__289 = paddle._C_ops.add(depthwise_conv2d_47, parameter_613)

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__290 = paddle._C_ops.add(reshape__132, add__289)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf16)
        shape_46 = paddle._C_ops.shape(paddle.cast(add__290, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_109 = paddle._C_ops.slice(shape_46, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__50, flatten__51 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__290, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_113 = paddle._C_ops.transpose(flatten__50, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_113, parameter_614, parameter_615, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_47 = paddle._C_ops.shape(paddle.cast(layer_norm_138, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_110 = paddle._C_ops.slice(shape_47, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf16) <- (-1x49x512xf16, 512x1536xf16)
        matmul_126 = paddle.matmul(layer_norm_138, parameter_616, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf16) <- (-1x49x1536xf16, 1536xf16)
        add__291 = paddle._C_ops.add(matmul_126, parameter_617)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_67 = [slice_110, constant_16, constant_9, constant_17, constant_3]

        # pd_op.reshape_: (-1x49x3x8x64xf16, 0x-1x49x1536xf16) <- (-1x49x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__291, combine_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf16) <- (-1x49x3x8x64xf16)
        transpose_114 = paddle._C_ops.transpose(reshape__134, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_111 = paddle._C_ops.slice(transpose_114, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(transpose_114, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_113 = paddle._C_ops.slice(transpose_114, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_115 = paddle._C_ops.transpose(slice_112, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_127 = paddle.matmul(slice_111, transpose_115, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__21 = paddle._C_ops.scale(matmul_127, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__21 = paddle._C_ops.softmax(scale__21, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_128 = paddle.matmul(softmax__21, slice_113, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_116 = paddle._C_ops.transpose(matmul_128, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_68 = [slice_110, constant_16, constant_15]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_116, combine_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_129 = paddle.matmul(reshape__136, parameter_618, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__292 = paddle._C_ops.add(matmul_129, parameter_619)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_42 = parameter_620 * add__292

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__293 = paddle._C_ops.add(transpose_113, multiply_42)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__293, parameter_621, parameter_622, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_130 = paddle.matmul(layer_norm_141, parameter_623, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__294 = paddle._C_ops.add(matmul_130, parameter_624)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_34 = paddle._C_ops.gelu(add__294, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_131 = paddle.matmul(gelu_34, parameter_625, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__295 = paddle._C_ops.add(matmul_131, parameter_626)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_43 = parameter_627 * add__295

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__296 = paddle._C_ops.add(add__293, multiply_43)

        # pd_op.transpose: (-1x512x49xf16) <- (-1x49x512xf16)
        transpose_117 = paddle._C_ops.transpose(add__296, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_69 = [slice_109, constant_15, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf16, 0x-1x512x49xf16) <- (-1x512x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_117, combine_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 512x1x3x3xf16)
        depthwise_conv2d_48 = paddle._C_ops.depthwise_conv2d(reshape__138, parameter_628, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__297 = paddle._C_ops.add(depthwise_conv2d_48, parameter_629)

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__298 = paddle._C_ops.add(reshape__138, add__297)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf16)
        shape_48 = paddle._C_ops.shape(paddle.cast(add__298, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_114 = paddle._C_ops.slice(shape_48, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__52, flatten__53 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__298, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_118 = paddle._C_ops.transpose(flatten__52, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_118, parameter_630, parameter_631, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_49 = paddle._C_ops.shape(paddle.cast(layer_norm_144, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_115 = paddle._C_ops.slice(shape_49, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf16) <- (-1x49x512xf16, 512x1536xf16)
        matmul_132 = paddle.matmul(layer_norm_144, parameter_632, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf16) <- (-1x49x1536xf16, 1536xf16)
        add__299 = paddle._C_ops.add(matmul_132, parameter_633)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_70 = [slice_115, constant_16, constant_9, constant_17, constant_3]

        # pd_op.reshape_: (-1x49x3x8x64xf16, 0x-1x49x1536xf16) <- (-1x49x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__299, combine_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf16) <- (-1x49x3x8x64xf16)
        transpose_119 = paddle._C_ops.transpose(reshape__140, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_116 = paddle._C_ops.slice(transpose_119, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_117 = paddle._C_ops.slice(transpose_119, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_118 = paddle._C_ops.slice(transpose_119, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_120 = paddle._C_ops.transpose(slice_117, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_133 = paddle.matmul(slice_116, transpose_120, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__22 = paddle._C_ops.scale(matmul_133, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__22 = paddle._C_ops.softmax(scale__22, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_134 = paddle.matmul(softmax__22, slice_118, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_121 = paddle._C_ops.transpose(matmul_134, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_71 = [slice_115, constant_16, constant_15]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_121, combine_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_135 = paddle.matmul(reshape__142, parameter_634, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__300 = paddle._C_ops.add(matmul_135, parameter_635)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_44 = parameter_636 * add__300

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__301 = paddle._C_ops.add(transpose_118, multiply_44)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_147, layer_norm_148, layer_norm_149 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__301, parameter_637, parameter_638, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_136 = paddle.matmul(layer_norm_147, parameter_639, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__302 = paddle._C_ops.add(matmul_136, parameter_640)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_35 = paddle._C_ops.gelu(add__302, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_137 = paddle.matmul(gelu_35, parameter_641, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__303 = paddle._C_ops.add(matmul_137, parameter_642)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_45 = parameter_643 * add__303

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__304 = paddle._C_ops.add(add__301, multiply_45)

        # pd_op.transpose: (-1x512x49xf16) <- (-1x49x512xf16)
        transpose_122 = paddle._C_ops.transpose(add__304, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_72 = [slice_114, constant_15, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf16, 0x-1x512x49xf16) <- (-1x512x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_122, combine_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 512x1x3x3xf16)
        depthwise_conv2d_49 = paddle._C_ops.depthwise_conv2d(reshape__144, parameter_644, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__305 = paddle._C_ops.add(depthwise_conv2d_49, parameter_645)

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__306 = paddle._C_ops.add(reshape__144, add__305)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf16)
        shape_50 = paddle._C_ops.shape(paddle.cast(add__306, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(shape_50, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__54, flatten__55 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__306, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_123 = paddle._C_ops.transpose(flatten__54, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_150, layer_norm_151, layer_norm_152 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_123, parameter_646, parameter_647, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_51 = paddle._C_ops.shape(paddle.cast(layer_norm_150, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_120 = paddle._C_ops.slice(shape_51, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf16) <- (-1x49x512xf16, 512x1536xf16)
        matmul_138 = paddle.matmul(layer_norm_150, parameter_648, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf16) <- (-1x49x1536xf16, 1536xf16)
        add__307 = paddle._C_ops.add(matmul_138, parameter_649)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_73 = [slice_120, constant_16, constant_9, constant_17, constant_3]

        # pd_op.reshape_: (-1x49x3x8x64xf16, 0x-1x49x1536xf16) <- (-1x49x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__307, combine_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf16) <- (-1x49x3x8x64xf16)
        transpose_124 = paddle._C_ops.transpose(reshape__146, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_121 = paddle._C_ops.slice(transpose_124, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(transpose_124, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_123 = paddle._C_ops.slice(transpose_124, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_125 = paddle._C_ops.transpose(slice_122, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_139 = paddle.matmul(slice_121, transpose_125, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__23 = paddle._C_ops.scale(matmul_139, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__23 = paddle._C_ops.softmax(scale__23, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_140 = paddle.matmul(softmax__23, slice_123, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_126 = paddle._C_ops.transpose(matmul_140, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_74 = [slice_120, constant_16, constant_15]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_126, combine_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_141 = paddle.matmul(reshape__148, parameter_650, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__308 = paddle._C_ops.add(matmul_141, parameter_651)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_46 = parameter_652 * add__308

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__309 = paddle._C_ops.add(transpose_123, multiply_46)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_153, layer_norm_154, layer_norm_155 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__309, parameter_653, parameter_654, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_142 = paddle.matmul(layer_norm_153, parameter_655, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__310 = paddle._C_ops.add(matmul_142, parameter_656)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_36 = paddle._C_ops.gelu(add__310, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_143 = paddle.matmul(gelu_36, parameter_657, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__311 = paddle._C_ops.add(matmul_143, parameter_658)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_47 = parameter_659 * add__311

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__312 = paddle._C_ops.add(add__309, multiply_47)

        # pd_op.transpose: (-1x512x49xf16) <- (-1x49x512xf16)
        transpose_127 = paddle._C_ops.transpose(add__312, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_75 = [slice_119, constant_15, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf16, 0x-1x512x49xf16) <- (-1x512x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_127, combine_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 512x1x3x3xf16)
        depthwise_conv2d_50 = paddle._C_ops.depthwise_conv2d(reshape__150, parameter_660, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__313 = paddle._C_ops.add(depthwise_conv2d_50, parameter_661)

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__314 = paddle._C_ops.add(reshape__150, add__313)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf16)
        shape_52 = paddle._C_ops.shape(paddle.cast(add__314, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_124 = paddle._C_ops.slice(shape_52, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__56, flatten__57 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__314, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_128 = paddle._C_ops.transpose(flatten__56, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_156, layer_norm_157, layer_norm_158 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_128, parameter_662, parameter_663, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_53 = paddle._C_ops.shape(paddle.cast(layer_norm_156, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_125 = paddle._C_ops.slice(shape_53, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf16) <- (-1x49x512xf16, 512x1536xf16)
        matmul_144 = paddle.matmul(layer_norm_156, parameter_664, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf16) <- (-1x49x1536xf16, 1536xf16)
        add__315 = paddle._C_ops.add(matmul_144, parameter_665)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_76 = [slice_125, constant_16, constant_9, constant_17, constant_3]

        # pd_op.reshape_: (-1x49x3x8x64xf16, 0x-1x49x1536xf16) <- (-1x49x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__315, combine_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf16) <- (-1x49x3x8x64xf16)
        transpose_129 = paddle._C_ops.transpose(reshape__152, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_126 = paddle._C_ops.slice(transpose_129, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_127 = paddle._C_ops.slice(transpose_129, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_128 = paddle._C_ops.slice(transpose_129, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_130 = paddle._C_ops.transpose(slice_127, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_145 = paddle.matmul(slice_126, transpose_130, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__24 = paddle._C_ops.scale(matmul_145, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__24 = paddle._C_ops.softmax(scale__24, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_146 = paddle.matmul(softmax__24, slice_128, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_131 = paddle._C_ops.transpose(matmul_146, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_77 = [slice_125, constant_16, constant_15]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_131, combine_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_147 = paddle.matmul(reshape__154, parameter_666, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__316 = paddle._C_ops.add(matmul_147, parameter_667)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_48 = parameter_668 * add__316

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__317 = paddle._C_ops.add(transpose_128, multiply_48)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_159, layer_norm_160, layer_norm_161 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__317, parameter_669, parameter_670, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_148 = paddle.matmul(layer_norm_159, parameter_671, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__318 = paddle._C_ops.add(matmul_148, parameter_672)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_37 = paddle._C_ops.gelu(add__318, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_149 = paddle.matmul(gelu_37, parameter_673, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__319 = paddle._C_ops.add(matmul_149, parameter_674)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_49 = parameter_675 * add__319

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__320 = paddle._C_ops.add(add__317, multiply_49)

        # pd_op.transpose: (-1x512x49xf16) <- (-1x49x512xf16)
        transpose_132 = paddle._C_ops.transpose(add__320, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_78 = [slice_124, constant_15, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf16, 0x-1x512x49xf16) <- (-1x512x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_132, combine_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 512x1x3x3xf16)
        depthwise_conv2d_51 = paddle._C_ops.depthwise_conv2d(reshape__156, parameter_676, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__321 = paddle._C_ops.add(depthwise_conv2d_51, parameter_677)

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__322 = paddle._C_ops.add(reshape__156, add__321)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf16)
        shape_54 = paddle._C_ops.shape(paddle.cast(add__322, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_129 = paddle._C_ops.slice(shape_54, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__58, flatten__59 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__322, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_133 = paddle._C_ops.transpose(flatten__58, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_162, layer_norm_163, layer_norm_164 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_133, parameter_678, parameter_679, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_55 = paddle._C_ops.shape(paddle.cast(layer_norm_162, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_130 = paddle._C_ops.slice(shape_55, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf16) <- (-1x49x512xf16, 512x1536xf16)
        matmul_150 = paddle.matmul(layer_norm_162, parameter_680, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf16) <- (-1x49x1536xf16, 1536xf16)
        add__323 = paddle._C_ops.add(matmul_150, parameter_681)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_79 = [slice_130, constant_16, constant_9, constant_17, constant_3]

        # pd_op.reshape_: (-1x49x3x8x64xf16, 0x-1x49x1536xf16) <- (-1x49x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__323, combine_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf16) <- (-1x49x3x8x64xf16)
        transpose_134 = paddle._C_ops.transpose(reshape__158, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_131 = paddle._C_ops.slice(transpose_134, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_132 = paddle._C_ops.slice(transpose_134, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_133 = paddle._C_ops.slice(transpose_134, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_135 = paddle._C_ops.transpose(slice_132, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_151 = paddle.matmul(slice_131, transpose_135, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__25 = paddle._C_ops.scale(matmul_151, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__25 = paddle._C_ops.softmax(scale__25, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_152 = paddle.matmul(softmax__25, slice_133, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_136 = paddle._C_ops.transpose(matmul_152, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_80 = [slice_130, constant_16, constant_15]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_136, combine_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_153 = paddle.matmul(reshape__160, parameter_682, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__324 = paddle._C_ops.add(matmul_153, parameter_683)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_50 = parameter_684 * add__324

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__325 = paddle._C_ops.add(transpose_133, multiply_50)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_165, layer_norm_166, layer_norm_167 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__325, parameter_685, parameter_686, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_154 = paddle.matmul(layer_norm_165, parameter_687, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__326 = paddle._C_ops.add(matmul_154, parameter_688)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_38 = paddle._C_ops.gelu(add__326, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_155 = paddle.matmul(gelu_38, parameter_689, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__327 = paddle._C_ops.add(matmul_155, parameter_690)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_51 = parameter_691 * add__327

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__328 = paddle._C_ops.add(add__325, multiply_51)

        # pd_op.transpose: (-1x512x49xf16) <- (-1x49x512xf16)
        transpose_137 = paddle._C_ops.transpose(add__328, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_81 = [slice_129, constant_15, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf16, 0x-1x512x49xf16) <- (-1x512x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_137, combine_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 512x1x3x3xf16)
        depthwise_conv2d_52 = paddle._C_ops.depthwise_conv2d(reshape__162, parameter_692, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__329 = paddle._C_ops.add(depthwise_conv2d_52, parameter_693)

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__330 = paddle._C_ops.add(reshape__162, add__329)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf16)
        shape_56 = paddle._C_ops.shape(paddle.cast(add__330, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(shape_56, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__60, flatten__61 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__330, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_138 = paddle._C_ops.transpose(flatten__60, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_168, layer_norm_169, layer_norm_170 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_138, parameter_694, parameter_695, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_57 = paddle._C_ops.shape(paddle.cast(layer_norm_168, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_135 = paddle._C_ops.slice(shape_57, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf16) <- (-1x49x512xf16, 512x1536xf16)
        matmul_156 = paddle.matmul(layer_norm_168, parameter_696, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf16) <- (-1x49x1536xf16, 1536xf16)
        add__331 = paddle._C_ops.add(matmul_156, parameter_697)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_82 = [slice_135, constant_16, constant_9, constant_17, constant_3]

        # pd_op.reshape_: (-1x49x3x8x64xf16, 0x-1x49x1536xf16) <- (-1x49x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__331, combine_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf16) <- (-1x49x3x8x64xf16)
        transpose_139 = paddle._C_ops.transpose(reshape__164, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_136 = paddle._C_ops.slice(transpose_139, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_137 = paddle._C_ops.slice(transpose_139, [0], constant_1, constant_11, [1], [0])

        # pd_op.slice: (-1x8x49x64xf16) <- (3x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_138 = paddle._C_ops.slice(transpose_139, [0], constant_11, constant_12, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_140 = paddle._C_ops.transpose(slice_137, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_157 = paddle.matmul(slice_136, transpose_140, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__26 = paddle._C_ops.scale(matmul_157, constant_13, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__26 = paddle._C_ops.softmax(scale__26, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_158 = paddle.matmul(softmax__26, slice_138, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_141 = paddle._C_ops.transpose(matmul_158, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_83 = [slice_135, constant_16, constant_15]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_141, combine_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_159 = paddle.matmul(reshape__166, parameter_698, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__332 = paddle._C_ops.add(matmul_159, parameter_699)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_52 = parameter_700 * add__332

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__333 = paddle._C_ops.add(transpose_138, multiply_52)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_171, layer_norm_172, layer_norm_173 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__333, parameter_701, parameter_702, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_160 = paddle.matmul(layer_norm_171, parameter_703, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__334 = paddle._C_ops.add(matmul_160, parameter_704)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_39 = paddle._C_ops.gelu(add__334, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_161 = paddle.matmul(gelu_39, parameter_705, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__335 = paddle._C_ops.add(matmul_161, parameter_706)

        # pd_op.multiply: (-1x49x512xf16) <- (512xf16, -1x49x512xf16)
        multiply_53 = parameter_707 * add__335

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__336 = paddle._C_ops.add(add__333, multiply_53)

        # pd_op.transpose: (-1x512x49xf16) <- (-1x49x512xf16)
        transpose_142 = paddle._C_ops.transpose(add__336, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_84 = [slice_134, constant_15, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf16, 0x-1x512x49xf16) <- (-1x512x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_142, combine_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512x7x7xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x7x7xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__168, parameter_708, parameter_709, parameter_710, parameter_711, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__62, flatten__63 = (lambda x, f: f(x))(paddle._C_ops.flatten(batch_norm__156, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.mean: (-1x512xf16) <- (-1x512x49xf16)
        mean_0 = paddle._C_ops.mean(flatten__62, [-1], False)

        # pd_op.matmul: (-1x1000xf16) <- (-1x512xf16, 512x1000xf16)
        matmul_162 = paddle.matmul(mean_0, parameter_712, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__337 = paddle._C_ops.add(matmul_162, parameter_713)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__27 = paddle._C_ops.softmax(add__337, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__27, paddle.float32)
        return cast_1



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


class CinnTestBase:
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def _test_entry(self):
        dy_outs = self.entry(use_cinn=False)
        cinn_outs = self.entry(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

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

class ModuleOp(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_693, parameter_677, parameter_661, parameter_645, parameter_629, parameter_613, constant_17, constant_16, parameter_597, constant_15, constant_14, parameter_593, parameter_577, parameter_561, parameter_545, parameter_529, parameter_513, parameter_497, parameter_481, parameter_465, parameter_449, parameter_433, parameter_417, parameter_401, parameter_385, parameter_369, parameter_353, parameter_337, parameter_321, parameter_305, parameter_289, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, parameter_273, constant_7, constant_6, parameter_269, parameter_267, parameter_265, parameter_259, parameter_257, parameter_255, parameter_249, parameter_247, parameter_245, parameter_239, parameter_237, parameter_235, parameter_229, parameter_227, parameter_225, parameter_219, parameter_217, parameter_215, parameter_209, parameter_207, parameter_205, parameter_199, parameter_197, parameter_195, parameter_189, parameter_187, parameter_185, parameter_179, parameter_177, parameter_175, parameter_169, parameter_167, parameter_165, parameter_159, parameter_157, parameter_155, parameter_149, parameter_147, parameter_145, parameter_139, parameter_137, parameter_135, parameter_129, parameter_127, parameter_125, parameter_119, parameter_117, parameter_115, parameter_109, constant_5, constant_4, parameter_105, parameter_103, parameter_101, parameter_95, parameter_93, parameter_91, parameter_85, parameter_83, parameter_81, parameter_75, parameter_73, parameter_71, parameter_65, parameter_63, parameter_61, parameter_55, parameter_53, parameter_51, parameter_45, parameter_43, parameter_41, parameter_35, parameter_33, parameter_31, parameter_25, parameter_23, parameter_21, parameter_15, parameter_13, parameter_11, parameter_5, constant_3, constant_2, constant_1, constant_0, parameter_1, parameter_0, parameter_3, parameter_2, parameter_4, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_12, parameter_14, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_22, parameter_24, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_32, parameter_34, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_42, parameter_44, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_52, parameter_54, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_62, parameter_64, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_72, parameter_74, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_82, parameter_84, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_92, parameter_94, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_102, parameter_104, parameter_107, parameter_106, parameter_108, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_116, parameter_118, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_126, parameter_128, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_136, parameter_138, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_146, parameter_148, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_156, parameter_158, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_166, parameter_168, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_176, parameter_178, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_186, parameter_188, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_196, parameter_198, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_206, parameter_208, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_216, parameter_218, parameter_223, parameter_220, parameter_222, parameter_221, parameter_224, parameter_226, parameter_228, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_236, parameter_238, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_246, parameter_248, parameter_253, parameter_250, parameter_252, parameter_251, parameter_254, parameter_256, parameter_258, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_266, parameter_268, parameter_271, parameter_270, parameter_272, parameter_275, parameter_274, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_282, parameter_281, parameter_283, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_291, parameter_290, parameter_292, parameter_293, parameter_294, parameter_295, parameter_296, parameter_298, parameter_297, parameter_299, parameter_300, parameter_301, parameter_302, parameter_303, parameter_304, parameter_307, parameter_306, parameter_308, parameter_309, parameter_310, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_323, parameter_322, parameter_324, parameter_325, parameter_326, parameter_327, parameter_328, parameter_330, parameter_329, parameter_331, parameter_332, parameter_333, parameter_334, parameter_335, parameter_336, parameter_339, parameter_338, parameter_340, parameter_341, parameter_342, parameter_343, parameter_344, parameter_346, parameter_345, parameter_347, parameter_348, parameter_349, parameter_350, parameter_351, parameter_352, parameter_355, parameter_354, parameter_356, parameter_357, parameter_358, parameter_359, parameter_360, parameter_362, parameter_361, parameter_363, parameter_364, parameter_365, parameter_366, parameter_367, parameter_368, parameter_371, parameter_370, parameter_372, parameter_373, parameter_374, parameter_375, parameter_376, parameter_378, parameter_377, parameter_379, parameter_380, parameter_381, parameter_382, parameter_383, parameter_384, parameter_387, parameter_386, parameter_388, parameter_389, parameter_390, parameter_391, parameter_392, parameter_394, parameter_393, parameter_395, parameter_396, parameter_397, parameter_398, parameter_399, parameter_400, parameter_403, parameter_402, parameter_404, parameter_405, parameter_406, parameter_407, parameter_408, parameter_410, parameter_409, parameter_411, parameter_412, parameter_413, parameter_414, parameter_415, parameter_416, parameter_419, parameter_418, parameter_420, parameter_421, parameter_422, parameter_423, parameter_424, parameter_426, parameter_425, parameter_427, parameter_428, parameter_429, parameter_430, parameter_431, parameter_432, parameter_435, parameter_434, parameter_436, parameter_437, parameter_438, parameter_439, parameter_440, parameter_442, parameter_441, parameter_443, parameter_444, parameter_445, parameter_446, parameter_447, parameter_448, parameter_451, parameter_450, parameter_452, parameter_453, parameter_454, parameter_455, parameter_456, parameter_458, parameter_457, parameter_459, parameter_460, parameter_461, parameter_462, parameter_463, parameter_464, parameter_467, parameter_466, parameter_468, parameter_469, parameter_470, parameter_471, parameter_472, parameter_474, parameter_473, parameter_475, parameter_476, parameter_477, parameter_478, parameter_479, parameter_480, parameter_483, parameter_482, parameter_484, parameter_485, parameter_486, parameter_487, parameter_488, parameter_490, parameter_489, parameter_491, parameter_492, parameter_493, parameter_494, parameter_495, parameter_496, parameter_499, parameter_498, parameter_500, parameter_501, parameter_502, parameter_503, parameter_504, parameter_506, parameter_505, parameter_507, parameter_508, parameter_509, parameter_510, parameter_511, parameter_512, parameter_515, parameter_514, parameter_516, parameter_517, parameter_518, parameter_519, parameter_520, parameter_522, parameter_521, parameter_523, parameter_524, parameter_525, parameter_526, parameter_527, parameter_528, parameter_531, parameter_530, parameter_532, parameter_533, parameter_534, parameter_535, parameter_536, parameter_538, parameter_537, parameter_539, parameter_540, parameter_541, parameter_542, parameter_543, parameter_544, parameter_547, parameter_546, parameter_548, parameter_549, parameter_550, parameter_551, parameter_552, parameter_554, parameter_553, parameter_555, parameter_556, parameter_557, parameter_558, parameter_559, parameter_560, parameter_563, parameter_562, parameter_564, parameter_565, parameter_566, parameter_567, parameter_568, parameter_570, parameter_569, parameter_571, parameter_572, parameter_573, parameter_574, parameter_575, parameter_576, parameter_579, parameter_578, parameter_580, parameter_581, parameter_582, parameter_583, parameter_584, parameter_586, parameter_585, parameter_587, parameter_588, parameter_589, parameter_590, parameter_591, parameter_592, parameter_595, parameter_594, parameter_596, parameter_599, parameter_598, parameter_600, parameter_601, parameter_602, parameter_603, parameter_604, parameter_606, parameter_605, parameter_607, parameter_608, parameter_609, parameter_610, parameter_611, parameter_612, parameter_615, parameter_614, parameter_616, parameter_617, parameter_618, parameter_619, parameter_620, parameter_622, parameter_621, parameter_623, parameter_624, parameter_625, parameter_626, parameter_627, parameter_628, parameter_631, parameter_630, parameter_632, parameter_633, parameter_634, parameter_635, parameter_636, parameter_638, parameter_637, parameter_639, parameter_640, parameter_641, parameter_642, parameter_643, parameter_644, parameter_647, parameter_646, parameter_648, parameter_649, parameter_650, parameter_651, parameter_652, parameter_654, parameter_653, parameter_655, parameter_656, parameter_657, parameter_658, parameter_659, parameter_660, parameter_663, parameter_662, parameter_664, parameter_665, parameter_666, parameter_667, parameter_668, parameter_670, parameter_669, parameter_671, parameter_672, parameter_673, parameter_674, parameter_675, parameter_676, parameter_679, parameter_678, parameter_680, parameter_681, parameter_682, parameter_683, parameter_684, parameter_686, parameter_685, parameter_687, parameter_688, parameter_689, parameter_690, parameter_691, parameter_692, parameter_695, parameter_694, parameter_696, parameter_697, parameter_698, parameter_699, parameter_700, parameter_702, parameter_701, parameter_703, parameter_704, parameter_705, parameter_706, parameter_707, parameter_711, parameter_708, parameter_710, parameter_709, parameter_712, parameter_713, feed_0):
        return self.builtin_module_4575_0_0(parameter_693, parameter_677, parameter_661, parameter_645, parameter_629, parameter_613, constant_17, constant_16, parameter_597, constant_15, constant_14, parameter_593, parameter_577, parameter_561, parameter_545, parameter_529, parameter_513, parameter_497, parameter_481, parameter_465, parameter_449, parameter_433, parameter_417, parameter_401, parameter_385, parameter_369, parameter_353, parameter_337, parameter_321, parameter_305, parameter_289, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, parameter_273, constant_7, constant_6, parameter_269, parameter_267, parameter_265, parameter_259, parameter_257, parameter_255, parameter_249, parameter_247, parameter_245, parameter_239, parameter_237, parameter_235, parameter_229, parameter_227, parameter_225, parameter_219, parameter_217, parameter_215, parameter_209, parameter_207, parameter_205, parameter_199, parameter_197, parameter_195, parameter_189, parameter_187, parameter_185, parameter_179, parameter_177, parameter_175, parameter_169, parameter_167, parameter_165, parameter_159, parameter_157, parameter_155, parameter_149, parameter_147, parameter_145, parameter_139, parameter_137, parameter_135, parameter_129, parameter_127, parameter_125, parameter_119, parameter_117, parameter_115, parameter_109, constant_5, constant_4, parameter_105, parameter_103, parameter_101, parameter_95, parameter_93, parameter_91, parameter_85, parameter_83, parameter_81, parameter_75, parameter_73, parameter_71, parameter_65, parameter_63, parameter_61, parameter_55, parameter_53, parameter_51, parameter_45, parameter_43, parameter_41, parameter_35, parameter_33, parameter_31, parameter_25, parameter_23, parameter_21, parameter_15, parameter_13, parameter_11, parameter_5, constant_3, constant_2, constant_1, constant_0, parameter_1, parameter_0, parameter_3, parameter_2, parameter_4, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_12, parameter_14, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_22, parameter_24, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_32, parameter_34, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_42, parameter_44, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_52, parameter_54, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_62, parameter_64, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_72, parameter_74, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_82, parameter_84, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_92, parameter_94, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_102, parameter_104, parameter_107, parameter_106, parameter_108, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_116, parameter_118, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_126, parameter_128, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_136, parameter_138, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_146, parameter_148, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_156, parameter_158, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_166, parameter_168, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_176, parameter_178, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_186, parameter_188, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_196, parameter_198, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_206, parameter_208, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_216, parameter_218, parameter_223, parameter_220, parameter_222, parameter_221, parameter_224, parameter_226, parameter_228, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_236, parameter_238, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_246, parameter_248, parameter_253, parameter_250, parameter_252, parameter_251, parameter_254, parameter_256, parameter_258, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_266, parameter_268, parameter_271, parameter_270, parameter_272, parameter_275, parameter_274, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_282, parameter_281, parameter_283, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_291, parameter_290, parameter_292, parameter_293, parameter_294, parameter_295, parameter_296, parameter_298, parameter_297, parameter_299, parameter_300, parameter_301, parameter_302, parameter_303, parameter_304, parameter_307, parameter_306, parameter_308, parameter_309, parameter_310, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_323, parameter_322, parameter_324, parameter_325, parameter_326, parameter_327, parameter_328, parameter_330, parameter_329, parameter_331, parameter_332, parameter_333, parameter_334, parameter_335, parameter_336, parameter_339, parameter_338, parameter_340, parameter_341, parameter_342, parameter_343, parameter_344, parameter_346, parameter_345, parameter_347, parameter_348, parameter_349, parameter_350, parameter_351, parameter_352, parameter_355, parameter_354, parameter_356, parameter_357, parameter_358, parameter_359, parameter_360, parameter_362, parameter_361, parameter_363, parameter_364, parameter_365, parameter_366, parameter_367, parameter_368, parameter_371, parameter_370, parameter_372, parameter_373, parameter_374, parameter_375, parameter_376, parameter_378, parameter_377, parameter_379, parameter_380, parameter_381, parameter_382, parameter_383, parameter_384, parameter_387, parameter_386, parameter_388, parameter_389, parameter_390, parameter_391, parameter_392, parameter_394, parameter_393, parameter_395, parameter_396, parameter_397, parameter_398, parameter_399, parameter_400, parameter_403, parameter_402, parameter_404, parameter_405, parameter_406, parameter_407, parameter_408, parameter_410, parameter_409, parameter_411, parameter_412, parameter_413, parameter_414, parameter_415, parameter_416, parameter_419, parameter_418, parameter_420, parameter_421, parameter_422, parameter_423, parameter_424, parameter_426, parameter_425, parameter_427, parameter_428, parameter_429, parameter_430, parameter_431, parameter_432, parameter_435, parameter_434, parameter_436, parameter_437, parameter_438, parameter_439, parameter_440, parameter_442, parameter_441, parameter_443, parameter_444, parameter_445, parameter_446, parameter_447, parameter_448, parameter_451, parameter_450, parameter_452, parameter_453, parameter_454, parameter_455, parameter_456, parameter_458, parameter_457, parameter_459, parameter_460, parameter_461, parameter_462, parameter_463, parameter_464, parameter_467, parameter_466, parameter_468, parameter_469, parameter_470, parameter_471, parameter_472, parameter_474, parameter_473, parameter_475, parameter_476, parameter_477, parameter_478, parameter_479, parameter_480, parameter_483, parameter_482, parameter_484, parameter_485, parameter_486, parameter_487, parameter_488, parameter_490, parameter_489, parameter_491, parameter_492, parameter_493, parameter_494, parameter_495, parameter_496, parameter_499, parameter_498, parameter_500, parameter_501, parameter_502, parameter_503, parameter_504, parameter_506, parameter_505, parameter_507, parameter_508, parameter_509, parameter_510, parameter_511, parameter_512, parameter_515, parameter_514, parameter_516, parameter_517, parameter_518, parameter_519, parameter_520, parameter_522, parameter_521, parameter_523, parameter_524, parameter_525, parameter_526, parameter_527, parameter_528, parameter_531, parameter_530, parameter_532, parameter_533, parameter_534, parameter_535, parameter_536, parameter_538, parameter_537, parameter_539, parameter_540, parameter_541, parameter_542, parameter_543, parameter_544, parameter_547, parameter_546, parameter_548, parameter_549, parameter_550, parameter_551, parameter_552, parameter_554, parameter_553, parameter_555, parameter_556, parameter_557, parameter_558, parameter_559, parameter_560, parameter_563, parameter_562, parameter_564, parameter_565, parameter_566, parameter_567, parameter_568, parameter_570, parameter_569, parameter_571, parameter_572, parameter_573, parameter_574, parameter_575, parameter_576, parameter_579, parameter_578, parameter_580, parameter_581, parameter_582, parameter_583, parameter_584, parameter_586, parameter_585, parameter_587, parameter_588, parameter_589, parameter_590, parameter_591, parameter_592, parameter_595, parameter_594, parameter_596, parameter_599, parameter_598, parameter_600, parameter_601, parameter_602, parameter_603, parameter_604, parameter_606, parameter_605, parameter_607, parameter_608, parameter_609, parameter_610, parameter_611, parameter_612, parameter_615, parameter_614, parameter_616, parameter_617, parameter_618, parameter_619, parameter_620, parameter_622, parameter_621, parameter_623, parameter_624, parameter_625, parameter_626, parameter_627, parameter_628, parameter_631, parameter_630, parameter_632, parameter_633, parameter_634, parameter_635, parameter_636, parameter_638, parameter_637, parameter_639, parameter_640, parameter_641, parameter_642, parameter_643, parameter_644, parameter_647, parameter_646, parameter_648, parameter_649, parameter_650, parameter_651, parameter_652, parameter_654, parameter_653, parameter_655, parameter_656, parameter_657, parameter_658, parameter_659, parameter_660, parameter_663, parameter_662, parameter_664, parameter_665, parameter_666, parameter_667, parameter_668, parameter_670, parameter_669, parameter_671, parameter_672, parameter_673, parameter_674, parameter_675, parameter_676, parameter_679, parameter_678, parameter_680, parameter_681, parameter_682, parameter_683, parameter_684, parameter_686, parameter_685, parameter_687, parameter_688, parameter_689, parameter_690, parameter_691, parameter_692, parameter_695, parameter_694, parameter_696, parameter_697, parameter_698, parameter_699, parameter_700, parameter_702, parameter_701, parameter_703, parameter_704, parameter_705, parameter_706, parameter_707, parameter_711, parameter_708, parameter_710, parameter_709, parameter_712, parameter_713, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_4575_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_693
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_677
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_661
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_645
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_629
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_613
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_17
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            # constant_16
            paddle.to_tensor([49], dtype='int32').reshape([1]),
            # parameter_597
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_15
            paddle.to_tensor([512], dtype='int32').reshape([1]),
            # constant_14
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            # parameter_593
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_577
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_561
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_545
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_529
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_513
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_497
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_481
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_465
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_449
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_433
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_417
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_401
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_385
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_369
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_353
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_13
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_12
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            # constant_11
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_10
            paddle.to_tensor([5], dtype='int32').reshape([1]),
            # constant_9
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            # constant_8
            paddle.to_tensor([196], dtype='int32').reshape([1]),
            # parameter_273
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_7
            paddle.to_tensor([320], dtype='int32').reshape([1]),
            # constant_6
            paddle.to_tensor([14], dtype='int32').reshape([1]),
            # parameter_269
            paddle.uniform([1, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_267
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_265
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_257
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_255
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_247
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_245
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_237
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_235
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_197
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_195
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_187
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_185
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_175
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_5
            paddle.to_tensor([128], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([28], dtype='int32').reshape([1]),
            # parameter_105
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_25
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_5
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_3
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_2
            paddle.to_tensor([56], dtype='int32').reshape([1]),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_1
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_0
            paddle.uniform([64, 3, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([64, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_82
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([64, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_92
            paddle.uniform([64, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([128, 64, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([128, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_168
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_176
            paddle.uniform([128, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([128, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_206
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_216
            paddle.uniform([128, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_226
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_228
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_233
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_236
            paddle.uniform([128, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_246
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_248
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([128, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_258
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_266
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([320, 128, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_275
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_280
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_282
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_284
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_285
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_286
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_287
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_291
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_295
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_296
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_300
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_301
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_302
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_303
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_307
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_309
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_310
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_312
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_314
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_316
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_317
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_318
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_319
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_320
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_323
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_325
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_326
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_327
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_328
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_330
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_332
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_333
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_335
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_336
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_341
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_342
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_343
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_344
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_346
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_348
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_350
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_351
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_352
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_355
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_357
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_358
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_359
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_360
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_362
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_365
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_366
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_367
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_368
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_371
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_373
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_374
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_375
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_376
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_378
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_380
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_381
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_382
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_383
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_384
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_387
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_389
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_390
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_391
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_392
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_394
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_396
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_397
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_398
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_399
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_400
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_403
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_405
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_406
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_407
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_408
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_410
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_412
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_413
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_414
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_415
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_416
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_419
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_421
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_422
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_423
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_424
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_426
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_428
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_429
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_430
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_431
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_432
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_435
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_437
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_438
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_439
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_440
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_442
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_444
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_445
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_446
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_447
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_448
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_451
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_453
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_454
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_455
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_456
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_458
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_460
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_461
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_462
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_463
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_464
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_467
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_469
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_470
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_471
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_472
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_474
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_476
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_477
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_478
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_479
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_480
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_483
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_484
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_485
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_486
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_487
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_488
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_490
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_492
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_493
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_494
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_495
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_496
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_499
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_501
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_502
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_503
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_504
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_506
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_508
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_509
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_510
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_511
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_512
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_515
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_514
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_517
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_518
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_519
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_520
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_522
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_524
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_525
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_526
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_527
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_528
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_531
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_533
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_534
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_535
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_536
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_538
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_537
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_539
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_540
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_541
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_542
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_543
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_544
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_547
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_548
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_549
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_550
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_551
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_552
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_554
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_553
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_556
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_557
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_558
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_559
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_560
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_563
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_564
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_565
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_566
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_567
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_568
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_570
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_569
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_571
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_572
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_573
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_574
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_575
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_576
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_579
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_578
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_580
            paddle.uniform([320, 960], dtype='float16', min=0, max=0.5),
            # parameter_581
            paddle.uniform([960], dtype='float16', min=0, max=0.5),
            # parameter_582
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_583
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_584
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_586
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_585
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_587
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_588
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_589
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_590
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_591
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_592
            paddle.uniform([512, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_595
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_594
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_596
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_599
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_598
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_600
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_601
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_602
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_603
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_604
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_606
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_605
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_607
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_608
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_609
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_610
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_611
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_612
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_615
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_614
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_616
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_617
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_618
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_619
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_620
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_622
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_621
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_623
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_624
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_625
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_626
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_627
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_628
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_631
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_630
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_632
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_633
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_634
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_635
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_636
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_638
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_637
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_639
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_640
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_641
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_642
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_643
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_644
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_647
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_646
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_648
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_649
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_650
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_651
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_652
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_654
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_653
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_655
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_656
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_657
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_658
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_659
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_660
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_663
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_662
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_664
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_665
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_666
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_667
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_668
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_670
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_669
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_671
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_672
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_673
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_674
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_675
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_676
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_679
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_678
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_680
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_681
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_682
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_683
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_684
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_686
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_685
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_687
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_688
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_689
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_690
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_691
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_692
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_695
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_694
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_696
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_697
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_698
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_699
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_700
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_702
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_701
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_703
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_704
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_705
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_706
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_707
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_711
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_708
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_710
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_709
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_712
            paddle.uniform([512, 1000], dtype='float16', min=0, max=0.5),
            # parameter_713
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_693
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_677
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_661
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_645
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_629
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_613
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # constant_17
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_597
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # constant_15
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_593
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_577
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_561
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_545
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_529
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_513
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_497
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_481
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_465
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_449
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_433
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_417
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_401
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_385
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_369
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_353
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_337
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_321
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_273
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_269
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float16'),
            # parameter_267
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_265
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_257
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_255
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_247
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_245
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_237
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_235
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_197
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_195
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_187
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_185
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_175
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_105
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_25
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_5
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_0
            paddle.static.InputSpec(shape=[64, 3, 4, 4], dtype='float16'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_12
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_32
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_42
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_62
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_82
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_92
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[128, 64, 2, 2], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_168
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_176
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float16'),
            # parameter_198
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_206
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_216
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_226
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_228
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_233
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_236
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_246
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_248
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float16'),
            # parameter_258
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_266
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[320, 128, 2, 2], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_275
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_280
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_282
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_284
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_285
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_286
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_287
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_291
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_295
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_296
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_300
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_301
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_302
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_303
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_307
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_309
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_310
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_312
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_314
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_316
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_317
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_318
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_319
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_320
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_323
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_325
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_326
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_327
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_328
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_330
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_332
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_333
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_335
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_336
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_341
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_342
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_343
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_344
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_346
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_348
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_349
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_350
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_351
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_352
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_355
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_357
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_358
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_359
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_360
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_362
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_365
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_366
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_367
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_368
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_371
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_373
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_374
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_375
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_376
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_378
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_380
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_381
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_382
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_383
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_384
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_387
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_389
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_390
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_391
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_392
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_394
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_396
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_397
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_398
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_399
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_400
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_403
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_405
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_406
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_407
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_408
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_410
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_412
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_413
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_414
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_415
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_416
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_419
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_421
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_422
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_423
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_424
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_426
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_428
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_429
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_430
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_431
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_432
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_435
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_437
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_438
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_439
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_440
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_442
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_444
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_445
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_446
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_447
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_448
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_451
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_453
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_454
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_455
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_456
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_458
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_460
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_461
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_462
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_463
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_464
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_467
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_469
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_470
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_471
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_472
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_474
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_476
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_477
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_478
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_479
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_480
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_483
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_484
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_485
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_486
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_487
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_488
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_490
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_492
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_493
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_494
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_495
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_496
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_499
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_501
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_502
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_503
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_504
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_506
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_508
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_509
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_510
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_511
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_512
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_515
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_514
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_517
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_518
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_519
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_520
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_522
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_524
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_525
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_526
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_527
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_528
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_531
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_533
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_534
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_535
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_536
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_538
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_537
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_539
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_540
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_541
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_542
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_543
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_544
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_547
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_548
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_549
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_550
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_551
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_552
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_554
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_553
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_556
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_557
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_558
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_559
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_560
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_563
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_564
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_565
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_566
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_567
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_568
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_570
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_569
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_571
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_572
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_573
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_574
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_575
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_576
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_579
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_578
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_580
            paddle.static.InputSpec(shape=[320, 960], dtype='float16'),
            # parameter_581
            paddle.static.InputSpec(shape=[960], dtype='float16'),
            # parameter_582
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_583
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_584
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_586
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_585
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_587
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_588
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_589
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_590
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_591
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_592
            paddle.static.InputSpec(shape=[512, 320, 2, 2], dtype='float16'),
            # parameter_595
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_594
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_596
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_599
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_598
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_600
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_601
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_602
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_603
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_604
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_606
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_605
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_607
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_608
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_609
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_610
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_611
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_612
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_615
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_614
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_616
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_617
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_618
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_619
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_620
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_622
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_621
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_623
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_624
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_625
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_626
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_627
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_628
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_631
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_630
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_632
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_633
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_634
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_635
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_636
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_638
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_637
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_639
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_640
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_641
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_642
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_643
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_644
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_647
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_646
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_648
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_649
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_650
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_651
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_652
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_654
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_653
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_655
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_656
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_657
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_658
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_659
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_660
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_663
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_662
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_664
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_665
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_666
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_667
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_668
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_670
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_669
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_671
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_672
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_673
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_674
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_675
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_676
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_679
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_678
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_680
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_681
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_682
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_683
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_684
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_686
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_685
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_687
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_688
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_689
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_690
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_691
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_692
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_695
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_694
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_696
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_697
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_698
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_699
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_700
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_702
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_701
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_703
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_704
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_705
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_706
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_707
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_711
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_708
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_710
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_709
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_712
            paddle.static.InputSpec(shape=[512, 1000], dtype='float16'),
            # parameter_713
            paddle.static.InputSpec(shape=[1000], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = ModuleOp()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        paddle.seed(2024)
        out = net(*self.inputs)
        return out

    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()