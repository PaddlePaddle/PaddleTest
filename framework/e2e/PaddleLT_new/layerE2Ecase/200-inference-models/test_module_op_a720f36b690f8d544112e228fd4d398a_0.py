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
    return [993][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2502_0_0(self, parameter_365, constant_27, constant_26, constant_25, parameter_349, parameter_335, parameter_305, parameter_275, parameter_245, parameter_215, parameter_185, parameter_155, parameter_125, parameter_95, parameter_89, constant_24, constant_23, constant_22, constant_21, parameter_73, parameter_59, parameter_53, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, parameter_37, constant_14, parameter_23, parameter_17, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, parameter_1, constant_1, constant_0, parameter_0, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_19, parameter_18, parameter_20, parameter_21, parameter_22, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_39, parameter_38, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_55, parameter_54, parameter_56, parameter_57, parameter_58, parameter_61, parameter_60, parameter_62, parameter_63, parameter_64, parameter_65, parameter_67, parameter_66, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_75, parameter_74, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_91, parameter_90, parameter_92, parameter_93, parameter_94, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_101, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_113, parameter_115, parameter_114, parameter_116, parameter_117, parameter_118, parameter_119, parameter_121, parameter_120, parameter_122, parameter_123, parameter_124, parameter_127, parameter_126, parameter_128, parameter_129, parameter_130, parameter_131, parameter_133, parameter_132, parameter_134, parameter_135, parameter_136, parameter_137, parameter_139, parameter_138, parameter_140, parameter_141, parameter_142, parameter_143, parameter_145, parameter_144, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_157, parameter_156, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_171, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, parameter_178, parameter_179, parameter_181, parameter_180, parameter_182, parameter_183, parameter_184, parameter_187, parameter_186, parameter_188, parameter_189, parameter_190, parameter_191, parameter_193, parameter_192, parameter_194, parameter_195, parameter_196, parameter_197, parameter_199, parameter_198, parameter_200, parameter_201, parameter_202, parameter_203, parameter_205, parameter_204, parameter_206, parameter_207, parameter_208, parameter_209, parameter_211, parameter_210, parameter_212, parameter_213, parameter_214, parameter_217, parameter_216, parameter_218, parameter_219, parameter_220, parameter_221, parameter_223, parameter_222, parameter_224, parameter_225, parameter_226, parameter_227, parameter_229, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_235, parameter_234, parameter_236, parameter_237, parameter_238, parameter_239, parameter_241, parameter_240, parameter_242, parameter_243, parameter_244, parameter_247, parameter_246, parameter_248, parameter_249, parameter_250, parameter_251, parameter_253, parameter_252, parameter_254, parameter_255, parameter_256, parameter_257, parameter_259, parameter_258, parameter_260, parameter_261, parameter_262, parameter_263, parameter_265, parameter_264, parameter_266, parameter_267, parameter_268, parameter_269, parameter_271, parameter_270, parameter_272, parameter_273, parameter_274, parameter_277, parameter_276, parameter_278, parameter_279, parameter_280, parameter_281, parameter_283, parameter_282, parameter_284, parameter_285, parameter_286, parameter_287, parameter_289, parameter_288, parameter_290, parameter_291, parameter_292, parameter_293, parameter_295, parameter_294, parameter_296, parameter_297, parameter_298, parameter_299, parameter_301, parameter_300, parameter_302, parameter_303, parameter_304, parameter_307, parameter_306, parameter_308, parameter_309, parameter_310, parameter_311, parameter_313, parameter_312, parameter_314, parameter_315, parameter_316, parameter_317, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_345, parameter_346, parameter_347, parameter_348, parameter_351, parameter_350, parameter_353, parameter_352, parameter_354, parameter_355, parameter_356, parameter_357, parameter_359, parameter_358, parameter_360, parameter_361, parameter_362, parameter_363, parameter_364, parameter_367, parameter_366, parameter_368, parameter_369, parameter_370, parameter_371, parameter_372, parameter_373, parameter_375, parameter_374, parameter_376, parameter_377, parameter_378, parameter_379, parameter_381, parameter_380, parameter_382, parameter_383, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.shape: (4xi32) <- (-1x3x224x224xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(cast_0, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x3x224x224xf16, 96x3x4x4xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 1x96x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, parameter_1)

        # pd_op.flatten_: (-1x96x3136xf16, None) <- (-1x96x56x56xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x96xf16) <- (-1x96x3136xf16)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.layer_norm: (-1x3136x96xf16, -3136xf32, -3136xf32) <- (-1x3136x96xf16, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_0, parameter_2, parameter_3, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x3136x96xf16, -3136xf32, -3136xf32) <- (-1x3136x96xf16, 96xf32, 96xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_0, parameter_4, parameter_5, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x3136x96xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(layer_norm_3, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_1, constant_2, constant_3, constant_2, constant_3, constant_4]

        # pd_op.reshape_: (-1x8x7x8x7x96xf16, 0x-1x3136x96xf16) <- (-1x3136x96xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_3, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x8x7x7x96xf16) <- (-1x8x7x8x7x96xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x8x8x7x7x288xf16) <- (-1x8x8x7x7x96xf16, 96x288xf16)
        matmul_0 = paddle.matmul(transpose_1, parameter_6, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x8x8x7x7x288xf16) <- (-1x8x8x7x7x288xf16, 288xf16)
        add__1 = paddle._C_ops.add_(matmul_0, parameter_7)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, constant_5, constant_6, constant_7, constant_7, constant_8]

        # pd_op.reshape_: (-1x64x49x3x3x32xf16, 0x-1x8x8x7x7x288xf16) <- (-1x8x8x7x7x288xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__1, combine_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x64x3x49x32xf16) <- (-1x64x49x3x3x32xf16)
        transpose_2 = paddle._C_ops.transpose(reshape__2, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x64x3x49x32xf16) <- (3x-1x64x3x49x32xf16, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_2, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x64x3x49x32xf16) <- (3x-1x64x3x49x32xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_2, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x64x3x49x32xf16) <- (3x-1x64x3x49x32xf16, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(transpose_2, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x64x3x32x49xf16) <- (-1x64x3x49x32xf16)
        transpose_3 = paddle._C_ops.transpose(slice_3, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x64x3x49x49xf16) <- (-1x64x3x49x32xf16, -1x64x3x32x49xf16)
        matmul_1 = paddle.matmul(slice_2, transpose_3, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x64x3x49x49xf16) <- (-1x64x3x49x49xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_1, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x64x3x49x49xf16) <- (-1x64x3x49x49xf16)
        softmax__0 = paddle._C_ops.softmax_(scale__0, -1)

        # pd_op.matmul: (-1x64x3x49x32xf16) <- (-1x64x3x49x49xf16, -1x64x3x49x32xf16)
        matmul_2 = paddle.matmul(softmax__0, slice_4, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x64x49x3x32xf16) <- (-1x64x3x49x32xf16)
        transpose_4 = paddle._C_ops.transpose(matmul_2, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_1, constant_2, constant_2, constant_3, constant_3, constant_4]

        # pd_op.reshape_: (-1x8x8x7x7x96xf16, 0x-1x64x49x3x32xf16) <- (-1x64x49x3x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x7x8x7x96xf16) <- (-1x8x8x7x7x96xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__4, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_3 = [slice_1, constant_12, constant_4]

        # pd_op.reshape_: (-1x3136x96xf16, 0x-1x8x7x8x7x96xf16) <- (-1x8x7x8x7x96xf16, [1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_5, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x3136x96xf16) <- (-1x3136x96xf16, 96x96xf16)
        matmul_3 = paddle.matmul(reshape__6, parameter_8, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x96xf16) <- (-1x3136x96xf16, 96xf16)
        add__2 = paddle._C_ops.add_(matmul_3, parameter_9)

        # pd_op.add_: (-1x3136x96xf16) <- (-1x3136x96xf16, -1x3136x96xf16)
        add__3 = paddle._C_ops.add_(layer_norm_0, add__2)

        # pd_op.layer_norm: (-1x3136x96xf16, -3136xf32, -3136xf32) <- (-1x3136x96xf16, 96xf32, 96xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__3, parameter_10, parameter_11, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x3136x384xf16) <- (-1x3136x96xf16, 96x384xf16)
        matmul_4 = paddle.matmul(layer_norm_6, parameter_12, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x384xf16) <- (-1x3136x384xf16, 384xf16)
        add__4 = paddle._C_ops.add_(matmul_4, parameter_13)

        # pd_op.gelu: (-1x3136x384xf16) <- (-1x3136x384xf16)
        gelu_0 = paddle._C_ops.gelu(add__4, False)

        # pd_op.matmul: (-1x3136x96xf16) <- (-1x3136x384xf16, 384x96xf16)
        matmul_5 = paddle.matmul(gelu_0, parameter_14, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x96xf16) <- (-1x3136x96xf16, 96xf16)
        add__5 = paddle._C_ops.add_(matmul_5, parameter_15)

        # pd_op.add_: (-1x3136x96xf16) <- (-1x3136x96xf16, -1x3136x96xf16)
        add__6 = paddle._C_ops.add_(add__3, add__5)

        # pd_op.shape: (3xi32) <- (-1x3136x96xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(add__6, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_2, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x96x3136xf16) <- (-1x3136x96xf16)
        transpose_6 = paddle._C_ops.transpose(add__6, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_4 = [slice_5, constant_4, constant_13, constant_13]

        # pd_op.reshape_: (-1x96x56x56xf16, 0x-1x96x3136xf16) <- (-1x96x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, combine_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 96x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(reshape__8, parameter_16, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 1x96x1x1xf16)
        add__7 = paddle._C_ops.add_(depthwise_conv2d_0, parameter_17)

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, -1x96x56x56xf16)
        add__8 = paddle._C_ops.add_(add__7, reshape__8)

        # pd_op.flatten_: (-1x96x3136xf16, None) <- (-1x96x56x56xf16)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__8, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x96xf16) <- (-1x96x3136xf16)
        transpose_7 = paddle._C_ops.transpose(flatten__2, [0, 2, 1])

        # pd_op.layer_norm: (-1x3136x96xf16, -3136xf32, -3136xf32) <- (-1x3136x96xf16, 96xf32, 96xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_7, parameter_18, parameter_19, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x3136x96xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(layer_norm_9, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_3, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x3136x96xf16) <- (-1x3136x96xf16, 96x96xf16)
        matmul_6 = paddle.matmul(layer_norm_9, parameter_20, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x96xf16) <- (-1x3136x96xf16, 96xf16)
        add__9 = paddle._C_ops.add_(matmul_6, parameter_21)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_6, constant_12, constant_7, constant_8]

        # pd_op.reshape_: (-1x3136x3x32xf16, 0x-1x3136x96xf16) <- (-1x3136x96xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__9, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3x3136x32xf16) <- (-1x3136x3x32xf16)
        transpose_8 = paddle._C_ops.transpose(reshape__10, [0, 2, 1, 3])

        # pd_op.transpose: (-1x96x3136xf16) <- (-1x3136x96xf16)
        transpose_9 = paddle._C_ops.transpose(layer_norm_9, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_6, constant_4, constant_13, constant_13]

        # pd_op.reshape_: (-1x96x56x56xf16, 0x-1x96x3136xf16) <- (-1x96x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x96x7x7xf16) <- (-1x96x56x56xf16, 96x96x8x8xf16)
        conv2d_1 = paddle._C_ops.conv2d(reshape__12, parameter_22, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x96x7x7xf16) <- (-1x96x7x7xf16, 1x96x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_1, parameter_23)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_7 = [slice_6, constant_4, constant_6]

        # pd_op.reshape_: (-1x96x49xf16, 0x-1x96x7x7xf16) <- (-1x96x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__10, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x96xf16) <- (-1x96x49xf16)
        transpose_10 = paddle._C_ops.transpose(reshape__14, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x96xf16, -49xf32, -49xf32) <- (-1x49x96xf16, 96xf32, 96xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_10, parameter_24, parameter_25, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x192xf16) <- (-1x49x96xf16, 96x192xf16)
        matmul_7 = paddle.matmul(layer_norm_12, parameter_26, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x192xf16) <- (-1x49x192xf16, 192xf16)
        add__11 = paddle._C_ops.add_(matmul_7, parameter_27)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_6, constant_6, constant_14, constant_7, constant_8]

        # pd_op.reshape_: (-1x49x2x3x32xf16, 0x-1x49x192xf16) <- (-1x49x192xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__11, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x3x49x32xf16) <- (-1x49x2x3x32xf16)
        transpose_11 = paddle._C_ops.transpose(reshape__16, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x3x49x32xf16) <- (2x-1x3x49x32xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_11, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x3x49x32xf16) <- (2x-1x3x49x32xf16, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(transpose_11, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x3x32x49xf16) <- (-1x3x49x32xf16)
        transpose_12 = paddle._C_ops.transpose(slice_7, [0, 1, 3, 2])

        # pd_op.matmul: (-1x3x3136x49xf16) <- (-1x3x3136x32xf16, -1x3x32x49xf16)
        matmul_8 = paddle.matmul(transpose_8, transpose_12, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x3x3136x49xf16) <- (-1x3x3136x49xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_8, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x3x3136x49xf16) <- (-1x3x3136x49xf16)
        softmax__1 = paddle._C_ops.softmax_(scale__1, -1)

        # pd_op.matmul: (-1x3x3136x32xf16) <- (-1x3x3136x49xf16, -1x3x49x32xf16)
        matmul_9 = paddle.matmul(softmax__1, slice_8, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x3136x3x32xf16) <- (-1x3x3136x32xf16)
        transpose_13 = paddle._C_ops.transpose(matmul_9, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_9 = [slice_6, constant_12, constant_4]

        # pd_op.reshape_: (-1x3136x96xf16, 0x-1x3136x3x32xf16) <- (-1x3136x3x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_13, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x3136x96xf16) <- (-1x3136x96xf16, 96x96xf16)
        matmul_10 = paddle.matmul(reshape__18, parameter_28, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x96xf16) <- (-1x3136x96xf16, 96xf16)
        add__12 = paddle._C_ops.add_(matmul_10, parameter_29)

        # pd_op.add_: (-1x3136x96xf16) <- (-1x3136x96xf16, -1x3136x96xf16)
        add__13 = paddle._C_ops.add_(transpose_7, add__12)

        # pd_op.layer_norm: (-1x3136x96xf16, -3136xf32, -3136xf32) <- (-1x3136x96xf16, 96xf32, 96xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__13, parameter_30, parameter_31, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x3136x384xf16) <- (-1x3136x96xf16, 96x384xf16)
        matmul_11 = paddle.matmul(layer_norm_15, parameter_32, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x384xf16) <- (-1x3136x384xf16, 384xf16)
        add__14 = paddle._C_ops.add_(matmul_11, parameter_33)

        # pd_op.gelu: (-1x3136x384xf16) <- (-1x3136x384xf16)
        gelu_1 = paddle._C_ops.gelu(add__14, False)

        # pd_op.matmul: (-1x3136x96xf16) <- (-1x3136x384xf16, 384x96xf16)
        matmul_12 = paddle.matmul(gelu_1, parameter_34, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x96xf16) <- (-1x3136x96xf16, 96xf16)
        add__15 = paddle._C_ops.add_(matmul_12, parameter_35)

        # pd_op.add_: (-1x3136x96xf16) <- (-1x3136x96xf16, -1x3136x96xf16)
        add__16 = paddle._C_ops.add_(add__13, add__15)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [slice_0, constant_13, constant_13, constant_4]

        # pd_op.reshape_: (-1x56x56x96xf16, 0x-1x3136x96xf16) <- (-1x3136x96xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__16, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x96x56x56xf16) <- (-1x56x56x96xf16)
        transpose_14 = paddle._C_ops.transpose(reshape__20, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x96x56x56xf16, 192x96x2x2xf16)
        conv2d_2 = paddle._C_ops.conv2d(transpose_14, parameter_36, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 1x192x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_2, parameter_37)

        # pd_op.flatten_: (-1x192x784xf16, None) <- (-1x192x28x28xf16)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__17, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x192xf16) <- (-1x192x784xf16)
        transpose_15 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x784x192xf16, -784xf32, -784xf32) <- (-1x784x192xf16, 192xf32, 192xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_15, parameter_38, parameter_39, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x784x192xf16, -784xf32, -784xf32) <- (-1x784x192xf16, 192xf32, 192xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_18, parameter_40, parameter_41, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x192xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(layer_norm_21, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_4, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_9, constant_15, constant_3, constant_15, constant_3, constant_16]

        # pd_op.reshape_: (-1x4x7x4x7x192xf16, 0x-1x784x192xf16) <- (-1x784x192xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_21, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x4x7x7x192xf16) <- (-1x4x7x4x7x192xf16)
        transpose_16 = paddle._C_ops.transpose(reshape__22, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x4x4x7x7x576xf16) <- (-1x4x4x7x7x192xf16, 192x576xf16)
        matmul_13 = paddle.matmul(transpose_16, parameter_42, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x4x4x7x7x576xf16) <- (-1x4x4x7x7x576xf16, 576xf16)
        add__18 = paddle._C_ops.add_(matmul_13, parameter_43)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_9, constant_17, constant_6, constant_7, constant_18, constant_8]

        # pd_op.reshape_: (-1x16x49x3x6x32xf16, 0x-1x4x4x7x7x576xf16) <- (-1x4x4x7x7x576xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__18, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x6x49x32xf16) <- (-1x16x49x3x6x32xf16)
        transpose_17 = paddle._C_ops.transpose(reshape__24, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x16x6x49x32xf16) <- (3x-1x16x6x49x32xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_17, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x16x6x49x32xf16) <- (3x-1x16x6x49x32xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_17, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x16x6x49x32xf16) <- (3x-1x16x6x49x32xf16, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(transpose_17, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x16x6x32x49xf16) <- (-1x16x6x49x32xf16)
        transpose_18 = paddle._C_ops.transpose(slice_11, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x16x6x49x49xf16) <- (-1x16x6x49x32xf16, -1x16x6x32x49xf16)
        matmul_14 = paddle.matmul(slice_10, transpose_18, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x16x6x49x49xf16) <- (-1x16x6x49x49xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_14, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x16x6x49x49xf16) <- (-1x16x6x49x49xf16)
        softmax__2 = paddle._C_ops.softmax_(scale__2, -1)

        # pd_op.matmul: (-1x16x6x49x32xf16) <- (-1x16x6x49x49xf16, -1x16x6x49x32xf16)
        matmul_15 = paddle.matmul(softmax__2, slice_12, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x49x6x32xf16) <- (-1x16x6x49x32xf16)
        transpose_19 = paddle._C_ops.transpose(matmul_15, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_9, constant_15, constant_15, constant_3, constant_3, constant_16]

        # pd_op.reshape_: (-1x4x4x7x7x192xf16, 0x-1x16x49x6x32xf16) <- (-1x16x49x6x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_19, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x7x4x7x192xf16) <- (-1x4x4x7x7x192xf16)
        transpose_20 = paddle._C_ops.transpose(reshape__26, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_14 = [slice_9, constant_19, constant_16]

        # pd_op.reshape_: (-1x784x192xf16, 0x-1x4x7x4x7x192xf16) <- (-1x4x7x4x7x192xf16, [1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x192xf16) <- (-1x784x192xf16, 192x192xf16)
        matmul_16 = paddle.matmul(reshape__28, parameter_44, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, 192xf16)
        add__19 = paddle._C_ops.add_(matmul_16, parameter_45)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, -1x784x192xf16)
        add__20 = paddle._C_ops.add_(layer_norm_18, add__19)

        # pd_op.layer_norm: (-1x784x192xf16, -784xf32, -784xf32) <- (-1x784x192xf16, 192xf32, 192xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__20, parameter_46, parameter_47, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x768xf16) <- (-1x784x192xf16, 192x768xf16)
        matmul_17 = paddle.matmul(layer_norm_24, parameter_48, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x768xf16) <- (-1x784x768xf16, 768xf16)
        add__21 = paddle._C_ops.add_(matmul_17, parameter_49)

        # pd_op.gelu: (-1x784x768xf16) <- (-1x784x768xf16)
        gelu_2 = paddle._C_ops.gelu(add__21, False)

        # pd_op.matmul: (-1x784x192xf16) <- (-1x784x768xf16, 768x192xf16)
        matmul_18 = paddle.matmul(gelu_2, parameter_50, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, 192xf16)
        add__22 = paddle._C_ops.add_(matmul_18, parameter_51)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, -1x784x192xf16)
        add__23 = paddle._C_ops.add_(add__20, add__22)

        # pd_op.shape: (3xi32) <- (-1x784x192xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(add__23, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_5, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x192x784xf16) <- (-1x784x192xf16)
        transpose_21 = paddle._C_ops.transpose(add__23, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_15 = [slice_13, constant_16, constant_20, constant_20]

        # pd_op.reshape_: (-1x192x28x28xf16, 0x-1x192x784xf16) <- (-1x192x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_21, combine_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 192x1x3x3xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(reshape__30, parameter_52, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 1x192x1x1xf16)
        add__24 = paddle._C_ops.add_(depthwise_conv2d_1, parameter_53)

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, -1x192x28x28xf16)
        add__25 = paddle._C_ops.add_(add__24, reshape__30)

        # pd_op.flatten_: (-1x192x784xf16, None) <- (-1x192x28x28xf16)
        flatten__6, flatten__7 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__25, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x192xf16) <- (-1x192x784xf16)
        transpose_22 = paddle._C_ops.transpose(flatten__6, [0, 2, 1])

        # pd_op.layer_norm: (-1x784x192xf16, -784xf32, -784xf32) <- (-1x784x192xf16, 192xf32, 192xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_22, parameter_54, parameter_55, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x192xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(layer_norm_27, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_6, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x784x192xf16) <- (-1x784x192xf16, 192x192xf16)
        matmul_19 = paddle.matmul(layer_norm_27, parameter_56, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, 192xf16)
        add__26 = paddle._C_ops.add_(matmul_19, parameter_57)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_14, constant_19, constant_18, constant_8]

        # pd_op.reshape_: (-1x784x6x32xf16, 0x-1x784x192xf16) <- (-1x784x192xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__26, combine_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x784x32xf16) <- (-1x784x6x32xf16)
        transpose_23 = paddle._C_ops.transpose(reshape__32, [0, 2, 1, 3])

        # pd_op.transpose: (-1x192x784xf16) <- (-1x784x192xf16)
        transpose_24 = paddle._C_ops.transpose(layer_norm_27, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_14, constant_16, constant_20, constant_20]

        # pd_op.reshape_: (-1x192x28x28xf16, 0x-1x192x784xf16) <- (-1x192x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, combine_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x192x7x7xf16) <- (-1x192x28x28xf16, 192x192x4x4xf16)
        conv2d_3 = paddle._C_ops.conv2d(reshape__34, parameter_58, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x7x7xf16) <- (-1x192x7x7xf16, 1x192x1x1xf16)
        add__27 = paddle._C_ops.add_(conv2d_3, parameter_59)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_18 = [slice_14, constant_16, constant_6]

        # pd_op.reshape_: (-1x192x49xf16, 0x-1x192x7x7xf16) <- (-1x192x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__27, combine_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x192xf16) <- (-1x192x49xf16)
        transpose_25 = paddle._C_ops.transpose(reshape__36, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x192xf16, -49xf32, -49xf32) <- (-1x49x192xf16, 192xf32, 192xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_25, parameter_60, parameter_61, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x384xf16) <- (-1x49x192xf16, 192x384xf16)
        matmul_20 = paddle.matmul(layer_norm_30, parameter_62, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x384xf16) <- (-1x49x384xf16, 384xf16)
        add__28 = paddle._C_ops.add_(matmul_20, parameter_63)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_14, constant_6, constant_14, constant_18, constant_8]

        # pd_op.reshape_: (-1x49x2x6x32xf16, 0x-1x49x384xf16) <- (-1x49x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__28, combine_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x49x32xf16) <- (-1x49x2x6x32xf16)
        transpose_26 = paddle._C_ops.transpose(reshape__38, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x49x32xf16) <- (2x-1x6x49x32xf16, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_26, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x49x32xf16) <- (2x-1x6x49x32xf16, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_26, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x6x32x49xf16) <- (-1x6x49x32xf16)
        transpose_27 = paddle._C_ops.transpose(slice_15, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x784x49xf16) <- (-1x6x784x32xf16, -1x6x32x49xf16)
        matmul_21 = paddle.matmul(transpose_23, transpose_27, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x784x49xf16) <- (-1x6x784x49xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(matmul_21, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x6x784x49xf16) <- (-1x6x784x49xf16)
        softmax__3 = paddle._C_ops.softmax_(scale__3, -1)

        # pd_op.matmul: (-1x6x784x32xf16) <- (-1x6x784x49xf16, -1x6x49x32xf16)
        matmul_22 = paddle.matmul(softmax__3, slice_16, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x784x6x32xf16) <- (-1x6x784x32xf16)
        transpose_28 = paddle._C_ops.transpose(matmul_22, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_20 = [slice_14, constant_19, constant_16]

        # pd_op.reshape_: (-1x784x192xf16, 0x-1x784x6x32xf16) <- (-1x784x6x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, combine_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x192xf16) <- (-1x784x192xf16, 192x192xf16)
        matmul_23 = paddle.matmul(reshape__40, parameter_64, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, 192xf16)
        add__29 = paddle._C_ops.add_(matmul_23, parameter_65)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, -1x784x192xf16)
        add__30 = paddle._C_ops.add_(transpose_22, add__29)

        # pd_op.layer_norm: (-1x784x192xf16, -784xf32, -784xf32) <- (-1x784x192xf16, 192xf32, 192xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__30, parameter_66, parameter_67, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x768xf16) <- (-1x784x192xf16, 192x768xf16)
        matmul_24 = paddle.matmul(layer_norm_33, parameter_68, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x768xf16) <- (-1x784x768xf16, 768xf16)
        add__31 = paddle._C_ops.add_(matmul_24, parameter_69)

        # pd_op.gelu: (-1x784x768xf16) <- (-1x784x768xf16)
        gelu_3 = paddle._C_ops.gelu(add__31, False)

        # pd_op.matmul: (-1x784x192xf16) <- (-1x784x768xf16, 768x192xf16)
        matmul_25 = paddle.matmul(gelu_3, parameter_70, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, 192xf16)
        add__32 = paddle._C_ops.add_(matmul_25, parameter_71)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, -1x784x192xf16)
        add__33 = paddle._C_ops.add_(add__30, add__32)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_0, constant_20, constant_20, constant_16]

        # pd_op.reshape_: (-1x28x28x192xf16, 0x-1x784x192xf16) <- (-1x784x192xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__33, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x192x28x28xf16) <- (-1x28x28x192xf16)
        transpose_29 = paddle._C_ops.transpose(reshape__42, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x192x28x28xf16, 384x192x2x2xf16)
        conv2d_4 = paddle._C_ops.conv2d(transpose_29, parameter_72, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__34 = paddle._C_ops.add_(conv2d_4, parameter_73)

        # pd_op.flatten_: (-1x384x196xf16, None) <- (-1x384x14x14xf16)
        flatten__8, flatten__9 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__34, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x384xf16) <- (-1x384x196xf16)
        transpose_30 = paddle._C_ops.transpose(flatten__8, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_30, parameter_74, parameter_75, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_36, parameter_76, parameter_77, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(layer_norm_39, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_7, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_22 = [slice_17, constant_14, constant_3, constant_14, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x7x2x7x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_39, combine_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x7x7x384xf16) <- (-1x2x7x2x7x384xf16)
        transpose_31 = paddle._C_ops.transpose(reshape__44, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x384xf16, 384x1152xf16)
        matmul_26 = paddle.matmul(transpose_31, parameter_78, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, 1152xf16)
        add__35 = paddle._C_ops.add_(matmul_26, parameter_79)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_17, constant_15, constant_6, constant_7, constant_22, constant_8]

        # pd_op.reshape_: (-1x4x49x3x12x32xf16, 0x-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__35, combine_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x12x49x32xf16) <- (-1x4x49x3x12x32xf16)
        transpose_32 = paddle._C_ops.transpose(reshape__46, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_32, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_32, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(transpose_32, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x4x12x32x49xf16) <- (-1x4x12x49x32xf16)
        transpose_33 = paddle._C_ops.transpose(slice_19, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x4x12x49x49xf16) <- (-1x4x12x49x32xf16, -1x4x12x32x49xf16)
        matmul_27 = paddle.matmul(slice_18, transpose_33, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_27, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16)
        softmax__4 = paddle._C_ops.softmax_(scale__4, -1)

        # pd_op.matmul: (-1x4x12x49x32xf16) <- (-1x4x12x49x49xf16, -1x4x12x49x32xf16)
        matmul_28 = paddle.matmul(softmax__4, slice_20, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x4x49x12x32xf16) <- (-1x4x12x49x32xf16)
        transpose_34 = paddle._C_ops.transpose(matmul_28, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_17, constant_14, constant_14, constant_3, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x2x7x7x384xf16, 0x-1x4x49x12x32xf16) <- (-1x4x49x12x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_34, combine_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x7x2x7x384xf16) <- (-1x2x2x7x7x384xf16)
        transpose_35 = paddle._C_ops.transpose(reshape__48, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_17, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x2x7x2x7x384xf16) <- (-1x2x7x2x7x384xf16, [1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_35, combine_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_29 = paddle.matmul(reshape__50, parameter_80, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__36 = paddle._C_ops.add_(matmul_29, parameter_81)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__37 = paddle._C_ops.add_(layer_norm_36, add__36)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__37, parameter_82, parameter_83, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_30 = paddle.matmul(layer_norm_42, parameter_84, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__38 = paddle._C_ops.add_(matmul_30, parameter_85)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_4 = paddle._C_ops.gelu(add__38, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_31 = paddle.matmul(gelu_4, parameter_86, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__39 = paddle._C_ops.add_(matmul_31, parameter_87)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__40 = paddle._C_ops.add_(add__37, add__39)

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(add__40, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(shape_8, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_36 = paddle._C_ops.transpose(add__40, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_21, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_36, combine_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(reshape__52, parameter_88, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__41 = paddle._C_ops.add_(depthwise_conv2d_2, parameter_89)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__42 = paddle._C_ops.add_(add__41, reshape__52)

        # pd_op.flatten_: (-1x384x196xf16, None) <- (-1x384x14x14xf16)
        flatten__10, flatten__11 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__42, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x384xf16) <- (-1x384x196xf16)
        transpose_37 = paddle._C_ops.transpose(flatten__10, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_37, parameter_90, parameter_91, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(layer_norm_45, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(shape_9, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_32 = paddle.matmul(layer_norm_45, parameter_92, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__43 = paddle._C_ops.add_(matmul_32, parameter_93)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_27 = [slice_22, constant_23, constant_22, constant_8]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__43, combine_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_38 = paddle._C_ops.transpose(reshape__54, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_39 = paddle._C_ops.transpose(layer_norm_45, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_28 = [slice_22, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_39, combine_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf16) <- (-1x384x14x14xf16, 384x384x2x2xf16)
        conv2d_5 = paddle._C_ops.conv2d(reshape__56, parameter_94, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x7x7xf16) <- (-1x384x7x7xf16, 1x384x1x1xf16)
        add__44 = paddle._C_ops.add_(conv2d_5, parameter_95)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_29 = [slice_22, constant_21, constant_6]

        # pd_op.reshape_: (-1x384x49xf16, 0x-1x384x7x7xf16) <- (-1x384x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__44, combine_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_40 = paddle._C_ops.transpose(reshape__58, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x384xf16, -49xf32, -49xf32) <- (-1x49x384xf16, 384xf32, 384xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_40, parameter_96, parameter_97, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x384xf16, 384x768xf16)
        matmul_33 = paddle.matmul(layer_norm_48, parameter_98, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__45 = paddle._C_ops.add_(matmul_33, parameter_99)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_30 = [slice_22, constant_6, constant_14, constant_22, constant_8]

        # pd_op.reshape_: (-1x49x2x12x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__45, combine_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x12x49x32xf16) <- (-1x49x2x12x32xf16)
        transpose_41 = paddle._C_ops.transpose(reshape__60, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_41, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_41, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x12x49x32xf16)
        transpose_42 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_34 = paddle.matmul(transpose_38, transpose_42, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_34, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__5 = paddle._C_ops.softmax_(scale__5, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_35 = paddle.matmul(softmax__5, slice_24, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_43 = paddle._C_ops.transpose(matmul_35, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_31 = [slice_22, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_43, combine_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_36 = paddle.matmul(reshape__62, parameter_100, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__46 = paddle._C_ops.add_(matmul_36, parameter_101)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__47 = paddle._C_ops.add_(transpose_37, add__46)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__47, parameter_102, parameter_103, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_37 = paddle.matmul(layer_norm_51, parameter_104, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__48 = paddle._C_ops.add_(matmul_37, parameter_105)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_5 = paddle._C_ops.gelu(add__48, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_38 = paddle.matmul(gelu_5, parameter_106, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__49 = paddle._C_ops.add_(matmul_38, parameter_107)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__50 = paddle._C_ops.add_(add__47, add__49)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__50, parameter_108, parameter_109, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_10 = paddle._C_ops.shape(paddle.cast(layer_norm_54, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_10, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_25, constant_14, constant_3, constant_14, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x7x2x7x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_54, combine_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x7x7x384xf16) <- (-1x2x7x2x7x384xf16)
        transpose_44 = paddle._C_ops.transpose(reshape__64, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x384xf16, 384x1152xf16)
        matmul_39 = paddle.matmul(transpose_44, parameter_110, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, 1152xf16)
        add__51 = paddle._C_ops.add_(matmul_39, parameter_111)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_25, constant_15, constant_6, constant_7, constant_22, constant_8]

        # pd_op.reshape_: (-1x4x49x3x12x32xf16, 0x-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__51, combine_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x12x49x32xf16) <- (-1x4x49x3x12x32xf16)
        transpose_45 = paddle._C_ops.transpose(reshape__66, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_45, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_45, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(transpose_45, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x4x12x32x49xf16) <- (-1x4x12x49x32xf16)
        transpose_46 = paddle._C_ops.transpose(slice_27, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x4x12x49x49xf16) <- (-1x4x12x49x32xf16, -1x4x12x32x49xf16)
        matmul_40 = paddle.matmul(slice_26, transpose_46, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(matmul_40, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16)
        softmax__6 = paddle._C_ops.softmax_(scale__6, -1)

        # pd_op.matmul: (-1x4x12x49x32xf16) <- (-1x4x12x49x49xf16, -1x4x12x49x32xf16)
        matmul_41 = paddle.matmul(softmax__6, slice_28, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x4x49x12x32xf16) <- (-1x4x12x49x32xf16)
        transpose_47 = paddle._C_ops.transpose(matmul_41, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_34 = [slice_25, constant_14, constant_14, constant_3, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x2x7x7x384xf16, 0x-1x4x49x12x32xf16) <- (-1x4x49x12x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_47, combine_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x7x2x7x384xf16) <- (-1x2x2x7x7x384xf16)
        transpose_48 = paddle._C_ops.transpose(reshape__68, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_35 = [slice_25, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x2x7x2x7x384xf16) <- (-1x2x7x2x7x384xf16, [1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_48, combine_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_42 = paddle.matmul(reshape__70, parameter_112, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__52 = paddle._C_ops.add_(matmul_42, parameter_113)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__53 = paddle._C_ops.add_(add__50, add__52)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__53, parameter_114, parameter_115, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_43 = paddle.matmul(layer_norm_57, parameter_116, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__54 = paddle._C_ops.add_(matmul_43, parameter_117)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_6 = paddle._C_ops.gelu(add__54, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_44 = paddle.matmul(gelu_6, parameter_118, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__55 = paddle._C_ops.add_(matmul_44, parameter_119)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__56 = paddle._C_ops.add_(add__53, add__55)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__56, parameter_120, parameter_121, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(layer_norm_60, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_11, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_45 = paddle.matmul(layer_norm_60, parameter_122, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__57 = paddle._C_ops.add_(matmul_45, parameter_123)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_36 = [slice_29, constant_23, constant_22, constant_8]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__57, combine_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_49 = paddle._C_ops.transpose(reshape__72, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_50 = paddle._C_ops.transpose(layer_norm_60, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_29, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_50, combine_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf16) <- (-1x384x14x14xf16, 384x384x2x2xf16)
        conv2d_6 = paddle._C_ops.conv2d(reshape__74, parameter_124, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x7x7xf16) <- (-1x384x7x7xf16, 1x384x1x1xf16)
        add__58 = paddle._C_ops.add_(conv2d_6, parameter_125)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_38 = [slice_29, constant_21, constant_6]

        # pd_op.reshape_: (-1x384x49xf16, 0x-1x384x7x7xf16) <- (-1x384x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__58, combine_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_51 = paddle._C_ops.transpose(reshape__76, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x384xf16, -49xf32, -49xf32) <- (-1x49x384xf16, 384xf32, 384xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_51, parameter_126, parameter_127, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x384xf16, 384x768xf16)
        matmul_46 = paddle.matmul(layer_norm_63, parameter_128, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__59 = paddle._C_ops.add_(matmul_46, parameter_129)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_39 = [slice_29, constant_6, constant_14, constant_22, constant_8]

        # pd_op.reshape_: (-1x49x2x12x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__59, combine_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x12x49x32xf16) <- (-1x49x2x12x32xf16)
        transpose_52 = paddle._C_ops.transpose(reshape__78, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_52, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_52, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x12x49x32xf16)
        transpose_53 = paddle._C_ops.transpose(slice_30, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_47 = paddle.matmul(transpose_49, transpose_53, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(matmul_47, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__7 = paddle._C_ops.softmax_(scale__7, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_48 = paddle.matmul(softmax__7, slice_31, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_54 = paddle._C_ops.transpose(matmul_48, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_40 = [slice_29, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_54, combine_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_49 = paddle.matmul(reshape__80, parameter_130, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__60 = paddle._C_ops.add_(matmul_49, parameter_131)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__61 = paddle._C_ops.add_(add__56, add__60)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__61, parameter_132, parameter_133, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_50 = paddle.matmul(layer_norm_66, parameter_134, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__62 = paddle._C_ops.add_(matmul_50, parameter_135)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_7 = paddle._C_ops.gelu(add__62, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_51 = paddle.matmul(gelu_7, parameter_136, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__63 = paddle._C_ops.add_(matmul_51, parameter_137)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__64 = paddle._C_ops.add_(add__61, add__63)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__64, parameter_138, parameter_139, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_12 = paddle._C_ops.shape(paddle.cast(layer_norm_69, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_12, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_32, constant_14, constant_3, constant_14, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x7x2x7x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_69, combine_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x7x7x384xf16) <- (-1x2x7x2x7x384xf16)
        transpose_55 = paddle._C_ops.transpose(reshape__82, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x384xf16, 384x1152xf16)
        matmul_52 = paddle.matmul(transpose_55, parameter_140, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, 1152xf16)
        add__65 = paddle._C_ops.add_(matmul_52, parameter_141)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_32, constant_15, constant_6, constant_7, constant_22, constant_8]

        # pd_op.reshape_: (-1x4x49x3x12x32xf16, 0x-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__65, combine_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x12x49x32xf16) <- (-1x4x49x3x12x32xf16)
        transpose_56 = paddle._C_ops.transpose(reshape__84, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_56, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_56, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_56, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x4x12x32x49xf16) <- (-1x4x12x49x32xf16)
        transpose_57 = paddle._C_ops.transpose(slice_34, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x4x12x49x49xf16) <- (-1x4x12x49x32xf16, -1x4x12x32x49xf16)
        matmul_53 = paddle.matmul(slice_33, transpose_57, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16, 1xf32)
        scale__8 = paddle._C_ops.scale_(matmul_53, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16)
        softmax__8 = paddle._C_ops.softmax_(scale__8, -1)

        # pd_op.matmul: (-1x4x12x49x32xf16) <- (-1x4x12x49x49xf16, -1x4x12x49x32xf16)
        matmul_54 = paddle.matmul(softmax__8, slice_35, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x4x49x12x32xf16) <- (-1x4x12x49x32xf16)
        transpose_58 = paddle._C_ops.transpose(matmul_54, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_43 = [slice_32, constant_14, constant_14, constant_3, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x2x7x7x384xf16, 0x-1x4x49x12x32xf16) <- (-1x4x49x12x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_58, combine_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x7x2x7x384xf16) <- (-1x2x2x7x7x384xf16)
        transpose_59 = paddle._C_ops.transpose(reshape__86, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_44 = [slice_32, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x2x7x2x7x384xf16) <- (-1x2x7x2x7x384xf16, [1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_59, combine_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_55 = paddle.matmul(reshape__88, parameter_142, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__66 = paddle._C_ops.add_(matmul_55, parameter_143)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__67 = paddle._C_ops.add_(add__64, add__66)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__67, parameter_144, parameter_145, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_56 = paddle.matmul(layer_norm_72, parameter_146, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__68 = paddle._C_ops.add_(matmul_56, parameter_147)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_8 = paddle._C_ops.gelu(add__68, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_57 = paddle.matmul(gelu_8, parameter_148, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__69 = paddle._C_ops.add_(matmul_57, parameter_149)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__70 = paddle._C_ops.add_(add__67, add__69)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__70, parameter_150, parameter_151, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_13 = paddle._C_ops.shape(paddle.cast(layer_norm_75, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(shape_13, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_58 = paddle.matmul(layer_norm_75, parameter_152, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__71 = paddle._C_ops.add_(matmul_58, parameter_153)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [slice_36, constant_23, constant_22, constant_8]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__71, combine_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_60 = paddle._C_ops.transpose(reshape__90, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_61 = paddle._C_ops.transpose(layer_norm_75, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_46 = [slice_36, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_61, combine_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf16) <- (-1x384x14x14xf16, 384x384x2x2xf16)
        conv2d_7 = paddle._C_ops.conv2d(reshape__92, parameter_154, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x7x7xf16) <- (-1x384x7x7xf16, 1x384x1x1xf16)
        add__72 = paddle._C_ops.add_(conv2d_7, parameter_155)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_47 = [slice_36, constant_21, constant_6]

        # pd_op.reshape_: (-1x384x49xf16, 0x-1x384x7x7xf16) <- (-1x384x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__72, combine_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_62 = paddle._C_ops.transpose(reshape__94, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x384xf16, -49xf32, -49xf32) <- (-1x49x384xf16, 384xf32, 384xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_62, parameter_156, parameter_157, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x384xf16, 384x768xf16)
        matmul_59 = paddle.matmul(layer_norm_78, parameter_158, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__73 = paddle._C_ops.add_(matmul_59, parameter_159)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_48 = [slice_36, constant_6, constant_14, constant_22, constant_8]

        # pd_op.reshape_: (-1x49x2x12x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__73, combine_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x12x49x32xf16) <- (-1x49x2x12x32xf16)
        transpose_63 = paddle._C_ops.transpose(reshape__96, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(transpose_63, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(transpose_63, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x12x49x32xf16)
        transpose_64 = paddle._C_ops.transpose(slice_37, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_60 = paddle.matmul(transpose_60, transpose_64, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__9 = paddle._C_ops.scale_(matmul_60, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__9 = paddle._C_ops.softmax_(scale__9, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_61 = paddle.matmul(softmax__9, slice_38, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_65 = paddle._C_ops.transpose(matmul_61, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_49 = [slice_36, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_65, combine_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_62 = paddle.matmul(reshape__98, parameter_160, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__74 = paddle._C_ops.add_(matmul_62, parameter_161)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__75 = paddle._C_ops.add_(add__70, add__74)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__75, parameter_162, parameter_163, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_63 = paddle.matmul(layer_norm_81, parameter_164, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__76 = paddle._C_ops.add_(matmul_63, parameter_165)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_9 = paddle._C_ops.gelu(add__76, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_64 = paddle.matmul(gelu_9, parameter_166, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__77 = paddle._C_ops.add_(matmul_64, parameter_167)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__78 = paddle._C_ops.add_(add__75, add__77)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__78, parameter_168, parameter_169, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_14 = paddle._C_ops.shape(paddle.cast(layer_norm_84, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(shape_14, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_50 = [slice_39, constant_14, constant_3, constant_14, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x7x2x7x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_84, combine_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x7x7x384xf16) <- (-1x2x7x2x7x384xf16)
        transpose_66 = paddle._C_ops.transpose(reshape__100, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x384xf16, 384x1152xf16)
        matmul_65 = paddle.matmul(transpose_66, parameter_170, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, 1152xf16)
        add__79 = paddle._C_ops.add_(matmul_65, parameter_171)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_51 = [slice_39, constant_15, constant_6, constant_7, constant_22, constant_8]

        # pd_op.reshape_: (-1x4x49x3x12x32xf16, 0x-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__79, combine_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x12x49x32xf16) <- (-1x4x49x3x12x32xf16)
        transpose_67 = paddle._C_ops.transpose(reshape__102, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(transpose_67, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(transpose_67, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(transpose_67, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x4x12x32x49xf16) <- (-1x4x12x49x32xf16)
        transpose_68 = paddle._C_ops.transpose(slice_41, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x4x12x49x49xf16) <- (-1x4x12x49x32xf16, -1x4x12x32x49xf16)
        matmul_66 = paddle.matmul(slice_40, transpose_68, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16, 1xf32)
        scale__10 = paddle._C_ops.scale_(matmul_66, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16)
        softmax__10 = paddle._C_ops.softmax_(scale__10, -1)

        # pd_op.matmul: (-1x4x12x49x32xf16) <- (-1x4x12x49x49xf16, -1x4x12x49x32xf16)
        matmul_67 = paddle.matmul(softmax__10, slice_42, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x4x49x12x32xf16) <- (-1x4x12x49x32xf16)
        transpose_69 = paddle._C_ops.transpose(matmul_67, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_52 = [slice_39, constant_14, constant_14, constant_3, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x2x7x7x384xf16, 0x-1x4x49x12x32xf16) <- (-1x4x49x12x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_69, combine_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x7x2x7x384xf16) <- (-1x2x2x7x7x384xf16)
        transpose_70 = paddle._C_ops.transpose(reshape__104, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_53 = [slice_39, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x2x7x2x7x384xf16) <- (-1x2x7x2x7x384xf16, [1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_70, combine_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_68 = paddle.matmul(reshape__106, parameter_172, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__80 = paddle._C_ops.add_(matmul_68, parameter_173)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__81 = paddle._C_ops.add_(add__78, add__80)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__81, parameter_174, parameter_175, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_69 = paddle.matmul(layer_norm_87, parameter_176, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__82 = paddle._C_ops.add_(matmul_69, parameter_177)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_10 = paddle._C_ops.gelu(add__82, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_70 = paddle.matmul(gelu_10, parameter_178, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__83 = paddle._C_ops.add_(matmul_70, parameter_179)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__84 = paddle._C_ops.add_(add__81, add__83)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__84, parameter_180, parameter_181, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_15 = paddle._C_ops.shape(paddle.cast(layer_norm_90, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(shape_15, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_71 = paddle.matmul(layer_norm_90, parameter_182, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__85 = paddle._C_ops.add_(matmul_71, parameter_183)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_54 = [slice_43, constant_23, constant_22, constant_8]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__85, combine_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_71 = paddle._C_ops.transpose(reshape__108, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_72 = paddle._C_ops.transpose(layer_norm_90, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_55 = [slice_43, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_72, combine_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf16) <- (-1x384x14x14xf16, 384x384x2x2xf16)
        conv2d_8 = paddle._C_ops.conv2d(reshape__110, parameter_184, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x7x7xf16) <- (-1x384x7x7xf16, 1x384x1x1xf16)
        add__86 = paddle._C_ops.add_(conv2d_8, parameter_185)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_56 = [slice_43, constant_21, constant_6]

        # pd_op.reshape_: (-1x384x49xf16, 0x-1x384x7x7xf16) <- (-1x384x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__86, combine_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_73 = paddle._C_ops.transpose(reshape__112, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x384xf16, -49xf32, -49xf32) <- (-1x49x384xf16, 384xf32, 384xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_73, parameter_186, parameter_187, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x384xf16, 384x768xf16)
        matmul_72 = paddle.matmul(layer_norm_93, parameter_188, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__87 = paddle._C_ops.add_(matmul_72, parameter_189)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_57 = [slice_43, constant_6, constant_14, constant_22, constant_8]

        # pd_op.reshape_: (-1x49x2x12x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__87, combine_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x12x49x32xf16) <- (-1x49x2x12x32xf16)
        transpose_74 = paddle._C_ops.transpose(reshape__114, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(transpose_74, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(transpose_74, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x12x49x32xf16)
        transpose_75 = paddle._C_ops.transpose(slice_44, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_73 = paddle.matmul(transpose_71, transpose_75, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__11 = paddle._C_ops.scale_(matmul_73, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__11 = paddle._C_ops.softmax_(scale__11, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_74 = paddle.matmul(softmax__11, slice_45, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_76 = paddle._C_ops.transpose(matmul_74, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_58 = [slice_43, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_76, combine_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_75 = paddle.matmul(reshape__116, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__88 = paddle._C_ops.add_(matmul_75, parameter_191)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__89 = paddle._C_ops.add_(add__84, add__88)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__89, parameter_192, parameter_193, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_76 = paddle.matmul(layer_norm_96, parameter_194, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__90 = paddle._C_ops.add_(matmul_76, parameter_195)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_11 = paddle._C_ops.gelu(add__90, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_77 = paddle.matmul(gelu_11, parameter_196, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__91 = paddle._C_ops.add_(matmul_77, parameter_197)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__92 = paddle._C_ops.add_(add__89, add__91)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__92, parameter_198, parameter_199, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_16 = paddle._C_ops.shape(paddle.cast(layer_norm_99, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(shape_16, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_59 = [slice_46, constant_14, constant_3, constant_14, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x7x2x7x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_99, combine_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x7x7x384xf16) <- (-1x2x7x2x7x384xf16)
        transpose_77 = paddle._C_ops.transpose(reshape__118, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x384xf16, 384x1152xf16)
        matmul_78 = paddle.matmul(transpose_77, parameter_200, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, 1152xf16)
        add__93 = paddle._C_ops.add_(matmul_78, parameter_201)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_60 = [slice_46, constant_15, constant_6, constant_7, constant_22, constant_8]

        # pd_op.reshape_: (-1x4x49x3x12x32xf16, 0x-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__93, combine_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x12x49x32xf16) <- (-1x4x49x3x12x32xf16)
        transpose_78 = paddle._C_ops.transpose(reshape__120, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(transpose_78, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(transpose_78, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(transpose_78, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x4x12x32x49xf16) <- (-1x4x12x49x32xf16)
        transpose_79 = paddle._C_ops.transpose(slice_48, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x4x12x49x49xf16) <- (-1x4x12x49x32xf16, -1x4x12x32x49xf16)
        matmul_79 = paddle.matmul(slice_47, transpose_79, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16, 1xf32)
        scale__12 = paddle._C_ops.scale_(matmul_79, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16)
        softmax__12 = paddle._C_ops.softmax_(scale__12, -1)

        # pd_op.matmul: (-1x4x12x49x32xf16) <- (-1x4x12x49x49xf16, -1x4x12x49x32xf16)
        matmul_80 = paddle.matmul(softmax__12, slice_49, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x4x49x12x32xf16) <- (-1x4x12x49x32xf16)
        transpose_80 = paddle._C_ops.transpose(matmul_80, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_61 = [slice_46, constant_14, constant_14, constant_3, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x2x7x7x384xf16, 0x-1x4x49x12x32xf16) <- (-1x4x49x12x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_80, combine_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x7x2x7x384xf16) <- (-1x2x2x7x7x384xf16)
        transpose_81 = paddle._C_ops.transpose(reshape__122, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_62 = [slice_46, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x2x7x2x7x384xf16) <- (-1x2x7x2x7x384xf16, [1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_81, combine_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_81 = paddle.matmul(reshape__124, parameter_202, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__94 = paddle._C_ops.add_(matmul_81, parameter_203)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__95 = paddle._C_ops.add_(add__92, add__94)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__95, parameter_204, parameter_205, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_82 = paddle.matmul(layer_norm_102, parameter_206, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__96 = paddle._C_ops.add_(matmul_82, parameter_207)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_12 = paddle._C_ops.gelu(add__96, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_83 = paddle.matmul(gelu_12, parameter_208, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__97 = paddle._C_ops.add_(matmul_83, parameter_209)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__98 = paddle._C_ops.add_(add__95, add__97)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__98, parameter_210, parameter_211, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_17 = paddle._C_ops.shape(paddle.cast(layer_norm_105, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(shape_17, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_84 = paddle.matmul(layer_norm_105, parameter_212, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__99 = paddle._C_ops.add_(matmul_84, parameter_213)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_63 = [slice_50, constant_23, constant_22, constant_8]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__99, combine_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_82 = paddle._C_ops.transpose(reshape__126, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_83 = paddle._C_ops.transpose(layer_norm_105, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_64 = [slice_50, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_83, combine_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf16) <- (-1x384x14x14xf16, 384x384x2x2xf16)
        conv2d_9 = paddle._C_ops.conv2d(reshape__128, parameter_214, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x7x7xf16) <- (-1x384x7x7xf16, 1x384x1x1xf16)
        add__100 = paddle._C_ops.add_(conv2d_9, parameter_215)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_65 = [slice_50, constant_21, constant_6]

        # pd_op.reshape_: (-1x384x49xf16, 0x-1x384x7x7xf16) <- (-1x384x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__100, combine_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_84 = paddle._C_ops.transpose(reshape__130, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x384xf16, -49xf32, -49xf32) <- (-1x49x384xf16, 384xf32, 384xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_84, parameter_216, parameter_217, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x384xf16, 384x768xf16)
        matmul_85 = paddle.matmul(layer_norm_108, parameter_218, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__101 = paddle._C_ops.add_(matmul_85, parameter_219)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_66 = [slice_50, constant_6, constant_14, constant_22, constant_8]

        # pd_op.reshape_: (-1x49x2x12x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__101, combine_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x12x49x32xf16) <- (-1x49x2x12x32xf16)
        transpose_85 = paddle._C_ops.transpose(reshape__132, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(transpose_85, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(transpose_85, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x12x49x32xf16)
        transpose_86 = paddle._C_ops.transpose(slice_51, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_86 = paddle.matmul(transpose_82, transpose_86, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__13 = paddle._C_ops.scale_(matmul_86, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__13 = paddle._C_ops.softmax_(scale__13, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_87 = paddle.matmul(softmax__13, slice_52, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_87 = paddle._C_ops.transpose(matmul_87, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_67 = [slice_50, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_87, combine_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_88 = paddle.matmul(reshape__134, parameter_220, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__102 = paddle._C_ops.add_(matmul_88, parameter_221)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__103 = paddle._C_ops.add_(add__98, add__102)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__103, parameter_222, parameter_223, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_89 = paddle.matmul(layer_norm_111, parameter_224, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__104 = paddle._C_ops.add_(matmul_89, parameter_225)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_13 = paddle._C_ops.gelu(add__104, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_90 = paddle.matmul(gelu_13, parameter_226, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__105 = paddle._C_ops.add_(matmul_90, parameter_227)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__106 = paddle._C_ops.add_(add__103, add__105)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__106, parameter_228, parameter_229, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_18 = paddle._C_ops.shape(paddle.cast(layer_norm_114, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(shape_18, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_68 = [slice_53, constant_14, constant_3, constant_14, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x7x2x7x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_114, combine_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x7x7x384xf16) <- (-1x2x7x2x7x384xf16)
        transpose_88 = paddle._C_ops.transpose(reshape__136, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x384xf16, 384x1152xf16)
        matmul_91 = paddle.matmul(transpose_88, parameter_230, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, 1152xf16)
        add__107 = paddle._C_ops.add_(matmul_91, parameter_231)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_69 = [slice_53, constant_15, constant_6, constant_7, constant_22, constant_8]

        # pd_op.reshape_: (-1x4x49x3x12x32xf16, 0x-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__107, combine_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x12x49x32xf16) <- (-1x4x49x3x12x32xf16)
        transpose_89 = paddle._C_ops.transpose(reshape__138, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(transpose_89, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(transpose_89, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(transpose_89, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x4x12x32x49xf16) <- (-1x4x12x49x32xf16)
        transpose_90 = paddle._C_ops.transpose(slice_55, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x4x12x49x49xf16) <- (-1x4x12x49x32xf16, -1x4x12x32x49xf16)
        matmul_92 = paddle.matmul(slice_54, transpose_90, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16, 1xf32)
        scale__14 = paddle._C_ops.scale_(matmul_92, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16)
        softmax__14 = paddle._C_ops.softmax_(scale__14, -1)

        # pd_op.matmul: (-1x4x12x49x32xf16) <- (-1x4x12x49x49xf16, -1x4x12x49x32xf16)
        matmul_93 = paddle.matmul(softmax__14, slice_56, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x4x49x12x32xf16) <- (-1x4x12x49x32xf16)
        transpose_91 = paddle._C_ops.transpose(matmul_93, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_70 = [slice_53, constant_14, constant_14, constant_3, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x2x7x7x384xf16, 0x-1x4x49x12x32xf16) <- (-1x4x49x12x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_91, combine_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x7x2x7x384xf16) <- (-1x2x2x7x7x384xf16)
        transpose_92 = paddle._C_ops.transpose(reshape__140, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_71 = [slice_53, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x2x7x2x7x384xf16) <- (-1x2x7x2x7x384xf16, [1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_92, combine_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_94 = paddle.matmul(reshape__142, parameter_232, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__108 = paddle._C_ops.add_(matmul_94, parameter_233)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__109 = paddle._C_ops.add_(add__106, add__108)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__109, parameter_234, parameter_235, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_95 = paddle.matmul(layer_norm_117, parameter_236, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__110 = paddle._C_ops.add_(matmul_95, parameter_237)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_14 = paddle._C_ops.gelu(add__110, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_96 = paddle.matmul(gelu_14, parameter_238, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__111 = paddle._C_ops.add_(matmul_96, parameter_239)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__112 = paddle._C_ops.add_(add__109, add__111)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__112, parameter_240, parameter_241, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_19 = paddle._C_ops.shape(paddle.cast(layer_norm_120, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(shape_19, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_97 = paddle.matmul(layer_norm_120, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__113 = paddle._C_ops.add_(matmul_97, parameter_243)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_72 = [slice_57, constant_23, constant_22, constant_8]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__113, combine_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_93 = paddle._C_ops.transpose(reshape__144, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_94 = paddle._C_ops.transpose(layer_norm_120, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_73 = [slice_57, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_94, combine_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf16) <- (-1x384x14x14xf16, 384x384x2x2xf16)
        conv2d_10 = paddle._C_ops.conv2d(reshape__146, parameter_244, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x7x7xf16) <- (-1x384x7x7xf16, 1x384x1x1xf16)
        add__114 = paddle._C_ops.add_(conv2d_10, parameter_245)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_74 = [slice_57, constant_21, constant_6]

        # pd_op.reshape_: (-1x384x49xf16, 0x-1x384x7x7xf16) <- (-1x384x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__114, combine_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_95 = paddle._C_ops.transpose(reshape__148, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x384xf16, -49xf32, -49xf32) <- (-1x49x384xf16, 384xf32, 384xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_95, parameter_246, parameter_247, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x384xf16, 384x768xf16)
        matmul_98 = paddle.matmul(layer_norm_123, parameter_248, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__115 = paddle._C_ops.add_(matmul_98, parameter_249)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_75 = [slice_57, constant_6, constant_14, constant_22, constant_8]

        # pd_op.reshape_: (-1x49x2x12x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__115, combine_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x12x49x32xf16) <- (-1x49x2x12x32xf16)
        transpose_96 = paddle._C_ops.transpose(reshape__150, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(transpose_96, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(transpose_96, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x12x49x32xf16)
        transpose_97 = paddle._C_ops.transpose(slice_58, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_99 = paddle.matmul(transpose_93, transpose_97, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__15 = paddle._C_ops.scale_(matmul_99, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__15 = paddle._C_ops.softmax_(scale__15, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_100 = paddle.matmul(softmax__15, slice_59, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_98 = paddle._C_ops.transpose(matmul_100, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_76 = [slice_57, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_98, combine_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_101 = paddle.matmul(reshape__152, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__116 = paddle._C_ops.add_(matmul_101, parameter_251)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__117 = paddle._C_ops.add_(add__112, add__116)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__117, parameter_252, parameter_253, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_102 = paddle.matmul(layer_norm_126, parameter_254, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__118 = paddle._C_ops.add_(matmul_102, parameter_255)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_15 = paddle._C_ops.gelu(add__118, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_103 = paddle.matmul(gelu_15, parameter_256, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__119 = paddle._C_ops.add_(matmul_103, parameter_257)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__120 = paddle._C_ops.add_(add__117, add__119)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__120, parameter_258, parameter_259, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_20 = paddle._C_ops.shape(paddle.cast(layer_norm_129, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(shape_20, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_77 = [slice_60, constant_14, constant_3, constant_14, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x7x2x7x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_129, combine_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x7x7x384xf16) <- (-1x2x7x2x7x384xf16)
        transpose_99 = paddle._C_ops.transpose(reshape__154, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x384xf16, 384x1152xf16)
        matmul_104 = paddle.matmul(transpose_99, parameter_260, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, 1152xf16)
        add__121 = paddle._C_ops.add_(matmul_104, parameter_261)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_78 = [slice_60, constant_15, constant_6, constant_7, constant_22, constant_8]

        # pd_op.reshape_: (-1x4x49x3x12x32xf16, 0x-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__121, combine_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x12x49x32xf16) <- (-1x4x49x3x12x32xf16)
        transpose_100 = paddle._C_ops.transpose(reshape__156, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(transpose_100, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(transpose_100, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(transpose_100, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x4x12x32x49xf16) <- (-1x4x12x49x32xf16)
        transpose_101 = paddle._C_ops.transpose(slice_62, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x4x12x49x49xf16) <- (-1x4x12x49x32xf16, -1x4x12x32x49xf16)
        matmul_105 = paddle.matmul(slice_61, transpose_101, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16, 1xf32)
        scale__16 = paddle._C_ops.scale_(matmul_105, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16)
        softmax__16 = paddle._C_ops.softmax_(scale__16, -1)

        # pd_op.matmul: (-1x4x12x49x32xf16) <- (-1x4x12x49x49xf16, -1x4x12x49x32xf16)
        matmul_106 = paddle.matmul(softmax__16, slice_63, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x4x49x12x32xf16) <- (-1x4x12x49x32xf16)
        transpose_102 = paddle._C_ops.transpose(matmul_106, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_79 = [slice_60, constant_14, constant_14, constant_3, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x2x7x7x384xf16, 0x-1x4x49x12x32xf16) <- (-1x4x49x12x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_102, combine_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x7x2x7x384xf16) <- (-1x2x2x7x7x384xf16)
        transpose_103 = paddle._C_ops.transpose(reshape__158, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_80 = [slice_60, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x2x7x2x7x384xf16) <- (-1x2x7x2x7x384xf16, [1xi32, 1xi32, 1xi32])
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_103, combine_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_107 = paddle.matmul(reshape__160, parameter_262, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__122 = paddle._C_ops.add_(matmul_107, parameter_263)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__123 = paddle._C_ops.add_(add__120, add__122)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__123, parameter_264, parameter_265, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_108 = paddle.matmul(layer_norm_132, parameter_266, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__124 = paddle._C_ops.add_(matmul_108, parameter_267)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_16 = paddle._C_ops.gelu(add__124, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_109 = paddle.matmul(gelu_16, parameter_268, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__125 = paddle._C_ops.add_(matmul_109, parameter_269)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__126 = paddle._C_ops.add_(add__123, add__125)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__126, parameter_270, parameter_271, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_21 = paddle._C_ops.shape(paddle.cast(layer_norm_135, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(shape_21, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_110 = paddle.matmul(layer_norm_135, parameter_272, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__127 = paddle._C_ops.add_(matmul_110, parameter_273)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_81 = [slice_64, constant_23, constant_22, constant_8]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__127, combine_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_104 = paddle._C_ops.transpose(reshape__162, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_105 = paddle._C_ops.transpose(layer_norm_135, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_82 = [slice_64, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_105, combine_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf16) <- (-1x384x14x14xf16, 384x384x2x2xf16)
        conv2d_11 = paddle._C_ops.conv2d(reshape__164, parameter_274, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x7x7xf16) <- (-1x384x7x7xf16, 1x384x1x1xf16)
        add__128 = paddle._C_ops.add_(conv2d_11, parameter_275)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_83 = [slice_64, constant_21, constant_6]

        # pd_op.reshape_: (-1x384x49xf16, 0x-1x384x7x7xf16) <- (-1x384x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__128, combine_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_106 = paddle._C_ops.transpose(reshape__166, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x384xf16, -49xf32, -49xf32) <- (-1x49x384xf16, 384xf32, 384xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_106, parameter_276, parameter_277, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x384xf16, 384x768xf16)
        matmul_111 = paddle.matmul(layer_norm_138, parameter_278, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__129 = paddle._C_ops.add_(matmul_111, parameter_279)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_84 = [slice_64, constant_6, constant_14, constant_22, constant_8]

        # pd_op.reshape_: (-1x49x2x12x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__129, combine_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x12x49x32xf16) <- (-1x49x2x12x32xf16)
        transpose_107 = paddle._C_ops.transpose(reshape__168, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(transpose_107, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(transpose_107, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x12x49x32xf16)
        transpose_108 = paddle._C_ops.transpose(slice_65, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_112 = paddle.matmul(transpose_104, transpose_108, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__17 = paddle._C_ops.scale_(matmul_112, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__17 = paddle._C_ops.softmax_(scale__17, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_113 = paddle.matmul(softmax__17, slice_66, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_109 = paddle._C_ops.transpose(matmul_113, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_85 = [slice_64, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_109, combine_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_114 = paddle.matmul(reshape__170, parameter_280, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__130 = paddle._C_ops.add_(matmul_114, parameter_281)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__131 = paddle._C_ops.add_(add__126, add__130)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__131, parameter_282, parameter_283, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_115 = paddle.matmul(layer_norm_141, parameter_284, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__132 = paddle._C_ops.add_(matmul_115, parameter_285)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_17 = paddle._C_ops.gelu(add__132, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_116 = paddle.matmul(gelu_17, parameter_286, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__133 = paddle._C_ops.add_(matmul_116, parameter_287)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__134 = paddle._C_ops.add_(add__131, add__133)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__134, parameter_288, parameter_289, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_22 = paddle._C_ops.shape(paddle.cast(layer_norm_144, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(shape_22, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_86 = [slice_67, constant_14, constant_3, constant_14, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x7x2x7x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__172, reshape__173 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_144, combine_86), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x7x7x384xf16) <- (-1x2x7x2x7x384xf16)
        transpose_110 = paddle._C_ops.transpose(reshape__172, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x384xf16, 384x1152xf16)
        matmul_117 = paddle.matmul(transpose_110, parameter_290, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, 1152xf16)
        add__135 = paddle._C_ops.add_(matmul_117, parameter_291)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_87 = [slice_67, constant_15, constant_6, constant_7, constant_22, constant_8]

        # pd_op.reshape_: (-1x4x49x3x12x32xf16, 0x-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__174, reshape__175 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__135, combine_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x12x49x32xf16) <- (-1x4x49x3x12x32xf16)
        transpose_111 = paddle._C_ops.transpose(reshape__174, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(transpose_111, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(transpose_111, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(transpose_111, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x4x12x32x49xf16) <- (-1x4x12x49x32xf16)
        transpose_112 = paddle._C_ops.transpose(slice_69, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x4x12x49x49xf16) <- (-1x4x12x49x32xf16, -1x4x12x32x49xf16)
        matmul_118 = paddle.matmul(slice_68, transpose_112, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16, 1xf32)
        scale__18 = paddle._C_ops.scale_(matmul_118, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16)
        softmax__18 = paddle._C_ops.softmax_(scale__18, -1)

        # pd_op.matmul: (-1x4x12x49x32xf16) <- (-1x4x12x49x49xf16, -1x4x12x49x32xf16)
        matmul_119 = paddle.matmul(softmax__18, slice_70, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x4x49x12x32xf16) <- (-1x4x12x49x32xf16)
        transpose_113 = paddle._C_ops.transpose(matmul_119, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_88 = [slice_67, constant_14, constant_14, constant_3, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x2x7x7x384xf16, 0x-1x4x49x12x32xf16) <- (-1x4x49x12x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__176, reshape__177 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_113, combine_88), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x7x2x7x384xf16) <- (-1x2x2x7x7x384xf16)
        transpose_114 = paddle._C_ops.transpose(reshape__176, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_89 = [slice_67, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x2x7x2x7x384xf16) <- (-1x2x7x2x7x384xf16, [1xi32, 1xi32, 1xi32])
        reshape__178, reshape__179 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_114, combine_89), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_120 = paddle.matmul(reshape__178, parameter_292, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__136 = paddle._C_ops.add_(matmul_120, parameter_293)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__137 = paddle._C_ops.add_(add__134, add__136)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_147, layer_norm_148, layer_norm_149 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__137, parameter_294, parameter_295, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_121 = paddle.matmul(layer_norm_147, parameter_296, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__138 = paddle._C_ops.add_(matmul_121, parameter_297)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_18 = paddle._C_ops.gelu(add__138, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_122 = paddle.matmul(gelu_18, parameter_298, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__139 = paddle._C_ops.add_(matmul_122, parameter_299)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__140 = paddle._C_ops.add_(add__137, add__139)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_150, layer_norm_151, layer_norm_152 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__140, parameter_300, parameter_301, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_23 = paddle._C_ops.shape(paddle.cast(layer_norm_150, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(shape_23, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_123 = paddle.matmul(layer_norm_150, parameter_302, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__141 = paddle._C_ops.add_(matmul_123, parameter_303)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_90 = [slice_71, constant_23, constant_22, constant_8]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__180, reshape__181 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__141, combine_90), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_115 = paddle._C_ops.transpose(reshape__180, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_116 = paddle._C_ops.transpose(layer_norm_150, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_91 = [slice_71, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__182, reshape__183 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_116, combine_91), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf16) <- (-1x384x14x14xf16, 384x384x2x2xf16)
        conv2d_12 = paddle._C_ops.conv2d(reshape__182, parameter_304, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x7x7xf16) <- (-1x384x7x7xf16, 1x384x1x1xf16)
        add__142 = paddle._C_ops.add_(conv2d_12, parameter_305)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_92 = [slice_71, constant_21, constant_6]

        # pd_op.reshape_: (-1x384x49xf16, 0x-1x384x7x7xf16) <- (-1x384x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__184, reshape__185 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__142, combine_92), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_117 = paddle._C_ops.transpose(reshape__184, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x384xf16, -49xf32, -49xf32) <- (-1x49x384xf16, 384xf32, 384xf32)
        layer_norm_153, layer_norm_154, layer_norm_155 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_117, parameter_306, parameter_307, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x384xf16, 384x768xf16)
        matmul_124 = paddle.matmul(layer_norm_153, parameter_308, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__143 = paddle._C_ops.add_(matmul_124, parameter_309)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_93 = [slice_71, constant_6, constant_14, constant_22, constant_8]

        # pd_op.reshape_: (-1x49x2x12x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__186, reshape__187 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__143, combine_93), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x12x49x32xf16) <- (-1x49x2x12x32xf16)
        transpose_118 = paddle._C_ops.transpose(reshape__186, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(transpose_118, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(transpose_118, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x12x49x32xf16)
        transpose_119 = paddle._C_ops.transpose(slice_72, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_125 = paddle.matmul(transpose_115, transpose_119, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__19 = paddle._C_ops.scale_(matmul_125, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__19 = paddle._C_ops.softmax_(scale__19, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_126 = paddle.matmul(softmax__19, slice_73, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_120 = paddle._C_ops.transpose(matmul_126, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_94 = [slice_71, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__188, reshape__189 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_120, combine_94), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_127 = paddle.matmul(reshape__188, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__144 = paddle._C_ops.add_(matmul_127, parameter_311)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__145 = paddle._C_ops.add_(add__140, add__144)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_156, layer_norm_157, layer_norm_158 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__145, parameter_312, parameter_313, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_128 = paddle.matmul(layer_norm_156, parameter_314, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__146 = paddle._C_ops.add_(matmul_128, parameter_315)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_19 = paddle._C_ops.gelu(add__146, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_129 = paddle.matmul(gelu_19, parameter_316, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__147 = paddle._C_ops.add_(matmul_129, parameter_317)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__148 = paddle._C_ops.add_(add__145, add__147)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_159, layer_norm_160, layer_norm_161 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__148, parameter_318, parameter_319, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_24 = paddle._C_ops.shape(paddle.cast(layer_norm_159, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(shape_24, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_95 = [slice_74, constant_14, constant_3, constant_14, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x7x2x7x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__190, reshape__191 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_159, combine_95), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x7x7x384xf16) <- (-1x2x7x2x7x384xf16)
        transpose_121 = paddle._C_ops.transpose(reshape__190, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x384xf16, 384x1152xf16)
        matmul_130 = paddle.matmul(transpose_121, parameter_320, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, 1152xf16)
        add__149 = paddle._C_ops.add_(matmul_130, parameter_321)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_96 = [slice_74, constant_15, constant_6, constant_7, constant_22, constant_8]

        # pd_op.reshape_: (-1x4x49x3x12x32xf16, 0x-1x2x2x7x7x1152xf16) <- (-1x2x2x7x7x1152xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__192, reshape__193 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__149, combine_96), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x12x49x32xf16) <- (-1x4x49x3x12x32xf16)
        transpose_122 = paddle._C_ops.transpose(reshape__192, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(transpose_122, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(transpose_122, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x4x12x49x32xf16) <- (3x-1x4x12x49x32xf16, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(transpose_122, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x4x12x32x49xf16) <- (-1x4x12x49x32xf16)
        transpose_123 = paddle._C_ops.transpose(slice_76, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x4x12x49x49xf16) <- (-1x4x12x49x32xf16, -1x4x12x32x49xf16)
        matmul_131 = paddle.matmul(slice_75, transpose_123, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16, 1xf32)
        scale__20 = paddle._C_ops.scale_(matmul_131, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x4x12x49x49xf16) <- (-1x4x12x49x49xf16)
        softmax__20 = paddle._C_ops.softmax_(scale__20, -1)

        # pd_op.matmul: (-1x4x12x49x32xf16) <- (-1x4x12x49x49xf16, -1x4x12x49x32xf16)
        matmul_132 = paddle.matmul(softmax__20, slice_77, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x4x49x12x32xf16) <- (-1x4x12x49x32xf16)
        transpose_124 = paddle._C_ops.transpose(matmul_132, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_97 = [slice_74, constant_14, constant_14, constant_3, constant_3, constant_21]

        # pd_op.reshape_: (-1x2x2x7x7x384xf16, 0x-1x4x49x12x32xf16) <- (-1x4x49x12x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__194, reshape__195 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_124, combine_97), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x7x2x7x384xf16) <- (-1x2x2x7x7x384xf16)
        transpose_125 = paddle._C_ops.transpose(reshape__194, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_98 = [slice_74, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x2x7x2x7x384xf16) <- (-1x2x7x2x7x384xf16, [1xi32, 1xi32, 1xi32])
        reshape__196, reshape__197 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_125, combine_98), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_133 = paddle.matmul(reshape__196, parameter_322, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__150 = paddle._C_ops.add_(matmul_133, parameter_323)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__151 = paddle._C_ops.add_(add__148, add__150)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_162, layer_norm_163, layer_norm_164 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__151, parameter_324, parameter_325, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_134 = paddle.matmul(layer_norm_162, parameter_326, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__152 = paddle._C_ops.add_(matmul_134, parameter_327)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_20 = paddle._C_ops.gelu(add__152, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_135 = paddle.matmul(gelu_20, parameter_328, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__153 = paddle._C_ops.add_(matmul_135, parameter_329)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__154 = paddle._C_ops.add_(add__151, add__153)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_165, layer_norm_166, layer_norm_167 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__154, parameter_330, parameter_331, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_25 = paddle._C_ops.shape(paddle.cast(layer_norm_165, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(shape_25, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_136 = paddle.matmul(layer_norm_165, parameter_332, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__155 = paddle._C_ops.add_(matmul_136, parameter_333)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_99 = [slice_78, constant_23, constant_22, constant_8]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__198, reshape__199 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__155, combine_99), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_126 = paddle._C_ops.transpose(reshape__198, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_127 = paddle._C_ops.transpose(layer_norm_165, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_100 = [slice_78, constant_21, constant_24, constant_24]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__200, reshape__201 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_127, combine_100), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf16) <- (-1x384x14x14xf16, 384x384x2x2xf16)
        conv2d_13 = paddle._C_ops.conv2d(reshape__200, parameter_334, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x7x7xf16) <- (-1x384x7x7xf16, 1x384x1x1xf16)
        add__156 = paddle._C_ops.add_(conv2d_13, parameter_335)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_101 = [slice_78, constant_21, constant_6]

        # pd_op.reshape_: (-1x384x49xf16, 0x-1x384x7x7xf16) <- (-1x384x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__202, reshape__203 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__156, combine_101), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_128 = paddle._C_ops.transpose(reshape__202, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x384xf16, -49xf32, -49xf32) <- (-1x49x384xf16, 384xf32, 384xf32)
        layer_norm_168, layer_norm_169, layer_norm_170 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_128, parameter_336, parameter_337, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x384xf16, 384x768xf16)
        matmul_137 = paddle.matmul(layer_norm_168, parameter_338, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__157 = paddle._C_ops.add_(matmul_137, parameter_339)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_102 = [slice_78, constant_6, constant_14, constant_22, constant_8]

        # pd_op.reshape_: (-1x49x2x12x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__204, reshape__205 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__157, combine_102), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x12x49x32xf16) <- (-1x49x2x12x32xf16)
        transpose_129 = paddle._C_ops.transpose(reshape__204, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(transpose_129, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x12x49x32xf16) <- (2x-1x12x49x32xf16, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(transpose_129, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x12x49x32xf16)
        transpose_130 = paddle._C_ops.transpose(slice_79, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_138 = paddle.matmul(transpose_126, transpose_130, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__21 = paddle._C_ops.scale_(matmul_138, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__21 = paddle._C_ops.softmax_(scale__21, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_139 = paddle.matmul(softmax__21, slice_80, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_131 = paddle._C_ops.transpose(matmul_139, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_103 = [slice_78, constant_23, constant_21]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__206, reshape__207 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_131, combine_103), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_140 = paddle.matmul(reshape__206, parameter_340, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__158 = paddle._C_ops.add_(matmul_140, parameter_341)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__159 = paddle._C_ops.add_(add__154, add__158)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_171, layer_norm_172, layer_norm_173 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__159, parameter_342, parameter_343, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1536xf16) <- (-1x196x384xf16, 384x1536xf16)
        matmul_141 = paddle.matmul(layer_norm_171, parameter_344, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1536xf16) <- (-1x196x1536xf16, 1536xf16)
        add__160 = paddle._C_ops.add_(matmul_141, parameter_345)

        # pd_op.gelu: (-1x196x1536xf16) <- (-1x196x1536xf16)
        gelu_21 = paddle._C_ops.gelu(add__160, False)

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x1536xf16, 1536x384xf16)
        matmul_142 = paddle.matmul(gelu_21, parameter_346, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__161 = paddle._C_ops.add_(matmul_142, parameter_347)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__162 = paddle._C_ops.add_(add__159, add__161)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_104 = [slice_0, constant_24, constant_24, constant_21]

        # pd_op.reshape_: (-1x14x14x384xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__208, reshape__209 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__162, combine_104), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x384x14x14xf16) <- (-1x14x14x384xf16)
        transpose_132 = paddle._C_ops.transpose(reshape__208, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x768x7x7xf16) <- (-1x384x14x14xf16, 768x384x2x2xf16)
        conv2d_14 = paddle._C_ops.conv2d(transpose_132, parameter_348, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, 1x768x1x1xf16)
        add__163 = paddle._C_ops.add_(conv2d_14, parameter_349)

        # pd_op.flatten_: (-1x768x49xf16, None) <- (-1x768x7x7xf16)
        flatten__12, flatten__13 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__163, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x768xf16) <- (-1x768x49xf16)
        transpose_133 = paddle._C_ops.transpose(flatten__12, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x768xf16, -49xf32, -49xf32) <- (-1x49x768xf16, 768xf32, 768xf32)
        layer_norm_174, layer_norm_175, layer_norm_176 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_133, parameter_350, parameter_351, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x49x768xf16, -49xf32, -49xf32) <- (-1x49x768xf16, 768xf32, 768xf32)
        layer_norm_177, layer_norm_178, layer_norm_179 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_174, parameter_352, parameter_353, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x768xf16)
        shape_26 = paddle._C_ops.shape(paddle.cast(layer_norm_177, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(shape_26, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_105 = [slice_81, constant_25, constant_3, constant_25, constant_3, constant_26]

        # pd_op.reshape_: (-1x1x7x1x7x768xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__210, reshape__211 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_177, combine_105), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x1x7x7x768xf16) <- (-1x1x7x1x7x768xf16)
        transpose_134 = paddle._C_ops.transpose(reshape__210, [0, 1, 3, 2, 4, 5])

        # pd_op.matmul: (-1x1x1x7x7x2304xf16) <- (-1x1x1x7x7x768xf16, 768x2304xf16)
        matmul_143 = paddle.matmul(transpose_134, parameter_354, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1x1x7x7x2304xf16) <- (-1x1x1x7x7x2304xf16, 2304xf16)
        add__164 = paddle._C_ops.add_(matmul_143, parameter_355)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_106 = [slice_81, constant_25, constant_6, constant_7, constant_27, constant_8]

        # pd_op.reshape_: (-1x1x49x3x24x32xf16, 0x-1x1x1x7x7x2304xf16) <- (-1x1x1x7x7x2304xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__212, reshape__213 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__164, combine_106), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x1x24x49x32xf16) <- (-1x1x49x3x24x32xf16)
        transpose_135 = paddle._C_ops.transpose(reshape__212, [3, 0, 1, 4, 2, 5])

        # pd_op.slice: (-1x1x24x49x32xf16) <- (3x-1x1x24x49x32xf16, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(transpose_135, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x1x24x49x32xf16) <- (3x-1x1x24x49x32xf16, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(transpose_135, [0], constant_1, constant_9, [1], [0])

        # pd_op.slice: (-1x1x24x49x32xf16) <- (3x-1x1x24x49x32xf16, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(transpose_135, [0], constant_9, constant_10, [1], [0])

        # pd_op.transpose: (-1x1x24x32x49xf16) <- (-1x1x24x49x32xf16)
        transpose_136 = paddle._C_ops.transpose(slice_83, [0, 1, 2, 4, 3])

        # pd_op.matmul: (-1x1x24x49x49xf16) <- (-1x1x24x49x32xf16, -1x1x24x32x49xf16)
        matmul_144 = paddle.matmul(slice_82, transpose_136, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x1x24x49x49xf16) <- (-1x1x24x49x49xf16, 1xf32)
        scale__22 = paddle._C_ops.scale_(matmul_144, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x1x24x49x49xf16) <- (-1x1x24x49x49xf16)
        softmax__22 = paddle._C_ops.softmax_(scale__22, -1)

        # pd_op.matmul: (-1x1x24x49x32xf16) <- (-1x1x24x49x49xf16, -1x1x24x49x32xf16)
        matmul_145 = paddle.matmul(softmax__22, slice_84, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x1x49x24x32xf16) <- (-1x1x24x49x32xf16)
        transpose_137 = paddle._C_ops.transpose(matmul_145, [0, 1, 3, 2, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_107 = [slice_81, constant_25, constant_25, constant_3, constant_3, constant_26]

        # pd_op.reshape_: (-1x1x1x7x7x768xf16, 0x-1x1x49x24x32xf16) <- (-1x1x49x24x32xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__214, reshape__215 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_137, combine_107), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x7x1x7x768xf16) <- (-1x1x1x7x7x768xf16)
        transpose_138 = paddle._C_ops.transpose(reshape__214, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_108 = [slice_81, constant_6, constant_26]

        # pd_op.reshape_: (-1x49x768xf16, 0x-1x1x7x1x7x768xf16) <- (-1x1x7x1x7x768xf16, [1xi32, 1xi32, 1xi32])
        reshape__216, reshape__217 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_138, combine_108), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x768xf16, 768x768xf16)
        matmul_146 = paddle.matmul(reshape__216, parameter_356, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__165 = paddle._C_ops.add_(matmul_146, parameter_357)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, -1x49x768xf16)
        add__166 = paddle._C_ops.add_(layer_norm_174, add__165)

        # pd_op.layer_norm: (-1x49x768xf16, -49xf32, -49xf32) <- (-1x49x768xf16, 768xf32, 768xf32)
        layer_norm_180, layer_norm_181, layer_norm_182 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__166, parameter_358, parameter_359, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x3072xf16) <- (-1x49x768xf16, 768x3072xf16)
        matmul_147 = paddle.matmul(layer_norm_180, parameter_360, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x3072xf16) <- (-1x49x3072xf16, 3072xf16)
        add__167 = paddle._C_ops.add_(matmul_147, parameter_361)

        # pd_op.gelu: (-1x49x3072xf16) <- (-1x49x3072xf16)
        gelu_22 = paddle._C_ops.gelu(add__167, False)

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x3072xf16, 3072x768xf16)
        matmul_148 = paddle.matmul(gelu_22, parameter_362, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__168 = paddle._C_ops.add_(matmul_148, parameter_363)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, -1x49x768xf16)
        add__169 = paddle._C_ops.add_(add__166, add__168)

        # pd_op.shape: (3xi32) <- (-1x49x768xf16)
        shape_27 = paddle._C_ops.shape(paddle.cast(add__169, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(shape_27, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x768x49xf16) <- (-1x49x768xf16)
        transpose_139 = paddle._C_ops.transpose(add__169, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_109 = [slice_85, constant_26, constant_3, constant_3]

        # pd_op.reshape_: (-1x768x7x7xf16, 0x-1x768x49xf16) <- (-1x768x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__218, reshape__219 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_139, combine_109), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x768x7x7xf16) <- (-1x768x7x7xf16, 768x1x3x3xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(reshape__218, parameter_364, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, 1x768x1x1xf16)
        add__170 = paddle._C_ops.add_(depthwise_conv2d_3, parameter_365)

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, -1x768x7x7xf16)
        add__171 = paddle._C_ops.add_(add__170, reshape__218)

        # pd_op.flatten_: (-1x768x49xf16, None) <- (-1x768x7x7xf16)
        flatten__14, flatten__15 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__171, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x768xf16) <- (-1x768x49xf16)
        transpose_140 = paddle._C_ops.transpose(flatten__14, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x768xf16, -49xf32, -49xf32) <- (-1x49x768xf16, 768xf32, 768xf32)
        layer_norm_183, layer_norm_184, layer_norm_185 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_140, parameter_366, parameter_367, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x768xf16)
        shape_28 = paddle._C_ops.shape(paddle.cast(layer_norm_183, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(shape_28, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x768xf16, 768x768xf16)
        matmul_149 = paddle.matmul(layer_norm_183, parameter_368, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__172 = paddle._C_ops.add_(matmul_149, parameter_369)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_110 = [slice_86, constant_6, constant_27, constant_8]

        # pd_op.reshape_: (-1x49x24x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__220, reshape__221 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__172, combine_110), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x24x49x32xf16) <- (-1x49x24x32xf16)
        transpose_141 = paddle._C_ops.transpose(reshape__220, [0, 2, 1, 3])

        # pd_op.matmul: (-1x49x1536xf16) <- (-1x49x768xf16, 768x1536xf16)
        matmul_150 = paddle.matmul(layer_norm_183, parameter_370, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf16) <- (-1x49x1536xf16, 1536xf16)
        add__173 = paddle._C_ops.add_(matmul_150, parameter_371)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_111 = [slice_86, constant_6, constant_14, constant_27, constant_8]

        # pd_op.reshape_: (-1x49x2x24x32xf16, 0x-1x49x1536xf16) <- (-1x49x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__222, reshape__223 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__173, combine_111), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x24x49x32xf16) <- (-1x49x2x24x32xf16)
        transpose_142 = paddle._C_ops.transpose(reshape__222, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x24x49x32xf16) <- (2x-1x24x49x32xf16, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(transpose_142, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x24x49x32xf16) <- (2x-1x24x49x32xf16, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(transpose_142, [0], constant_1, constant_9, [1], [0])

        # pd_op.transpose: (-1x24x32x49xf16) <- (-1x24x49x32xf16)
        transpose_143 = paddle._C_ops.transpose(slice_87, [0, 1, 3, 2])

        # pd_op.matmul: (-1x24x49x49xf16) <- (-1x24x49x32xf16, -1x24x32x49xf16)
        matmul_151 = paddle.matmul(transpose_141, transpose_143, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x24x49x49xf16) <- (-1x24x49x49xf16, 1xf32)
        scale__23 = paddle._C_ops.scale_(matmul_151, constant_11, float('0'), True)

        # pd_op.softmax_: (-1x24x49x49xf16) <- (-1x24x49x49xf16)
        softmax__23 = paddle._C_ops.softmax_(scale__23, -1)

        # pd_op.matmul: (-1x24x49x32xf16) <- (-1x24x49x49xf16, -1x24x49x32xf16)
        matmul_152 = paddle.matmul(softmax__23, slice_88, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x24x32xf16) <- (-1x24x49x32xf16)
        transpose_144 = paddle._C_ops.transpose(matmul_152, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_112 = [slice_86, constant_6, constant_26]

        # pd_op.reshape_: (-1x49x768xf16, 0x-1x49x24x32xf16) <- (-1x49x24x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__224, reshape__225 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_144, combine_112), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x768xf16, 768x768xf16)
        matmul_153 = paddle.matmul(reshape__224, parameter_372, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__174 = paddle._C_ops.add_(matmul_153, parameter_373)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, -1x49x768xf16)
        add__175 = paddle._C_ops.add_(transpose_140, add__174)

        # pd_op.layer_norm: (-1x49x768xf16, -49xf32, -49xf32) <- (-1x49x768xf16, 768xf32, 768xf32)
        layer_norm_186, layer_norm_187, layer_norm_188 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__175, parameter_374, parameter_375, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x3072xf16) <- (-1x49x768xf16, 768x3072xf16)
        matmul_154 = paddle.matmul(layer_norm_186, parameter_376, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x3072xf16) <- (-1x49x3072xf16, 3072xf16)
        add__176 = paddle._C_ops.add_(matmul_154, parameter_377)

        # pd_op.gelu: (-1x49x3072xf16) <- (-1x49x3072xf16)
        gelu_23 = paddle._C_ops.gelu(add__176, False)

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x3072xf16, 3072x768xf16)
        matmul_155 = paddle.matmul(gelu_23, parameter_378, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__177 = paddle._C_ops.add_(matmul_155, parameter_379)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, -1x49x768xf16)
        add__178 = paddle._C_ops.add_(add__175, add__177)

        # pd_op.layer_norm: (-1x49x768xf16, -49xf32, -49xf32) <- (-1x49x768xf16, 768xf32, 768xf32)
        layer_norm_189, layer_norm_190, layer_norm_191 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__178, parameter_380, parameter_381, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.mean: (-1x768xf16) <- (-1x49x768xf16)
        mean_0 = paddle._C_ops.mean(layer_norm_189, [1], False)

        # pd_op.matmul: (-1x1000xf16) <- (-1x768xf16, 768x1000xf16)
        matmul_156 = paddle.matmul(mean_0, parameter_382, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__179 = paddle._C_ops.add_(matmul_156, parameter_383)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__24 = paddle._C_ops.softmax_(add__179, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__24, paddle.float32)
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

    def forward(self, parameter_365, constant_27, constant_26, constant_25, parameter_349, parameter_335, parameter_305, parameter_275, parameter_245, parameter_215, parameter_185, parameter_155, parameter_125, parameter_95, parameter_89, constant_24, constant_23, constant_22, constant_21, parameter_73, parameter_59, parameter_53, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, parameter_37, constant_14, parameter_23, parameter_17, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, parameter_1, constant_1, constant_0, parameter_0, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_19, parameter_18, parameter_20, parameter_21, parameter_22, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_39, parameter_38, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_55, parameter_54, parameter_56, parameter_57, parameter_58, parameter_61, parameter_60, parameter_62, parameter_63, parameter_64, parameter_65, parameter_67, parameter_66, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_75, parameter_74, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_91, parameter_90, parameter_92, parameter_93, parameter_94, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_101, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_113, parameter_115, parameter_114, parameter_116, parameter_117, parameter_118, parameter_119, parameter_121, parameter_120, parameter_122, parameter_123, parameter_124, parameter_127, parameter_126, parameter_128, parameter_129, parameter_130, parameter_131, parameter_133, parameter_132, parameter_134, parameter_135, parameter_136, parameter_137, parameter_139, parameter_138, parameter_140, parameter_141, parameter_142, parameter_143, parameter_145, parameter_144, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_157, parameter_156, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_171, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, parameter_178, parameter_179, parameter_181, parameter_180, parameter_182, parameter_183, parameter_184, parameter_187, parameter_186, parameter_188, parameter_189, parameter_190, parameter_191, parameter_193, parameter_192, parameter_194, parameter_195, parameter_196, parameter_197, parameter_199, parameter_198, parameter_200, parameter_201, parameter_202, parameter_203, parameter_205, parameter_204, parameter_206, parameter_207, parameter_208, parameter_209, parameter_211, parameter_210, parameter_212, parameter_213, parameter_214, parameter_217, parameter_216, parameter_218, parameter_219, parameter_220, parameter_221, parameter_223, parameter_222, parameter_224, parameter_225, parameter_226, parameter_227, parameter_229, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_235, parameter_234, parameter_236, parameter_237, parameter_238, parameter_239, parameter_241, parameter_240, parameter_242, parameter_243, parameter_244, parameter_247, parameter_246, parameter_248, parameter_249, parameter_250, parameter_251, parameter_253, parameter_252, parameter_254, parameter_255, parameter_256, parameter_257, parameter_259, parameter_258, parameter_260, parameter_261, parameter_262, parameter_263, parameter_265, parameter_264, parameter_266, parameter_267, parameter_268, parameter_269, parameter_271, parameter_270, parameter_272, parameter_273, parameter_274, parameter_277, parameter_276, parameter_278, parameter_279, parameter_280, parameter_281, parameter_283, parameter_282, parameter_284, parameter_285, parameter_286, parameter_287, parameter_289, parameter_288, parameter_290, parameter_291, parameter_292, parameter_293, parameter_295, parameter_294, parameter_296, parameter_297, parameter_298, parameter_299, parameter_301, parameter_300, parameter_302, parameter_303, parameter_304, parameter_307, parameter_306, parameter_308, parameter_309, parameter_310, parameter_311, parameter_313, parameter_312, parameter_314, parameter_315, parameter_316, parameter_317, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_345, parameter_346, parameter_347, parameter_348, parameter_351, parameter_350, parameter_353, parameter_352, parameter_354, parameter_355, parameter_356, parameter_357, parameter_359, parameter_358, parameter_360, parameter_361, parameter_362, parameter_363, parameter_364, parameter_367, parameter_366, parameter_368, parameter_369, parameter_370, parameter_371, parameter_372, parameter_373, parameter_375, parameter_374, parameter_376, parameter_377, parameter_378, parameter_379, parameter_381, parameter_380, parameter_382, parameter_383, feed_0):
        return self.builtin_module_2502_0_0(parameter_365, constant_27, constant_26, constant_25, parameter_349, parameter_335, parameter_305, parameter_275, parameter_245, parameter_215, parameter_185, parameter_155, parameter_125, parameter_95, parameter_89, constant_24, constant_23, constant_22, constant_21, parameter_73, parameter_59, parameter_53, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, parameter_37, constant_14, parameter_23, parameter_17, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, parameter_1, constant_1, constant_0, parameter_0, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_19, parameter_18, parameter_20, parameter_21, parameter_22, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_39, parameter_38, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_55, parameter_54, parameter_56, parameter_57, parameter_58, parameter_61, parameter_60, parameter_62, parameter_63, parameter_64, parameter_65, parameter_67, parameter_66, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_75, parameter_74, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_91, parameter_90, parameter_92, parameter_93, parameter_94, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_101, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_113, parameter_115, parameter_114, parameter_116, parameter_117, parameter_118, parameter_119, parameter_121, parameter_120, parameter_122, parameter_123, parameter_124, parameter_127, parameter_126, parameter_128, parameter_129, parameter_130, parameter_131, parameter_133, parameter_132, parameter_134, parameter_135, parameter_136, parameter_137, parameter_139, parameter_138, parameter_140, parameter_141, parameter_142, parameter_143, parameter_145, parameter_144, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_157, parameter_156, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_171, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, parameter_178, parameter_179, parameter_181, parameter_180, parameter_182, parameter_183, parameter_184, parameter_187, parameter_186, parameter_188, parameter_189, parameter_190, parameter_191, parameter_193, parameter_192, parameter_194, parameter_195, parameter_196, parameter_197, parameter_199, parameter_198, parameter_200, parameter_201, parameter_202, parameter_203, parameter_205, parameter_204, parameter_206, parameter_207, parameter_208, parameter_209, parameter_211, parameter_210, parameter_212, parameter_213, parameter_214, parameter_217, parameter_216, parameter_218, parameter_219, parameter_220, parameter_221, parameter_223, parameter_222, parameter_224, parameter_225, parameter_226, parameter_227, parameter_229, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_235, parameter_234, parameter_236, parameter_237, parameter_238, parameter_239, parameter_241, parameter_240, parameter_242, parameter_243, parameter_244, parameter_247, parameter_246, parameter_248, parameter_249, parameter_250, parameter_251, parameter_253, parameter_252, parameter_254, parameter_255, parameter_256, parameter_257, parameter_259, parameter_258, parameter_260, parameter_261, parameter_262, parameter_263, parameter_265, parameter_264, parameter_266, parameter_267, parameter_268, parameter_269, parameter_271, parameter_270, parameter_272, parameter_273, parameter_274, parameter_277, parameter_276, parameter_278, parameter_279, parameter_280, parameter_281, parameter_283, parameter_282, parameter_284, parameter_285, parameter_286, parameter_287, parameter_289, parameter_288, parameter_290, parameter_291, parameter_292, parameter_293, parameter_295, parameter_294, parameter_296, parameter_297, parameter_298, parameter_299, parameter_301, parameter_300, parameter_302, parameter_303, parameter_304, parameter_307, parameter_306, parameter_308, parameter_309, parameter_310, parameter_311, parameter_313, parameter_312, parameter_314, parameter_315, parameter_316, parameter_317, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_345, parameter_346, parameter_347, parameter_348, parameter_351, parameter_350, parameter_353, parameter_352, parameter_354, parameter_355, parameter_356, parameter_357, parameter_359, parameter_358, parameter_360, parameter_361, parameter_362, parameter_363, parameter_364, parameter_367, parameter_366, parameter_368, parameter_369, parameter_370, parameter_371, parameter_372, parameter_373, parameter_375, parameter_374, parameter_376, parameter_377, parameter_378, parameter_379, parameter_381, parameter_380, parameter_382, parameter_383, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2502_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_365
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_27
            paddle.to_tensor([24], dtype='int32').reshape([1]),
            # constant_26
            paddle.to_tensor([768], dtype='int32').reshape([1]),
            # constant_25
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # parameter_349
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_275
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_245
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_185
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_24
            paddle.to_tensor([14], dtype='int32').reshape([1]),
            # constant_23
            paddle.to_tensor([196], dtype='int32').reshape([1]),
            # constant_22
            paddle.to_tensor([12], dtype='int32').reshape([1]),
            # constant_21
            paddle.to_tensor([384], dtype='int32').reshape([1]),
            # parameter_73
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_20
            paddle.to_tensor([28], dtype='int32').reshape([1]),
            # constant_19
            paddle.to_tensor([784], dtype='int32').reshape([1]),
            # constant_18
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            # constant_17
            paddle.to_tensor([16], dtype='int32').reshape([1]),
            # constant_16
            paddle.to_tensor([192], dtype='int32').reshape([1]),
            # constant_15
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            # parameter_37
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_14
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            # parameter_23
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_17
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_13
            paddle.to_tensor([56], dtype='int32').reshape([1]),
            # constant_12
            paddle.to_tensor([3136], dtype='int32').reshape([1]),
            # constant_11
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_10
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            # constant_9
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_8
            paddle.to_tensor([32], dtype='int32').reshape([1]),
            # constant_7
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            # constant_6
            paddle.to_tensor([49], dtype='int32').reshape([1]),
            # constant_5
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([96], dtype='int32').reshape([1]),
            # constant_3
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            # constant_2
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            # parameter_1
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_0
            paddle.uniform([96, 3, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_3
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([96, 288], dtype='float16', min=0, max=0.5),
            # parameter_7
            paddle.uniform([288], dtype='float16', min=0, max=0.5),
            # parameter_8
            paddle.uniform([96, 96], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([96, 384], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([384, 96], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_16
            paddle.uniform([96, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([96, 96], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([96, 96, 8, 8], dtype='float16', min=0, max=0.5),
            # parameter_25
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([96, 192], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([96, 96], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([96, 384], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([384, 96], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_36
            paddle.uniform([192, 96, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([192, 576], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([576], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_47
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([192, 768], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([768, 192], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([192, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([192, 192, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([192, 384], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([192, 768], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_70
            paddle.uniform([768, 192], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([384, 192, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([384, 1152], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([384, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([384, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_100
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([384, 1152], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_112
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([384, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([384, 1152], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_142
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([384, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_160
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([384, 1152], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_175
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_181
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([384, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_187
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_195
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_197
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([384, 1152], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_202
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([384, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_220
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_226
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([384, 1152], dtype='float16', min=0, max=0.5),
            # parameter_231
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_232
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_233
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_235
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_237
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([384, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_247
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_250
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_255
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_257
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([384, 1152], dtype='float16', min=0, max=0.5),
            # parameter_261
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_262
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_265
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_267
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_273
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_274
            paddle.uniform([384, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_280
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_281
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_283
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_285
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_286
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_287
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([384, 1152], dtype='float16', min=0, max=0.5),
            # parameter_291
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_295
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_297
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_299
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_301
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_303
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([384, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_307
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_309
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_310
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_313
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_315
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_316
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_317
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_319
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([384, 1152], dtype='float16', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_322
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_323
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_325
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_327
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_328
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_329
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_331
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_333
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([384, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_337
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_340
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_341
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_343
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_345
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_346
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_347
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_348
            paddle.uniform([768, 384, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_351
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_355
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_356
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_357
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_359
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_361
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_362
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_363
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([768, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_367
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_369
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_370
            paddle.uniform([768, 1536], dtype='float16', min=0, max=0.5),
            # parameter_371
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_372
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_373
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_375
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_377
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_378
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_379
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_381
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([768, 1000], dtype='float16', min=0, max=0.5),
            # parameter_383
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_365
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # constant_27
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_26
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_25
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_349
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_335
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_275
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_245
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_185
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # constant_24
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_23
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_22
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_21
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_73
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float16'),
            # constant_20
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_19
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_18
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_17
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_15
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_37
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float16'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_23
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float16'),
            # parameter_17
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float16'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float16'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[96, 3, 4, 4], dtype='float16'),
            # parameter_3
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[96, 288], dtype='float16'),
            # parameter_7
            paddle.static.InputSpec(shape=[288], dtype='float16'),
            # parameter_8
            paddle.static.InputSpec(shape=[96, 96], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[96, 384], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[384, 96], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_16
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[96, 96], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[96, 96, 8, 8], dtype='float16'),
            # parameter_25
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[96, 192], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[96, 96], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[96, 384], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[384, 96], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_36
            paddle.static.InputSpec(shape=[192, 96, 2, 2], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[192, 576], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[576], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_47
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[192, 768], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[768, 192], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[192, 192, 4, 4], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[192, 384], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[192, 768], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_70
            paddle.static.InputSpec(shape=[768, 192], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[384, 192, 2, 2], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[384, 1152], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_100
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[384, 1152], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_112
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[384, 1152], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_142
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_160
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[384, 1152], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_175
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_181
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float16'),
            # parameter_187
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_195
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_197
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[384, 1152], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_202
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_220
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_226
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[384, 1152], dtype='float16'),
            # parameter_231
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_232
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_233
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_235
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_237
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float16'),
            # parameter_247
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_250
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_255
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_257
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[384, 1152], dtype='float16'),
            # parameter_261
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_262
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_265
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_267
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_273
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_274
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_280
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_281
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_283
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_285
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_286
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_287
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[384, 1152], dtype='float16'),
            # parameter_291
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_295
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_297
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_299
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_301
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_303
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float16'),
            # parameter_307
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_309
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_310
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_313
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_315
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_316
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_317
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_319
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[384, 1152], dtype='float16'),
            # parameter_321
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_322
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_323
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_325
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_327
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_328
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_329
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_331
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_333
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[384, 384, 2, 2], dtype='float16'),
            # parameter_337
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_340
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_341
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_343
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_345
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_346
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_347
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_348
            paddle.static.InputSpec(shape=[768, 384, 2, 2], dtype='float16'),
            # parameter_351
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_355
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_356
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_357
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_359
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_361
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_362
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_363
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float16'),
            # parameter_367
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_369
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_370
            paddle.static.InputSpec(shape=[768, 1536], dtype='float16'),
            # parameter_371
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_372
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_373
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_375
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_377
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_378
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_379
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_381
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[768, 1000], dtype='float16'),
            # parameter_383
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