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
    return [893][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1935_0_0(self, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, parameter_1, constant_2, constant_1, constant_0, parameter_0, parameter_2, parameter_4, parameter_3, parameter_5, parameter_6, parameter_8, parameter_7, parameter_9, parameter_10, parameter_12, parameter_11, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_24, parameter_23, parameter_25, parameter_27, parameter_26, parameter_29, parameter_28, parameter_30, parameter_31, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_53, parameter_52, parameter_54, parameter_56, parameter_55, parameter_58, parameter_57, parameter_59, parameter_60, parameter_61, parameter_62, parameter_64, parameter_63, parameter_65, parameter_66, parameter_67, parameter_68, parameter_70, parameter_69, parameter_71, parameter_72, parameter_73, parameter_74, parameter_76, parameter_75, parameter_77, parameter_78, parameter_79, parameter_80, parameter_82, parameter_81, parameter_83, parameter_85, parameter_84, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_114, parameter_113, parameter_116, parameter_115, parameter_117, parameter_118, parameter_119, parameter_120, parameter_122, parameter_121, parameter_123, parameter_124, parameter_125, parameter_126, parameter_128, parameter_127, parameter_129, parameter_130, parameter_131, parameter_132, parameter_134, parameter_133, parameter_135, parameter_136, parameter_137, parameter_138, parameter_140, parameter_139, parameter_141, parameter_143, parameter_142, parameter_145, parameter_144, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_155, parameter_157, parameter_156, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_172, parameter_171, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_194, parameter_195, parameter_196, parameter_198, parameter_197, parameter_199, parameter_201, parameter_200, parameter_203, parameter_202, parameter_204, parameter_205, parameter_206, parameter_207, parameter_209, parameter_208, parameter_210, parameter_211, parameter_212, parameter_213, parameter_215, parameter_214, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_227, parameter_226, parameter_228, parameter_230, parameter_229, parameter_232, parameter_231, parameter_233, parameter_234, parameter_235, parameter_236, parameter_238, parameter_237, parameter_239, parameter_240, parameter_241, parameter_242, parameter_244, parameter_243, parameter_245, parameter_246, parameter_247, parameter_248, parameter_250, parameter_249, parameter_251, parameter_252, parameter_253, parameter_254, parameter_256, parameter_255, parameter_257, parameter_259, parameter_258, parameter_261, parameter_260, parameter_262, parameter_263, parameter_264, parameter_265, parameter_267, parameter_266, parameter_268, parameter_269, parameter_270, parameter_271, parameter_273, parameter_272, parameter_274, parameter_275, parameter_276, parameter_277, parameter_279, parameter_278, parameter_280, parameter_281, parameter_282, parameter_283, parameter_285, parameter_284, parameter_286, parameter_288, parameter_287, parameter_290, parameter_289, parameter_291, parameter_292, parameter_293, parameter_294, parameter_296, parameter_295, parameter_297, parameter_298, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_305, parameter_306, parameter_308, parameter_307, parameter_309, parameter_310, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_317, parameter_316, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_346, parameter_345, parameter_348, parameter_347, parameter_349, parameter_350, parameter_351, parameter_352, parameter_354, parameter_353, parameter_355, parameter_356, parameter_357, parameter_358, parameter_360, parameter_359, parameter_361, parameter_362, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.shape: (4xi32) <- (-1x3x224x224xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(cast_0, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # pd_op.unfold: (-1x768x196xf16) <- (-1x3x224x224xf16)
        unfold_0 = paddle._C_ops.unfold(cast_0, [16, 16], [16, 16], [0, 0, 0, 0], [1, 1])

        # pd_op.transpose: (-1x196x768xf16) <- (-1x768x196xf16)
        transpose_0 = paddle._C_ops.transpose(unfold_0, [0, 2, 1])

        # pd_op.reshape_: (-1x3x16x16xf16, 0x-1x196x768xf16) <- (-1x196x768xf16, 4xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_0, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x24x4x4xf16) <- (-1x3x16x16xf16, 24x3x7x7xf16)
        conv2d_0 = paddle._C_ops.conv2d(reshape__0, parameter_0, [4, 4], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x24x4x4xf16) <- (-1x24x4x4xf16, 1x24x1x1xf16)
        add__0 = paddle._C_ops.add(conv2d_0, parameter_1)

        # pd_op.reshape_: (-1x24x16xf16, 0x-1x24x4x4xf16) <- (-1x24x4x4xf16, 3xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__0, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x24xf16) <- (-1x24x16xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__2, [0, 2, 1])

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 1x16x24xf16)
        add__1 = paddle._C_ops.add(transpose_1, parameter_2)

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, 3xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__1, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_0, parameter_3, parameter_4, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_0 = paddle.matmul(layer_norm_0, parameter_5, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__2 = paddle._C_ops.add(matmul_0, parameter_6)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__2, parameter_7, parameter_8, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, constant_5, constant_5]

        # pd_op.expand: (-1x1x384xf16) <- (1x1x384xf16, [1xi32, 1xi32, 1xi32])
        expand_0 = paddle._C_ops.expand(parameter_9, combine_0)

        # pd_op.cast: (-1x1x384xf32) <- (-1x1x384xf16)
        cast_1 = paddle._C_ops.cast(expand_0, paddle.float32)

        # pd_op.cast: (-1x1x384xf16) <- (-1x1x384xf32)
        cast_2 = paddle._C_ops.cast(cast_1, paddle.float16)

        # builtin.combine: ([-1x1x384xf16, -1x196x384xf16]) <- (-1x1x384xf16, -1x196x384xf16)
        combine_1 = [cast_2, layer_norm_3]

        # pd_op.concat: (-1x197x384xf16) <- ([-1x1x384xf16, -1x196x384xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, constant_6)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 1x197x384xf16)
        add__3 = paddle._C_ops.add(concat_0, parameter_10)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__1, parameter_11, parameter_12, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(layer_norm_6, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_1 = paddle.matmul(layer_norm_6, parameter_13, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_1, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_1, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_2 = paddle._C_ops.transpose(reshape__4, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_2, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_2, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_2 = paddle.matmul(layer_norm_6, parameter_14, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_1, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_2, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__6, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_4 = paddle._C_ops.transpose(slice_3, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_3 = paddle.matmul(slice_2, transpose_4, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__0 = paddle._C_ops.scale(matmul_3, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__0 = paddle._C_ops.softmax(scale__0, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_4 = paddle.matmul(softmax__0, transpose_3, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_5 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_1, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_5, combine_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_5 = paddle.matmul(reshape__8, parameter_15, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__4 = paddle._C_ops.add(matmul_5, parameter_16)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__5 = paddle._C_ops.add(add__1, add__4)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__5, parameter_17, parameter_18, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_6 = paddle.matmul(layer_norm_9, parameter_19, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__6 = paddle._C_ops.add(matmul_6, parameter_20)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_0 = paddle._C_ops.gelu(add__6, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_7 = paddle.matmul(gelu_0, parameter_21, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__7 = paddle._C_ops.add(matmul_7, parameter_22)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__8 = paddle._C_ops.add(add__5, add__7)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(add__3, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_2, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_5 = [slice_4, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__8, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_2, parameter_23, parameter_24, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(add__3, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_8 = paddle.matmul(layer_norm_12, parameter_25, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_8, parameter_26, parameter_27, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__9 = paddle._C_ops.add(slice_5, layer_norm_15)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor(add__3, add__9, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__0, parameter_28, parameter_29, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(layer_norm_18, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_3, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_9 = paddle.matmul(layer_norm_18, parameter_30, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_6, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_9, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_6 = paddle._C_ops.transpose(reshape__10, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_6, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(transpose_6, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_10 = paddle.matmul(layer_norm_18, parameter_31, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_6, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_10, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_7 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_8 = paddle._C_ops.transpose(slice_8, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_11 = paddle.matmul(slice_7, transpose_8, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__1 = paddle._C_ops.scale(matmul_11, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__1 = paddle._C_ops.softmax(scale__1, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_12 = paddle.matmul(softmax__1, transpose_7, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_9 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_8 = [slice_6, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_9, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_13 = paddle.matmul(reshape__14, parameter_32, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__10 = paddle._C_ops.add(matmul_13, parameter_33)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__11 = paddle._C_ops.add(set_value_with_tensor__0, add__10)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__11, parameter_34, parameter_35, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_14 = paddle.matmul(layer_norm_21, parameter_36, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__12 = paddle._C_ops.add(matmul_14, parameter_37)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_1 = paddle._C_ops.gelu(add__12, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_15 = paddle.matmul(gelu_1, parameter_38, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__13 = paddle._C_ops.add(matmul_15, parameter_39)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__14 = paddle._C_ops.add(add__11, add__13)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__8, parameter_40, parameter_41, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(layer_norm_24, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_4, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_16 = paddle.matmul(layer_norm_24, parameter_42, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_9, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_16, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_10 = paddle._C_ops.transpose(reshape__16, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_10, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_10, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_17 = paddle.matmul(layer_norm_24, parameter_43, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [slice_9, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_17, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_11 = paddle._C_ops.transpose(reshape__18, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_12 = paddle._C_ops.transpose(slice_11, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_18 = paddle.matmul(slice_10, transpose_12, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__2 = paddle._C_ops.scale(matmul_18, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__2 = paddle._C_ops.softmax(scale__2, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_19 = paddle.matmul(softmax__2, transpose_11, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_13 = paddle._C_ops.transpose(matmul_19, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_11 = [slice_9, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_13, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_20 = paddle.matmul(reshape__20, parameter_44, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__15 = paddle._C_ops.add(matmul_20, parameter_45)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__16 = paddle._C_ops.add(add__8, add__15)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__16, parameter_46, parameter_47, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_21 = paddle.matmul(layer_norm_27, parameter_48, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__17 = paddle._C_ops.add(matmul_21, parameter_49)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_2 = paddle._C_ops.gelu(add__17, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_22 = paddle.matmul(gelu_2, parameter_50, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__18 = paddle._C_ops.add(matmul_22, parameter_51)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__19 = paddle._C_ops.add(add__16, add__18)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(add__14, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_5, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_12 = [slice_12, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__19, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_4, parameter_52, parameter_53, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(add__14, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_23 = paddle.matmul(layer_norm_30, parameter_54, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_23, parameter_55, parameter_56, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__20 = paddle._C_ops.add(slice_13, layer_norm_33)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor(add__14, add__20, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__1, parameter_57, parameter_58, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(layer_norm_36, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_6, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_24 = paddle.matmul(layer_norm_36, parameter_59, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_14, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_24, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_14 = paddle._C_ops.transpose(reshape__22, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_14, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_14, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_25 = paddle.matmul(layer_norm_36, parameter_60, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_14, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_25, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_15 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_16 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_26 = paddle.matmul(slice_15, transpose_16, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__3 = paddle._C_ops.scale(matmul_26, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__3 = paddle._C_ops.softmax(scale__3, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_27 = paddle.matmul(softmax__3, transpose_15, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_17 = paddle._C_ops.transpose(matmul_27, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_15 = [slice_14, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_17, combine_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_28 = paddle.matmul(reshape__26, parameter_61, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__21 = paddle._C_ops.add(matmul_28, parameter_62)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__22 = paddle._C_ops.add(set_value_with_tensor__1, add__21)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__22, parameter_63, parameter_64, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_29 = paddle.matmul(layer_norm_39, parameter_65, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__23 = paddle._C_ops.add(matmul_29, parameter_66)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_3 = paddle._C_ops.gelu(add__23, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_30 = paddle.matmul(gelu_3, parameter_67, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__24 = paddle._C_ops.add(matmul_30, parameter_68)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__25 = paddle._C_ops.add(add__22, add__24)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__19, parameter_69, parameter_70, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(layer_norm_42, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_7, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_31 = paddle.matmul(layer_norm_42, parameter_71, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_17, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_31, combine_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_18 = paddle._C_ops.transpose(reshape__28, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_18, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_18, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_32 = paddle.matmul(layer_norm_42, parameter_72, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_17, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_32, combine_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_19 = paddle._C_ops.transpose(reshape__30, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_20 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_33 = paddle.matmul(slice_18, transpose_20, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__4 = paddle._C_ops.scale(matmul_33, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__4 = paddle._C_ops.softmax(scale__4, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_34 = paddle.matmul(softmax__4, transpose_19, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_21 = paddle._C_ops.transpose(matmul_34, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_18 = [slice_17, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_21, combine_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_35 = paddle.matmul(reshape__32, parameter_73, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__26 = paddle._C_ops.add(matmul_35, parameter_74)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__27 = paddle._C_ops.add(add__19, add__26)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__27, parameter_75, parameter_76, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_36 = paddle.matmul(layer_norm_45, parameter_77, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__28 = paddle._C_ops.add(matmul_36, parameter_78)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_4 = paddle._C_ops.gelu(add__28, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_37 = paddle.matmul(gelu_4, parameter_79, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__29 = paddle._C_ops.add(matmul_37, parameter_80)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__30 = paddle._C_ops.add(add__27, add__29)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(add__25, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_8, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_19 = [slice_20, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__30, combine_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_6, parameter_81, parameter_82, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(add__25, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_38 = paddle.matmul(layer_norm_48, parameter_83, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_38, parameter_84, parameter_85, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__31 = paddle._C_ops.add(slice_21, layer_norm_51)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor(add__25, add__31, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__2, parameter_86, parameter_87, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(layer_norm_54, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(shape_9, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_39 = paddle.matmul(layer_norm_54, parameter_88, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_22, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_39, combine_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_22 = paddle._C_ops.transpose(reshape__34, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_22, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_22, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_40 = paddle.matmul(layer_norm_54, parameter_89, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_22, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_40, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_23 = paddle._C_ops.transpose(reshape__36, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_24 = paddle._C_ops.transpose(slice_24, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_41 = paddle.matmul(slice_23, transpose_24, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__5 = paddle._C_ops.scale(matmul_41, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__5 = paddle._C_ops.softmax(scale__5, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_42 = paddle.matmul(softmax__5, transpose_23, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_25 = paddle._C_ops.transpose(matmul_42, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_22 = [slice_22, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_25, combine_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_43 = paddle.matmul(reshape__38, parameter_90, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__32 = paddle._C_ops.add(matmul_43, parameter_91)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__33 = paddle._C_ops.add(set_value_with_tensor__2, add__32)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__33, parameter_92, parameter_93, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_44 = paddle.matmul(layer_norm_57, parameter_94, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__34 = paddle._C_ops.add(matmul_44, parameter_95)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_5 = paddle._C_ops.gelu(add__34, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_45 = paddle.matmul(gelu_5, parameter_96, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__35 = paddle._C_ops.add(matmul_45, parameter_97)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__36 = paddle._C_ops.add(add__33, add__35)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__30, parameter_98, parameter_99, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_10 = paddle._C_ops.shape(paddle.cast(layer_norm_60, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_10, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_46 = paddle.matmul(layer_norm_60, parameter_100, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_25, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_46, combine_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_26 = paddle._C_ops.transpose(reshape__40, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_26, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_26, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_47 = paddle.matmul(layer_norm_60, parameter_101, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_25, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_47, combine_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_27 = paddle._C_ops.transpose(reshape__42, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_28 = paddle._C_ops.transpose(slice_27, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_48 = paddle.matmul(slice_26, transpose_28, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__6 = paddle._C_ops.scale(matmul_48, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__6 = paddle._C_ops.softmax(scale__6, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_49 = paddle.matmul(softmax__6, transpose_27, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_29 = paddle._C_ops.transpose(matmul_49, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_25, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_29, combine_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_50 = paddle.matmul(reshape__44, parameter_102, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__37 = paddle._C_ops.add(matmul_50, parameter_103)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__38 = paddle._C_ops.add(add__30, add__37)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__38, parameter_104, parameter_105, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_51 = paddle.matmul(layer_norm_63, parameter_106, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__39 = paddle._C_ops.add(matmul_51, parameter_107)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_6 = paddle._C_ops.gelu(add__39, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_52 = paddle.matmul(gelu_6, parameter_108, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__40 = paddle._C_ops.add(matmul_52, parameter_109)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__41 = paddle._C_ops.add(add__38, add__40)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(add__36, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_11, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_26 = [slice_28, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__41, combine_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_8, parameter_110, parameter_111, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(add__36, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_53 = paddle.matmul(layer_norm_66, parameter_112, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_53, parameter_113, parameter_114, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__42 = paddle._C_ops.add(slice_29, layer_norm_69)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__3 = paddle._C_ops.set_value_with_tensor(add__36, add__42, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__3, parameter_115, parameter_116, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_12 = paddle._C_ops.shape(paddle.cast(layer_norm_72, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(shape_12, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_54 = paddle.matmul(layer_norm_72, parameter_117, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_27 = [slice_30, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_54, combine_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_30 = paddle._C_ops.transpose(reshape__46, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_30, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(transpose_30, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_55 = paddle.matmul(layer_norm_72, parameter_118, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_28 = [slice_30, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_55, combine_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_31 = paddle._C_ops.transpose(reshape__48, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_32 = paddle._C_ops.transpose(slice_32, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_56 = paddle.matmul(slice_31, transpose_32, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__7 = paddle._C_ops.scale(matmul_56, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__7 = paddle._C_ops.softmax(scale__7, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_57 = paddle.matmul(softmax__7, transpose_31, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_33 = paddle._C_ops.transpose(matmul_57, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_29 = [slice_30, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_33, combine_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_58 = paddle.matmul(reshape__50, parameter_119, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__43 = paddle._C_ops.add(matmul_58, parameter_120)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__44 = paddle._C_ops.add(set_value_with_tensor__3, add__43)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__44, parameter_121, parameter_122, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_59 = paddle.matmul(layer_norm_75, parameter_123, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__45 = paddle._C_ops.add(matmul_59, parameter_124)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_7 = paddle._C_ops.gelu(add__45, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_60 = paddle.matmul(gelu_7, parameter_125, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__46 = paddle._C_ops.add(matmul_60, parameter_126)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__47 = paddle._C_ops.add(add__44, add__46)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__41, parameter_127, parameter_128, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_13 = paddle._C_ops.shape(paddle.cast(layer_norm_78, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(shape_13, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_61 = paddle.matmul(layer_norm_78, parameter_129, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_30 = [slice_33, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_61, combine_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_34 = paddle._C_ops.transpose(reshape__52, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_34, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_34, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_62 = paddle.matmul(layer_norm_78, parameter_130, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [slice_33, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_62, combine_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_35 = paddle._C_ops.transpose(reshape__54, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_36 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_63 = paddle.matmul(slice_34, transpose_36, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__8 = paddle._C_ops.scale(matmul_63, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__8 = paddle._C_ops.softmax(scale__8, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_64 = paddle.matmul(softmax__8, transpose_35, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_37 = paddle._C_ops.transpose(matmul_64, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_32 = [slice_33, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_37, combine_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_65 = paddle.matmul(reshape__56, parameter_131, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__48 = paddle._C_ops.add(matmul_65, parameter_132)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__49 = paddle._C_ops.add(add__41, add__48)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__49, parameter_133, parameter_134, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_66 = paddle.matmul(layer_norm_81, parameter_135, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__50 = paddle._C_ops.add(matmul_66, parameter_136)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_8 = paddle._C_ops.gelu(add__50, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_67 = paddle.matmul(gelu_8, parameter_137, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__51 = paddle._C_ops.add(matmul_67, parameter_138)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__52 = paddle._C_ops.add(add__49, add__51)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_14 = paddle._C_ops.shape(paddle.cast(add__47, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(shape_14, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_33 = [slice_36, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__52, combine_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_10, parameter_139, parameter_140, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(add__47, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_68 = paddle.matmul(layer_norm_84, parameter_141, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_68, parameter_142, parameter_143, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__53 = paddle._C_ops.add(slice_37, layer_norm_87)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__4 = paddle._C_ops.set_value_with_tensor(add__47, add__53, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__4, parameter_144, parameter_145, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_15 = paddle._C_ops.shape(paddle.cast(layer_norm_90, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(shape_15, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_69 = paddle.matmul(layer_norm_90, parameter_146, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_34 = [slice_38, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_69, combine_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_38 = paddle._C_ops.transpose(reshape__58, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(transpose_38, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(transpose_38, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_70 = paddle.matmul(layer_norm_90, parameter_147, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_38, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_70, combine_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_39 = paddle._C_ops.transpose(reshape__60, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_40 = paddle._C_ops.transpose(slice_40, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_71 = paddle.matmul(slice_39, transpose_40, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__9 = paddle._C_ops.scale(matmul_71, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__9 = paddle._C_ops.softmax(scale__9, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_72 = paddle.matmul(softmax__9, transpose_39, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_41 = paddle._C_ops.transpose(matmul_72, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_36 = [slice_38, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_41, combine_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_73 = paddle.matmul(reshape__62, parameter_148, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__54 = paddle._C_ops.add(matmul_73, parameter_149)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__55 = paddle._C_ops.add(set_value_with_tensor__4, add__54)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__55, parameter_150, parameter_151, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_74 = paddle.matmul(layer_norm_93, parameter_152, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__56 = paddle._C_ops.add(matmul_74, parameter_153)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_9 = paddle._C_ops.gelu(add__56, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_75 = paddle.matmul(gelu_9, parameter_154, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__57 = paddle._C_ops.add(matmul_75, parameter_155)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__58 = paddle._C_ops.add(add__55, add__57)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__52, parameter_156, parameter_157, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_16 = paddle._C_ops.shape(paddle.cast(layer_norm_96, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(shape_16, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_76 = paddle.matmul(layer_norm_96, parameter_158, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_41, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_76, combine_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_42 = paddle._C_ops.transpose(reshape__64, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(transpose_42, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(transpose_42, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_77 = paddle.matmul(layer_norm_96, parameter_159, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_41, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_77, combine_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_43 = paddle._C_ops.transpose(reshape__66, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_44 = paddle._C_ops.transpose(slice_43, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_78 = paddle.matmul(slice_42, transpose_44, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__10 = paddle._C_ops.scale(matmul_78, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__10 = paddle._C_ops.softmax(scale__10, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_79 = paddle.matmul(softmax__10, transpose_43, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_45 = paddle._C_ops.transpose(matmul_79, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_39 = [slice_41, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_45, combine_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_80 = paddle.matmul(reshape__68, parameter_160, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__59 = paddle._C_ops.add(matmul_80, parameter_161)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__60 = paddle._C_ops.add(add__52, add__59)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__60, parameter_162, parameter_163, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_81 = paddle.matmul(layer_norm_99, parameter_164, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__61 = paddle._C_ops.add(matmul_81, parameter_165)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_10 = paddle._C_ops.gelu(add__61, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_82 = paddle.matmul(gelu_10, parameter_166, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__62 = paddle._C_ops.add(matmul_82, parameter_167)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__63 = paddle._C_ops.add(add__60, add__62)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_17 = paddle._C_ops.shape(paddle.cast(add__58, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(shape_17, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_40 = [slice_44, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__63, combine_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_12, parameter_168, parameter_169, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(add__58, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_83 = paddle.matmul(layer_norm_102, parameter_170, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_83, parameter_171, parameter_172, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__64 = paddle._C_ops.add(slice_45, layer_norm_105)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__5 = paddle._C_ops.set_value_with_tensor(add__58, add__64, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__5, parameter_173, parameter_174, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_18 = paddle._C_ops.shape(paddle.cast(layer_norm_108, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(shape_18, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_84 = paddle.matmul(layer_norm_108, parameter_175, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_46, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_84, combine_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_46 = paddle._C_ops.transpose(reshape__70, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(transpose_46, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(transpose_46, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_85 = paddle.matmul(layer_norm_108, parameter_176, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_46, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_85, combine_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_47 = paddle._C_ops.transpose(reshape__72, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_48 = paddle._C_ops.transpose(slice_48, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_86 = paddle.matmul(slice_47, transpose_48, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__11 = paddle._C_ops.scale(matmul_86, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__11 = paddle._C_ops.softmax(scale__11, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_87 = paddle.matmul(softmax__11, transpose_47, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_49 = paddle._C_ops.transpose(matmul_87, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_43 = [slice_46, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_49, combine_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_88 = paddle.matmul(reshape__74, parameter_177, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__65 = paddle._C_ops.add(matmul_88, parameter_178)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__66 = paddle._C_ops.add(set_value_with_tensor__5, add__65)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__66, parameter_179, parameter_180, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_89 = paddle.matmul(layer_norm_111, parameter_181, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__67 = paddle._C_ops.add(matmul_89, parameter_182)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_11 = paddle._C_ops.gelu(add__67, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_90 = paddle.matmul(gelu_11, parameter_183, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__68 = paddle._C_ops.add(matmul_90, parameter_184)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__69 = paddle._C_ops.add(add__66, add__68)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__63, parameter_185, parameter_186, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_19 = paddle._C_ops.shape(paddle.cast(layer_norm_114, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(shape_19, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_91 = paddle.matmul(layer_norm_114, parameter_187, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_49, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_91, combine_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_50 = paddle._C_ops.transpose(reshape__76, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(transpose_50, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(transpose_50, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_92 = paddle.matmul(layer_norm_114, parameter_188, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [slice_49, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_92, combine_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_51 = paddle._C_ops.transpose(reshape__78, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_52 = paddle._C_ops.transpose(slice_51, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_93 = paddle.matmul(slice_50, transpose_52, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__12 = paddle._C_ops.scale(matmul_93, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__12 = paddle._C_ops.softmax(scale__12, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_94 = paddle.matmul(softmax__12, transpose_51, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_53 = paddle._C_ops.transpose(matmul_94, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_46 = [slice_49, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_53, combine_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_95 = paddle.matmul(reshape__80, parameter_189, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__70 = paddle._C_ops.add(matmul_95, parameter_190)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__71 = paddle._C_ops.add(add__63, add__70)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__71, parameter_191, parameter_192, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_96 = paddle.matmul(layer_norm_117, parameter_193, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__72 = paddle._C_ops.add(matmul_96, parameter_194)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_12 = paddle._C_ops.gelu(add__72, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_97 = paddle.matmul(gelu_12, parameter_195, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__73 = paddle._C_ops.add(matmul_97, parameter_196)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__74 = paddle._C_ops.add(add__71, add__73)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_20 = paddle._C_ops.shape(paddle.cast(add__69, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(shape_20, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_47 = [slice_52, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__74, combine_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_14, parameter_197, parameter_198, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(add__69, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_98 = paddle.matmul(layer_norm_120, parameter_199, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_98, parameter_200, parameter_201, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__75 = paddle._C_ops.add(slice_53, layer_norm_123)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__6 = paddle._C_ops.set_value_with_tensor(add__69, add__75, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__6, parameter_202, parameter_203, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_21 = paddle._C_ops.shape(paddle.cast(layer_norm_126, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(shape_21, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_99 = paddle.matmul(layer_norm_126, parameter_204, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_48 = [slice_54, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_99, combine_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_54 = paddle._C_ops.transpose(reshape__82, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(transpose_54, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(transpose_54, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_100 = paddle.matmul(layer_norm_126, parameter_205, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_49 = [slice_54, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_100, combine_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_55 = paddle._C_ops.transpose(reshape__84, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_56 = paddle._C_ops.transpose(slice_56, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_101 = paddle.matmul(slice_55, transpose_56, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__13 = paddle._C_ops.scale(matmul_101, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__13 = paddle._C_ops.softmax(scale__13, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_102 = paddle.matmul(softmax__13, transpose_55, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_57 = paddle._C_ops.transpose(matmul_102, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_50 = [slice_54, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_57, combine_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_103 = paddle.matmul(reshape__86, parameter_206, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__76 = paddle._C_ops.add(matmul_103, parameter_207)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__77 = paddle._C_ops.add(set_value_with_tensor__6, add__76)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__77, parameter_208, parameter_209, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_104 = paddle.matmul(layer_norm_129, parameter_210, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__78 = paddle._C_ops.add(matmul_104, parameter_211)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_13 = paddle._C_ops.gelu(add__78, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_105 = paddle.matmul(gelu_13, parameter_212, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__79 = paddle._C_ops.add(matmul_105, parameter_213)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__80 = paddle._C_ops.add(add__77, add__79)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__74, parameter_214, parameter_215, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_22 = paddle._C_ops.shape(paddle.cast(layer_norm_132, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(shape_22, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_106 = paddle.matmul(layer_norm_132, parameter_216, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_51 = [slice_57, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_106, combine_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_58 = paddle._C_ops.transpose(reshape__88, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(transpose_58, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(transpose_58, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_107 = paddle.matmul(layer_norm_132, parameter_217, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_52 = [slice_57, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_107, combine_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_59 = paddle._C_ops.transpose(reshape__90, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_60 = paddle._C_ops.transpose(slice_59, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_108 = paddle.matmul(slice_58, transpose_60, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__14 = paddle._C_ops.scale(matmul_108, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__14 = paddle._C_ops.softmax(scale__14, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_109 = paddle.matmul(softmax__14, transpose_59, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_61 = paddle._C_ops.transpose(matmul_109, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_53 = [slice_57, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_61, combine_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_110 = paddle.matmul(reshape__92, parameter_218, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__81 = paddle._C_ops.add(matmul_110, parameter_219)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__82 = paddle._C_ops.add(add__74, add__81)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__82, parameter_220, parameter_221, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_111 = paddle.matmul(layer_norm_135, parameter_222, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__83 = paddle._C_ops.add(matmul_111, parameter_223)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_14 = paddle._C_ops.gelu(add__83, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_112 = paddle.matmul(gelu_14, parameter_224, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__84 = paddle._C_ops.add(matmul_112, parameter_225)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__85 = paddle._C_ops.add(add__82, add__84)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_23 = paddle._C_ops.shape(paddle.cast(add__80, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(shape_23, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_54 = [slice_60, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__85, combine_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_16, parameter_226, parameter_227, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(add__80, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_113 = paddle.matmul(layer_norm_138, parameter_228, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_113, parameter_229, parameter_230, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__86 = paddle._C_ops.add(slice_61, layer_norm_141)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__7 = paddle._C_ops.set_value_with_tensor(add__80, add__86, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__7, parameter_231, parameter_232, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_24 = paddle._C_ops.shape(paddle.cast(layer_norm_144, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(shape_24, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_114 = paddle.matmul(layer_norm_144, parameter_233, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_55 = [slice_62, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_114, combine_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_62 = paddle._C_ops.transpose(reshape__94, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(transpose_62, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(transpose_62, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_115 = paddle.matmul(layer_norm_144, parameter_234, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_56 = [slice_62, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_115, combine_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_63 = paddle._C_ops.transpose(reshape__96, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_64 = paddle._C_ops.transpose(slice_64, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_116 = paddle.matmul(slice_63, transpose_64, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__15 = paddle._C_ops.scale(matmul_116, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__15 = paddle._C_ops.softmax(scale__15, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_117 = paddle.matmul(softmax__15, transpose_63, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_65 = paddle._C_ops.transpose(matmul_117, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_57 = [slice_62, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_65, combine_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_118 = paddle.matmul(reshape__98, parameter_235, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__87 = paddle._C_ops.add(matmul_118, parameter_236)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__88 = paddle._C_ops.add(set_value_with_tensor__7, add__87)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_147, layer_norm_148, layer_norm_149 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__88, parameter_237, parameter_238, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_119 = paddle.matmul(layer_norm_147, parameter_239, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__89 = paddle._C_ops.add(matmul_119, parameter_240)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_15 = paddle._C_ops.gelu(add__89, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_120 = paddle.matmul(gelu_15, parameter_241, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__90 = paddle._C_ops.add(matmul_120, parameter_242)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__91 = paddle._C_ops.add(add__88, add__90)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_150, layer_norm_151, layer_norm_152 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__85, parameter_243, parameter_244, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_25 = paddle._C_ops.shape(paddle.cast(layer_norm_150, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(shape_25, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_121 = paddle.matmul(layer_norm_150, parameter_245, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_58 = [slice_65, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_121, combine_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_66 = paddle._C_ops.transpose(reshape__100, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(transpose_66, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(transpose_66, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_122 = paddle.matmul(layer_norm_150, parameter_246, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_59 = [slice_65, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_122, combine_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_67 = paddle._C_ops.transpose(reshape__102, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_68 = paddle._C_ops.transpose(slice_67, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_123 = paddle.matmul(slice_66, transpose_68, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__16 = paddle._C_ops.scale(matmul_123, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__16 = paddle._C_ops.softmax(scale__16, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_124 = paddle.matmul(softmax__16, transpose_67, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_69 = paddle._C_ops.transpose(matmul_124, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_60 = [slice_65, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_69, combine_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_125 = paddle.matmul(reshape__104, parameter_247, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__92 = paddle._C_ops.add(matmul_125, parameter_248)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__93 = paddle._C_ops.add(add__85, add__92)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_153, layer_norm_154, layer_norm_155 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__93, parameter_249, parameter_250, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_126 = paddle.matmul(layer_norm_153, parameter_251, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__94 = paddle._C_ops.add(matmul_126, parameter_252)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_16 = paddle._C_ops.gelu(add__94, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_127 = paddle.matmul(gelu_16, parameter_253, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__95 = paddle._C_ops.add(matmul_127, parameter_254)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__96 = paddle._C_ops.add(add__93, add__95)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_26 = paddle._C_ops.shape(paddle.cast(add__91, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(shape_26, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_61 = [slice_68, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__96, combine_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_156, layer_norm_157, layer_norm_158 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_18, parameter_255, parameter_256, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(add__91, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_128 = paddle.matmul(layer_norm_156, parameter_257, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_159, layer_norm_160, layer_norm_161 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_128, parameter_258, parameter_259, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__97 = paddle._C_ops.add(slice_69, layer_norm_159)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__8 = paddle._C_ops.set_value_with_tensor(add__91, add__97, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_162, layer_norm_163, layer_norm_164 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__8, parameter_260, parameter_261, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_27 = paddle._C_ops.shape(paddle.cast(layer_norm_162, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(shape_27, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_129 = paddle.matmul(layer_norm_162, parameter_262, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_62 = [slice_70, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_129, combine_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_70 = paddle._C_ops.transpose(reshape__106, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(transpose_70, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(transpose_70, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_130 = paddle.matmul(layer_norm_162, parameter_263, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_63 = [slice_70, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_130, combine_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_71 = paddle._C_ops.transpose(reshape__108, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_72 = paddle._C_ops.transpose(slice_72, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_131 = paddle.matmul(slice_71, transpose_72, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__17 = paddle._C_ops.scale(matmul_131, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__17 = paddle._C_ops.softmax(scale__17, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_132 = paddle.matmul(softmax__17, transpose_71, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_73 = paddle._C_ops.transpose(matmul_132, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_64 = [slice_70, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_73, combine_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_133 = paddle.matmul(reshape__110, parameter_264, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__98 = paddle._C_ops.add(matmul_133, parameter_265)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__99 = paddle._C_ops.add(set_value_with_tensor__8, add__98)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_165, layer_norm_166, layer_norm_167 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__99, parameter_266, parameter_267, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_134 = paddle.matmul(layer_norm_165, parameter_268, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__100 = paddle._C_ops.add(matmul_134, parameter_269)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_17 = paddle._C_ops.gelu(add__100, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_135 = paddle.matmul(gelu_17, parameter_270, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__101 = paddle._C_ops.add(matmul_135, parameter_271)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__102 = paddle._C_ops.add(add__99, add__101)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_168, layer_norm_169, layer_norm_170 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__96, parameter_272, parameter_273, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_28 = paddle._C_ops.shape(paddle.cast(layer_norm_168, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(shape_28, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_136 = paddle.matmul(layer_norm_168, parameter_274, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_65 = [slice_73, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_136, combine_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_74 = paddle._C_ops.transpose(reshape__112, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(transpose_74, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(transpose_74, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_137 = paddle.matmul(layer_norm_168, parameter_275, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_66 = [slice_73, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_137, combine_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_75 = paddle._C_ops.transpose(reshape__114, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_76 = paddle._C_ops.transpose(slice_75, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_138 = paddle.matmul(slice_74, transpose_76, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__18 = paddle._C_ops.scale(matmul_138, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__18 = paddle._C_ops.softmax(scale__18, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_139 = paddle.matmul(softmax__18, transpose_75, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_77 = paddle._C_ops.transpose(matmul_139, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_67 = [slice_73, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_77, combine_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_140 = paddle.matmul(reshape__116, parameter_276, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__103 = paddle._C_ops.add(matmul_140, parameter_277)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__104 = paddle._C_ops.add(add__96, add__103)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_171, layer_norm_172, layer_norm_173 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__104, parameter_278, parameter_279, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_141 = paddle.matmul(layer_norm_171, parameter_280, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__105 = paddle._C_ops.add(matmul_141, parameter_281)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_18 = paddle._C_ops.gelu(add__105, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_142 = paddle.matmul(gelu_18, parameter_282, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__106 = paddle._C_ops.add(matmul_142, parameter_283)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__107 = paddle._C_ops.add(add__104, add__106)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_29 = paddle._C_ops.shape(paddle.cast(add__102, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(shape_29, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_68 = [slice_76, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__107, combine_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_174, layer_norm_175, layer_norm_176 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_20, parameter_284, parameter_285, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(add__102, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_143 = paddle.matmul(layer_norm_174, parameter_286, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_177, layer_norm_178, layer_norm_179 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_143, parameter_287, parameter_288, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__108 = paddle._C_ops.add(slice_77, layer_norm_177)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__9 = paddle._C_ops.set_value_with_tensor(add__102, add__108, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_180, layer_norm_181, layer_norm_182 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__9, parameter_289, parameter_290, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_30 = paddle._C_ops.shape(paddle.cast(layer_norm_180, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(shape_30, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_144 = paddle.matmul(layer_norm_180, parameter_291, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_69 = [slice_78, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_144, combine_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_78 = paddle._C_ops.transpose(reshape__118, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(transpose_78, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(transpose_78, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_145 = paddle.matmul(layer_norm_180, parameter_292, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_70 = [slice_78, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_145, combine_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_79 = paddle._C_ops.transpose(reshape__120, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_80 = paddle._C_ops.transpose(slice_80, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_146 = paddle.matmul(slice_79, transpose_80, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__19 = paddle._C_ops.scale(matmul_146, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__19 = paddle._C_ops.softmax(scale__19, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_147 = paddle.matmul(softmax__19, transpose_79, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_81 = paddle._C_ops.transpose(matmul_147, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_71 = [slice_78, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_81, combine_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_148 = paddle.matmul(reshape__122, parameter_293, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__109 = paddle._C_ops.add(matmul_148, parameter_294)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__110 = paddle._C_ops.add(set_value_with_tensor__9, add__109)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_183, layer_norm_184, layer_norm_185 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__110, parameter_295, parameter_296, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_149 = paddle.matmul(layer_norm_183, parameter_297, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__111 = paddle._C_ops.add(matmul_149, parameter_298)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_19 = paddle._C_ops.gelu(add__111, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_150 = paddle.matmul(gelu_19, parameter_299, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__112 = paddle._C_ops.add(matmul_150, parameter_300)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__113 = paddle._C_ops.add(add__110, add__112)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_186, layer_norm_187, layer_norm_188 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__107, parameter_301, parameter_302, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_31 = paddle._C_ops.shape(paddle.cast(layer_norm_186, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(shape_31, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_151 = paddle.matmul(layer_norm_186, parameter_303, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_72 = [slice_81, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_151, combine_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_82 = paddle._C_ops.transpose(reshape__124, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(transpose_82, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(transpose_82, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_152 = paddle.matmul(layer_norm_186, parameter_304, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_73 = [slice_81, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_152, combine_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_83 = paddle._C_ops.transpose(reshape__126, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_84 = paddle._C_ops.transpose(slice_83, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_153 = paddle.matmul(slice_82, transpose_84, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__20 = paddle._C_ops.scale(matmul_153, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__20 = paddle._C_ops.softmax(scale__20, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_154 = paddle.matmul(softmax__20, transpose_83, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_85 = paddle._C_ops.transpose(matmul_154, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_74 = [slice_81, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_85, combine_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_155 = paddle.matmul(reshape__128, parameter_305, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__114 = paddle._C_ops.add(matmul_155, parameter_306)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__115 = paddle._C_ops.add(add__107, add__114)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_189, layer_norm_190, layer_norm_191 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__115, parameter_307, parameter_308, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_156 = paddle.matmul(layer_norm_189, parameter_309, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__116 = paddle._C_ops.add(matmul_156, parameter_310)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_20 = paddle._C_ops.gelu(add__116, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_157 = paddle.matmul(gelu_20, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__117 = paddle._C_ops.add(matmul_157, parameter_312)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__118 = paddle._C_ops.add(add__115, add__117)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_32 = paddle._C_ops.shape(paddle.cast(add__113, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(shape_32, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_75 = [slice_84, constant_14, constant_15]

        # pd_op.reshape: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__118, combine_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_192, layer_norm_193, layer_norm_194 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_22, parameter_313, parameter_314, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(add__113, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_158 = paddle.matmul(layer_norm_192, parameter_315, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_195, layer_norm_196, layer_norm_197 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_158, parameter_316, parameter_317, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__119 = paddle._C_ops.add(slice_85, layer_norm_195)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__10 = paddle._C_ops.set_value_with_tensor(add__113, add__119, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_198, layer_norm_199, layer_norm_200 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__10, parameter_318, parameter_319, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_33 = paddle._C_ops.shape(paddle.cast(layer_norm_198, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(shape_33, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_159 = paddle.matmul(layer_norm_198, parameter_320, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_76 = [slice_86, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_159, combine_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_86 = paddle._C_ops.transpose(reshape__130, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(transpose_86, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(transpose_86, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_160 = paddle.matmul(layer_norm_198, parameter_321, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_77 = [slice_86, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_160, combine_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_87 = paddle._C_ops.transpose(reshape__132, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_88 = paddle._C_ops.transpose(slice_88, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_161 = paddle.matmul(slice_87, transpose_88, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__21 = paddle._C_ops.scale(matmul_161, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__21 = paddle._C_ops.softmax(scale__21, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_162 = paddle.matmul(softmax__21, transpose_87, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_89 = paddle._C_ops.transpose(matmul_162, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_78 = [slice_86, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_89, combine_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_163 = paddle.matmul(reshape__134, parameter_322, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__120 = paddle._C_ops.add(matmul_163, parameter_323)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__121 = paddle._C_ops.add(set_value_with_tensor__10, add__120)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_201, layer_norm_202, layer_norm_203 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__121, parameter_324, parameter_325, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_164 = paddle.matmul(layer_norm_201, parameter_326, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__122 = paddle._C_ops.add(matmul_164, parameter_327)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_21 = paddle._C_ops.gelu(add__122, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_165 = paddle.matmul(gelu_21, parameter_328, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__123 = paddle._C_ops.add(matmul_165, parameter_329)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__124 = paddle._C_ops.add(add__121, add__123)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_204, layer_norm_205, layer_norm_206 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__118, parameter_330, parameter_331, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf16)
        shape_34 = paddle._C_ops.shape(paddle.cast(layer_norm_204, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(shape_34, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x16x48xf16) <- (-1x16x24xf16, 24x48xf16)
        matmul_166 = paddle.matmul(layer_norm_204, parameter_332, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_79 = [slice_89, constant_7, constant_8, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x2x4x6xf16, 0x-1x16x48xf16) <- (-1x16x48xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_166, combine_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf16) <- (-1x16x2x4x6xf16)
        transpose_90 = paddle._C_ops.transpose(reshape__136, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(transpose_90, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x4x16x6xf16) <- (2x-1x4x16x6xf16, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(transpose_90, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_167 = paddle.matmul(layer_norm_204, parameter_333, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_80 = [slice_89, constant_7, constant_9, constant_10]

        # pd_op.reshape_: (-1x16x4x6xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_167, combine_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf16) <- (-1x16x4x6xf16)
        transpose_91 = paddle._C_ops.transpose(reshape__138, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf16) <- (-1x4x16x6xf16)
        transpose_92 = paddle._C_ops.transpose(slice_91, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x6xf16, -1x4x6x16xf16)
        matmul_168 = paddle.matmul(slice_90, transpose_92, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x16x16xf16) <- (-1x4x16x16xf16, 1xf32)
        scale__22 = paddle._C_ops.scale(matmul_168, constant_12, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__22 = paddle._C_ops.softmax(scale__22, -1)

        # pd_op.matmul: (-1x4x16x6xf16) <- (-1x4x16x16xf16, -1x4x16x6xf16)
        matmul_169 = paddle.matmul(softmax__22, transpose_91, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x6xf16) <- (-1x4x16x6xf16)
        transpose_93 = paddle._C_ops.transpose(matmul_169, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_81 = [slice_89, constant_7, constant_13]

        # pd_op.reshape_: (-1x16x24xf16, 0x-1x16x4x6xf16) <- (-1x16x4x6xf16, [1xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_93, combine_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x24xf16, 24x24xf16)
        matmul_170 = paddle.matmul(reshape__140, parameter_334, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__125 = paddle._C_ops.add(matmul_170, parameter_335)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__126 = paddle._C_ops.add(add__118, add__125)

        # pd_op.layer_norm: (-1x16x24xf16, -16xf32, -16xf32) <- (-1x16x24xf16, 24xf32, 24xf32)
        layer_norm_207, layer_norm_208, layer_norm_209 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__126, parameter_336, parameter_337, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf16) <- (-1x16x24xf16, 24x96xf16)
        matmul_171 = paddle.matmul(layer_norm_207, parameter_338, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x96xf16) <- (-1x16x96xf16, 96xf16)
        add__127 = paddle._C_ops.add(matmul_171, parameter_339)

        # pd_op.gelu: (-1x16x96xf16) <- (-1x16x96xf16)
        gelu_22 = paddle._C_ops.gelu(add__127, False)

        # pd_op.matmul: (-1x16x24xf16) <- (-1x16x96xf16, 96x24xf16)
        matmul_172 = paddle.matmul(gelu_22, parameter_340, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, 24xf16)
        add__128 = paddle._C_ops.add(matmul_172, parameter_341)

        # pd_op.add_: (-1x16x24xf16) <- (-1x16x24xf16, -1x16x24xf16)
        add__129 = paddle._C_ops.add(add__126, add__128)

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_35 = paddle._C_ops.shape(paddle.cast(add__124, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(shape_35, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_82 = [slice_92, constant_14, constant_15]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x16x24xf16) <- (-1x16x24xf16, [1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__129, combine_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_210, layer_norm_211, layer_norm_212 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__142, parameter_342, parameter_343, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x196x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(add__124, [1], constant_1, constant_16, [1], [])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_173 = paddle.matmul(layer_norm_210, parameter_344, transpose_x=False, transpose_y=False)

        # pd_op.layer_norm: (-1x196x384xf16, -196xf32, -196xf32) <- (-1x196x384xf16, 384xf32, 384xf32)
        layer_norm_213, layer_norm_214, layer_norm_215 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_173, parameter_345, parameter_346, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, -1x196x384xf16)
        add__130 = paddle._C_ops.add(slice_93, layer_norm_213)

        # pd_op.set_value_with_tensor_: (-1x197x384xf16) <- (-1x197x384xf16, -1x196x384xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__11 = paddle._C_ops.set_value_with_tensor(add__124, add__130, constant_1, constant_17, constant_1, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_216, layer_norm_217, layer_norm_218 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__11, parameter_347, parameter_348, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf16)
        shape_36 = paddle._C_ops.shape(paddle.cast(layer_norm_216, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(shape_36, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x384xf16, 384x768xf16)
        matmul_174 = paddle.matmul(layer_norm_216, parameter_349, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_83 = [slice_94, constant_18, constant_8, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x2x6x64xf16, 0x-1x197x768xf16) <- (-1x197x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_174, combine_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf16) <- (-1x197x2x6x64xf16)
        transpose_94 = paddle._C_ops.transpose(reshape__144, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(transpose_94, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x6x197x64xf16) <- (2x-1x6x197x64xf16, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(transpose_94, [0], constant_1, constant_11, [1], [0])

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_175 = paddle.matmul(layer_norm_216, parameter_350, transpose_x=False, transpose_y=False)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_84 = [slice_94, constant_18, constant_10, constant_19]

        # pd_op.reshape_: (-1x197x6x64xf16, 0x-1x197x384xf16) <- (-1x197x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_175, combine_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf16) <- (-1x197x6x64xf16)
        transpose_95 = paddle._C_ops.transpose(reshape__146, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf16) <- (-1x6x197x64xf16)
        transpose_96 = paddle._C_ops.transpose(slice_96, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf16) <- (-1x6x197x64xf16, -1x6x64x197xf16)
        matmul_176 = paddle.matmul(slice_95, transpose_96, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x6x197x197xf16) <- (-1x6x197x197xf16, 1xf32)
        scale__23 = paddle._C_ops.scale(matmul_176, constant_20, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf16) <- (-1x6x197x197xf16)
        softmax__23 = paddle._C_ops.softmax(scale__23, -1)

        # pd_op.matmul: (-1x6x197x64xf16) <- (-1x6x197x197xf16, -1x6x197x64xf16)
        matmul_177 = paddle.matmul(softmax__23, transpose_95, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x197x6x64xf16) <- (-1x6x197x64xf16)
        transpose_97 = paddle._C_ops.transpose(matmul_177, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_85 = [slice_94, constant_18, constant_15]

        # pd_op.reshape_: (-1x197x384xf16, 0x-1x197x6x64xf16) <- (-1x197x6x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_97, combine_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x384xf16, 384x384xf16)
        matmul_178 = paddle.matmul(reshape__148, parameter_351, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__131 = paddle._C_ops.add(matmul_178, parameter_352)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__132 = paddle._C_ops.add(set_value_with_tensor__11, add__131)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_219, layer_norm_220, layer_norm_221 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__132, parameter_353, parameter_354, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf16) <- (-1x197x384xf16, 384x1536xf16)
        matmul_179 = paddle.matmul(layer_norm_219, parameter_355, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x1536xf16) <- (-1x197x1536xf16, 1536xf16)
        add__133 = paddle._C_ops.add(matmul_179, parameter_356)

        # pd_op.gelu: (-1x197x1536xf16) <- (-1x197x1536xf16)
        gelu_23 = paddle._C_ops.gelu(add__133, False)

        # pd_op.matmul: (-1x197x384xf16) <- (-1x197x1536xf16, 1536x384xf16)
        matmul_180 = paddle.matmul(gelu_23, parameter_357, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, 384xf16)
        add__134 = paddle._C_ops.add(matmul_180, parameter_358)

        # pd_op.add_: (-1x197x384xf16) <- (-1x197x384xf16, -1x197x384xf16)
        add__135 = paddle._C_ops.add(add__132, add__134)

        # pd_op.layer_norm: (-1x197x384xf16, -197xf32, -197xf32) <- (-1x197x384xf16, 384xf32, 384xf32)
        layer_norm_222, layer_norm_223, layer_norm_224 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__135, parameter_359, parameter_360, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x384xf16) <- (-1x197x384xf16, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(layer_norm_222, [1], constant_0, constant_1, [1], [1])

        # pd_op.matmul: (-1x1000xf16) <- (-1x384xf16, 384x1000xf16)
        matmul_181 = paddle.matmul(slice_97, parameter_361, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__136 = paddle._C_ops.add(matmul_181, parameter_362)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__24 = paddle._C_ops.softmax(add__136, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_3 = paddle._C_ops.cast(softmax__24, paddle.float32)
        return cast_3



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

    def forward(self, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, parameter_1, constant_2, constant_1, constant_0, parameter_0, parameter_2, parameter_4, parameter_3, parameter_5, parameter_6, parameter_8, parameter_7, parameter_9, parameter_10, parameter_12, parameter_11, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_24, parameter_23, parameter_25, parameter_27, parameter_26, parameter_29, parameter_28, parameter_30, parameter_31, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_53, parameter_52, parameter_54, parameter_56, parameter_55, parameter_58, parameter_57, parameter_59, parameter_60, parameter_61, parameter_62, parameter_64, parameter_63, parameter_65, parameter_66, parameter_67, parameter_68, parameter_70, parameter_69, parameter_71, parameter_72, parameter_73, parameter_74, parameter_76, parameter_75, parameter_77, parameter_78, parameter_79, parameter_80, parameter_82, parameter_81, parameter_83, parameter_85, parameter_84, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_114, parameter_113, parameter_116, parameter_115, parameter_117, parameter_118, parameter_119, parameter_120, parameter_122, parameter_121, parameter_123, parameter_124, parameter_125, parameter_126, parameter_128, parameter_127, parameter_129, parameter_130, parameter_131, parameter_132, parameter_134, parameter_133, parameter_135, parameter_136, parameter_137, parameter_138, parameter_140, parameter_139, parameter_141, parameter_143, parameter_142, parameter_145, parameter_144, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_155, parameter_157, parameter_156, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_172, parameter_171, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_194, parameter_195, parameter_196, parameter_198, parameter_197, parameter_199, parameter_201, parameter_200, parameter_203, parameter_202, parameter_204, parameter_205, parameter_206, parameter_207, parameter_209, parameter_208, parameter_210, parameter_211, parameter_212, parameter_213, parameter_215, parameter_214, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_227, parameter_226, parameter_228, parameter_230, parameter_229, parameter_232, parameter_231, parameter_233, parameter_234, parameter_235, parameter_236, parameter_238, parameter_237, parameter_239, parameter_240, parameter_241, parameter_242, parameter_244, parameter_243, parameter_245, parameter_246, parameter_247, parameter_248, parameter_250, parameter_249, parameter_251, parameter_252, parameter_253, parameter_254, parameter_256, parameter_255, parameter_257, parameter_259, parameter_258, parameter_261, parameter_260, parameter_262, parameter_263, parameter_264, parameter_265, parameter_267, parameter_266, parameter_268, parameter_269, parameter_270, parameter_271, parameter_273, parameter_272, parameter_274, parameter_275, parameter_276, parameter_277, parameter_279, parameter_278, parameter_280, parameter_281, parameter_282, parameter_283, parameter_285, parameter_284, parameter_286, parameter_288, parameter_287, parameter_290, parameter_289, parameter_291, parameter_292, parameter_293, parameter_294, parameter_296, parameter_295, parameter_297, parameter_298, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_305, parameter_306, parameter_308, parameter_307, parameter_309, parameter_310, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_317, parameter_316, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_346, parameter_345, parameter_348, parameter_347, parameter_349, parameter_350, parameter_351, parameter_352, parameter_354, parameter_353, parameter_355, parameter_356, parameter_357, parameter_358, parameter_360, parameter_359, parameter_361, parameter_362, feed_0):
        return self.builtin_module_1935_0_0(constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, parameter_1, constant_2, constant_1, constant_0, parameter_0, parameter_2, parameter_4, parameter_3, parameter_5, parameter_6, parameter_8, parameter_7, parameter_9, parameter_10, parameter_12, parameter_11, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_24, parameter_23, parameter_25, parameter_27, parameter_26, parameter_29, parameter_28, parameter_30, parameter_31, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_53, parameter_52, parameter_54, parameter_56, parameter_55, parameter_58, parameter_57, parameter_59, parameter_60, parameter_61, parameter_62, parameter_64, parameter_63, parameter_65, parameter_66, parameter_67, parameter_68, parameter_70, parameter_69, parameter_71, parameter_72, parameter_73, parameter_74, parameter_76, parameter_75, parameter_77, parameter_78, parameter_79, parameter_80, parameter_82, parameter_81, parameter_83, parameter_85, parameter_84, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_114, parameter_113, parameter_116, parameter_115, parameter_117, parameter_118, parameter_119, parameter_120, parameter_122, parameter_121, parameter_123, parameter_124, parameter_125, parameter_126, parameter_128, parameter_127, parameter_129, parameter_130, parameter_131, parameter_132, parameter_134, parameter_133, parameter_135, parameter_136, parameter_137, parameter_138, parameter_140, parameter_139, parameter_141, parameter_143, parameter_142, parameter_145, parameter_144, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_155, parameter_157, parameter_156, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_172, parameter_171, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_194, parameter_195, parameter_196, parameter_198, parameter_197, parameter_199, parameter_201, parameter_200, parameter_203, parameter_202, parameter_204, parameter_205, parameter_206, parameter_207, parameter_209, parameter_208, parameter_210, parameter_211, parameter_212, parameter_213, parameter_215, parameter_214, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_227, parameter_226, parameter_228, parameter_230, parameter_229, parameter_232, parameter_231, parameter_233, parameter_234, parameter_235, parameter_236, parameter_238, parameter_237, parameter_239, parameter_240, parameter_241, parameter_242, parameter_244, parameter_243, parameter_245, parameter_246, parameter_247, parameter_248, parameter_250, parameter_249, parameter_251, parameter_252, parameter_253, parameter_254, parameter_256, parameter_255, parameter_257, parameter_259, parameter_258, parameter_261, parameter_260, parameter_262, parameter_263, parameter_264, parameter_265, parameter_267, parameter_266, parameter_268, parameter_269, parameter_270, parameter_271, parameter_273, parameter_272, parameter_274, parameter_275, parameter_276, parameter_277, parameter_279, parameter_278, parameter_280, parameter_281, parameter_282, parameter_283, parameter_285, parameter_284, parameter_286, parameter_288, parameter_287, parameter_290, parameter_289, parameter_291, parameter_292, parameter_293, parameter_294, parameter_296, parameter_295, parameter_297, parameter_298, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_305, parameter_306, parameter_308, parameter_307, parameter_309, parameter_310, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_317, parameter_316, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_346, parameter_345, parameter_348, parameter_347, parameter_349, parameter_350, parameter_351, parameter_352, parameter_354, parameter_353, parameter_355, parameter_356, parameter_357, parameter_358, parameter_360, parameter_359, parameter_361, parameter_362, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1935_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_20
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_19
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_18
            paddle.to_tensor([197], dtype='int32').reshape([1]),
            # constant_17
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            # constant_16
            paddle.to_tensor([197], dtype='int64').reshape([1]),
            # constant_15
            paddle.to_tensor([384], dtype='int32').reshape([1]),
            # constant_14
            paddle.to_tensor([196], dtype='int32').reshape([1]),
            # constant_13
            paddle.to_tensor([24], dtype='int32').reshape([1]),
            # constant_12
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_11
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_10
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            # constant_9
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            # constant_8
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            # constant_7
            paddle.to_tensor([16], dtype='int32').reshape([1]),
            # constant_6
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_5
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([-1, 196, 384], dtype='int64').reshape([3]),
            # constant_3
            paddle.to_tensor([-1, 24, 16], dtype='int64').reshape([3]),
            # parameter_1
            paddle.uniform([1, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_2
            paddle.to_tensor([-1, 3, 16, 16], dtype='int64').reshape([4]),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_0
            paddle.uniform([24, 3, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_2
            paddle.uniform([1, 16, 24], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_6
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_8
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([1, 1, 384], dtype='float16', min=0, max=0.5),
            # parameter_10
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
            # parameter_12
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_16
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_32
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_47
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_60
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_62
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_66
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_70
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_82
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_96
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_108
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_120
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_122
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_132
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_140
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_160
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_176
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_180
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_182
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_192
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_195
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_198
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_206
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_212
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_230
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_235
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_236
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_240
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_242
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_246
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_247
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_248
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_250
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_252
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_265
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_267
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_270
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_273
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_275
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_276
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_281
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_282
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_283
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_285
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_296
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_299
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_300
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_302
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_306
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_308
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_310
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_312
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_314
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_317
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_321
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([24, 48], dtype='float16', min=0, max=0.5),
            # parameter_333
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([24, 24], dtype='float16', min=0, max=0.5),
            # parameter_335
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_337
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([24, 96], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_340
            paddle.uniform([96, 24], dtype='float16', min=0, max=0.5),
            # parameter_341
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_343
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_346
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([384, 768], dtype='float16', min=0, max=0.5),
            # parameter_350
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_351
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_352
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_354
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([384, 1536], dtype='float16', min=0, max=0.5),
            # parameter_356
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_357
            paddle.uniform([1536, 384], dtype='float16', min=0, max=0.5),
            # parameter_358
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_360
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([384, 1000], dtype='float16', min=0, max=0.5),
            # parameter_362
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_20
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_19
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_18
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_17
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_15
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_4
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_3
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float16'),
            # constant_2
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[24, 3, 7, 7], dtype='float16'),
            # parameter_2
            paddle.static.InputSpec(shape=[1, 16, 24], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_6
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_8
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[1, 1, 384], dtype='float16'),
            # parameter_10
            paddle.static.InputSpec(shape=[1, 197, 384], dtype='float16'),
            # parameter_12
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_16
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_32
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_47
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_60
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_62
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_66
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_70
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_82
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_96
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_108
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_120
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_122
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_132
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_140
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_160
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_176
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_180
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_182
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_192
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_195
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_198
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_206
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_212
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_230
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_235
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_236
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_240
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_242
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_246
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_247
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_248
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_250
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_252
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_265
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_267
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_270
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_273
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_275
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_276
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_281
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_282
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_283
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_285
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_296
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_299
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_300
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_302
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_306
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_308
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_310
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_312
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_314
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_317
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_321
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
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
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[24, 48], dtype='float16'),
            # parameter_333
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[24, 24], dtype='float16'),
            # parameter_335
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_337
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[24, 96], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_340
            paddle.static.InputSpec(shape=[96, 24], dtype='float16'),
            # parameter_341
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_343
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_346
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[384, 768], dtype='float16'),
            # parameter_350
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_351
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_352
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_354
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[384, 1536], dtype='float16'),
            # parameter_356
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_357
            paddle.static.InputSpec(shape=[1536, 384], dtype='float16'),
            # parameter_358
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_360
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[384, 1000], dtype='float16'),
            # parameter_362
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