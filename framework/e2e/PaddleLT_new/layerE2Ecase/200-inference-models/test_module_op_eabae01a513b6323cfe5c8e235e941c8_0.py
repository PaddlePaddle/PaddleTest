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
    return [608][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_3589_0_0(self, constant_19, parameter_339, parameter_333, parameter_331, parameter_330, parameter_328, parameter_326, parameter_324, parameter_323, parameter_321, parameter_319, parameter_317, parameter_316, parameter_314, parameter_308, parameter_302, parameter_300, parameter_299, parameter_297, parameter_295, parameter_293, parameter_292, parameter_290, parameter_288, constant_18, constant_17, parameter_286, constant_16, constant_15, parameter_285, constant_14, parameter_283, parameter_277, parameter_271, parameter_265, parameter_259, parameter_257, parameter_256, parameter_254, parameter_252, parameter_250, parameter_249, parameter_247, parameter_245, parameter_243, parameter_242, parameter_240, parameter_234, parameter_228, parameter_226, parameter_225, parameter_223, parameter_221, parameter_219, parameter_218, parameter_216, parameter_214, parameter_212, parameter_211, parameter_209, parameter_203, parameter_197, parameter_195, parameter_194, parameter_192, parameter_190, parameter_188, parameter_187, parameter_185, parameter_183, constant_13, parameter_181, parameter_180, parameter_178, parameter_172, parameter_166, parameter_160, parameter_154, parameter_152, parameter_151, parameter_149, parameter_147, parameter_145, parameter_144, parameter_142, parameter_140, parameter_138, constant_12, constant_11, parameter_137, constant_10, parameter_135, parameter_129, parameter_123, parameter_121, parameter_120, parameter_118, parameter_116, parameter_114, parameter_113, parameter_111, parameter_109, parameter_107, parameter_106, parameter_104, parameter_98, parameter_92, parameter_90, parameter_89, parameter_87, parameter_85, parameter_83, parameter_82, parameter_80, parameter_78, parameter_76, parameter_75, parameter_73, parameter_67, parameter_61, parameter_59, parameter_58, parameter_56, parameter_54, parameter_52, parameter_51, parameter_49, parameter_47, constant_9, constant_8, parameter_45, parameter_44, parameter_42, constant_7, parameter_36, parameter_30, parameter_24, parameter_22, parameter_21, parameter_19, parameter_17, parameter_15, parameter_14, parameter_12, parameter_10, constant_6, parameter_8, constant_5, constant_4, parameter_7, constant_3, parameter_5, constant_2, constant_1, constant_0, parameter_3, parameter_0, parameter_2, parameter_1, parameter_4, parameter_6, parameter_9, parameter_11, parameter_13, parameter_16, parameter_18, parameter_20, parameter_23, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_40, parameter_37, parameter_39, parameter_38, parameter_41, parameter_43, parameter_46, parameter_48, parameter_50, parameter_53, parameter_55, parameter_57, parameter_60, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_74, parameter_77, parameter_79, parameter_81, parameter_84, parameter_86, parameter_88, parameter_91, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_102, parameter_99, parameter_101, parameter_100, parameter_103, parameter_105, parameter_108, parameter_110, parameter_112, parameter_115, parameter_117, parameter_119, parameter_122, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_136, parameter_139, parameter_141, parameter_143, parameter_146, parameter_148, parameter_150, parameter_153, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_179, parameter_182, parameter_184, parameter_186, parameter_189, parameter_191, parameter_193, parameter_196, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_210, parameter_213, parameter_215, parameter_217, parameter_220, parameter_222, parameter_224, parameter_227, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_241, parameter_244, parameter_246, parameter_248, parameter_251, parameter_253, parameter_255, parameter_258, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_284, parameter_287, parameter_289, parameter_291, parameter_294, parameter_296, parameter_298, parameter_301, parameter_306, parameter_303, parameter_305, parameter_304, parameter_307, parameter_312, parameter_309, parameter_311, parameter_310, parameter_313, parameter_315, parameter_318, parameter_320, parameter_322, parameter_325, parameter_327, parameter_329, parameter_332, parameter_337, parameter_334, parameter_336, parameter_335, parameter_338, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_345, feed_0):

        # pd_op.cast: (1x2x350x25x1xf16) <- (1x2x350x25x1xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.transpose: (1x1x25x2x350xf16) <- (1x2x350x25x1xf16)
        transpose_0 = paddle._C_ops.transpose(cast_0, [0, 4, 3, 1, 2])

        # pd_op.reshape_: (1x50x350xf16, 0x1x1x25x2x350xf16) <- (1x1x25x2x350xf16, 3xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (1x50x350xf16, 50xf32, 50xf32, xf32, xf32, None) <- (1x50x350xf16, 50xf32, 50xf32, 50xf32, 50xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__0, parameter_0, parameter_1, parameter_2, parameter_3, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.reshape_: (1x1x25x2x350xf16, 0x1x50x350xf16) <- (1x50x350xf16, 5xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__0, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (1x1x2x350x25xf16) <- (1x1x25x2x350xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__2, [0, 1, 3, 4, 2])

        # pd_op.reshape_: (1x2x350x25xf16, 0x1x1x2x350x25xf16) <- (1x1x2x350x25xf16, 4xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x2x350x25xf16, 16x2x1x1xf16)
        conv2d_0 = paddle._C_ops.conv2d(reshape__4, parameter_4, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, parameter_5)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_2 = paddle._C_ops.transpose(add__0, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_2, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x2x350x25xf16, 16x2x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(reshape__4, parameter_6, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_1, parameter_7)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__1, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_0 = paddle.matmul(reshape__6, reshape__8, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_0, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__0 = paddle._C_ops.softmax_(scale__0, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__2 = paddle._C_ops.add_(softmax__0, parameter_8)

        # pd_op.reshape: (1x700x25xf16, 0x1x2x350x25xf16) <- (1x2x350x25xf16, 3xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__4, constant_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x700x25xf16) <- (1x700x25xf16, 1x25x25xf16)
        matmul_1 = paddle.matmul(reshape_0, add__2, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x2x350x25xf16, 0x1x700x25xf16) <- (1x700x25xf16, 4xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_1, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x2x350x25xf16, 64x2x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(reshape__10, parameter_9, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__3 = paddle._C_ops.add_(conv2d_2, parameter_10)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x2x350x25xf16, 16x2x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(reshape__4, parameter_11, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_3, parameter_12)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_3 = paddle._C_ops.transpose(add__4, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_3, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x2x350x25xf16, 16x2x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(reshape__4, parameter_13, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_4, parameter_14)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__5, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_2 = paddle.matmul(reshape__12, reshape__14, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_2, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__1 = paddle._C_ops.softmax_(scale__1, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__6 = paddle._C_ops.add_(softmax__1, parameter_15)

        # pd_op.reshape: (1x700x25xf16, 0x1x2x350x25xf16) <- (1x2x350x25xf16, 3xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__4, constant_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x700x25xf16) <- (1x700x25xf16, 1x25x25xf16)
        matmul_3 = paddle.matmul(reshape_2, add__6, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x2x350x25xf16, 0x1x700x25xf16) <- (1x700x25xf16, 4xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_3, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x2x350x25xf16, 64x2x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(reshape__16, parameter_16, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__7 = paddle._C_ops.add_(conv2d_5, parameter_17)

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__8 = paddle._C_ops.add_(add__7, add__3)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x2x350x25xf16, 16x2x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(reshape__4, parameter_18, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__9 = paddle._C_ops.add_(conv2d_6, parameter_19)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_4 = paddle._C_ops.transpose(add__9, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x2x350x25xf16, 16x2x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(reshape__4, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_7, parameter_21)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__10, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_4 = paddle.matmul(reshape__18, reshape__20, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_4, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__2 = paddle._C_ops.softmax_(scale__2, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__11 = paddle._C_ops.add_(softmax__2, parameter_22)

        # pd_op.reshape: (1x700x25xf16, 0x1x2x350x25xf16) <- (1x2x350x25xf16, 3xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__4, constant_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x700x25xf16) <- (1x700x25xf16, 1x25x25xf16)
        matmul_5 = paddle.matmul(reshape_4, add__11, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x2x350x25xf16, 0x1x700x25xf16) <- (1x700x25xf16, 4xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_5, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x2x350x25xf16, 64x2x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(reshape__22, parameter_23, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__12 = paddle._C_ops.add_(conv2d_8, parameter_24)

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__13 = paddle._C_ops.add_(add__12, add__8)

        # pd_op.batch_norm_: (1x64x350x25xf16, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__13, parameter_25, parameter_26, parameter_27, parameter_28, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x2x350x25xf16, 64x2x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(reshape__4, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_9, parameter_30)

        # pd_op.batch_norm_: (1x64x350x25xf16, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__14, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__15 = paddle._C_ops.add_(batch_norm__6, batch_norm__12)

        # pd_op.relu_: (1x64x350x25xf16) <- (1x64x350x25xf16)
        relu__0 = paddle._C_ops.relu_(add__15)

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x9x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu__0, parameter_35, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__16 = paddle._C_ops.add_(conv2d_10, parameter_36)

        # pd_op.batch_norm_: (1x64x350x25xf16, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__16, parameter_37, parameter_38, parameter_39, parameter_40, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.scale_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(batch_norm__18, constant_7, float('0'), True)

        # pd_op.relu_: (1x64x350x25xf16) <- (1x64x350x25xf16)
        relu__1 = paddle._C_ops.relu_(scale__3)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(relu__1, parameter_41, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_11, parameter_42)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_5 = paddle._C_ops.transpose(add__17, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_5, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu__1, parameter_43, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__18 = paddle._C_ops.add_(conv2d_12, parameter_44)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__18, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_6 = paddle.matmul(reshape__24, reshape__26, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_6, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__3 = paddle._C_ops.softmax_(scale__4, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__19 = paddle._C_ops.add_(softmax__3, parameter_45)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__1, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_7 = paddle.matmul(reshape_6, add__19, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_7, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(reshape__28, parameter_46, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__20 = paddle._C_ops.add_(conv2d_13, parameter_47)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu__1, parameter_48, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__21 = paddle._C_ops.add_(conv2d_14, parameter_49)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_6 = paddle._C_ops.transpose(add__21, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu__1, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__22 = paddle._C_ops.add_(conv2d_15, parameter_51)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__22, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_8 = paddle.matmul(reshape__30, reshape__32, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_8, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__4 = paddle._C_ops.softmax_(scale__5, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__23 = paddle._C_ops.add_(softmax__4, parameter_52)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__1, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_9 = paddle.matmul(reshape_8, add__23, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_9, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(reshape__34, parameter_53, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__24 = paddle._C_ops.add_(conv2d_16, parameter_54)

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__25 = paddle._C_ops.add_(add__24, add__20)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu__1, parameter_55, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__26 = paddle._C_ops.add_(conv2d_17, parameter_56)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_7 = paddle._C_ops.transpose(add__26, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(relu__1, parameter_57, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__27 = paddle._C_ops.add_(conv2d_18, parameter_58)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__27, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_10 = paddle.matmul(reshape__36, reshape__38, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(matmul_10, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__5 = paddle._C_ops.softmax_(scale__6, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__28 = paddle._C_ops.add_(softmax__5, parameter_59)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__1, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_11 = paddle.matmul(reshape_10, add__28, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_11, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(reshape__40, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__29 = paddle._C_ops.add_(conv2d_19, parameter_61)

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__30 = paddle._C_ops.add_(add__29, add__25)

        # pd_op.batch_norm_: (1x64x350x25xf16, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__30, parameter_62, parameter_63, parameter_64, parameter_65, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__31 = paddle._C_ops.add_(batch_norm__24, relu__1)

        # pd_op.relu_: (1x64x350x25xf16) <- (1x64x350x25xf16)
        relu__2 = paddle._C_ops.relu_(add__31)

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x9x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(relu__2, parameter_66, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__32 = paddle._C_ops.add_(conv2d_20, parameter_67)

        # pd_op.batch_norm_: (1x64x350x25xf16, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__32, parameter_68, parameter_69, parameter_70, parameter_71, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__33 = paddle._C_ops.add_(batch_norm__30, relu__1)

        # pd_op.relu_: (1x64x350x25xf16) <- (1x64x350x25xf16)
        relu__3 = paddle._C_ops.relu_(add__33)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu__3, parameter_72, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__34 = paddle._C_ops.add_(conv2d_21, parameter_73)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_8 = paddle._C_ops.transpose(add__34, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_8, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(relu__3, parameter_74, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__35 = paddle._C_ops.add_(conv2d_22, parameter_75)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__35, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_12 = paddle.matmul(reshape__42, reshape__44, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(matmul_12, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__6 = paddle._C_ops.softmax_(scale__7, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__36 = paddle._C_ops.add_(softmax__6, parameter_76)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__3, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_13 = paddle.matmul(reshape_12, add__36, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_13, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(reshape__46, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__37 = paddle._C_ops.add_(conv2d_23, parameter_78)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(relu__3, parameter_79, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__38 = paddle._C_ops.add_(conv2d_24, parameter_80)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_9 = paddle._C_ops.transpose(add__38, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__3, parameter_81, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__39 = paddle._C_ops.add_(conv2d_25, parameter_82)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__39, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_14 = paddle.matmul(reshape__48, reshape__50, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__8 = paddle._C_ops.scale_(matmul_14, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__7 = paddle._C_ops.softmax_(scale__8, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__40 = paddle._C_ops.add_(softmax__7, parameter_83)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__3, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_15 = paddle.matmul(reshape_14, add__40, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_15, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(reshape__52, parameter_84, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__41 = paddle._C_ops.add_(conv2d_26, parameter_85)

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__42 = paddle._C_ops.add_(add__41, add__37)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu__3, parameter_86, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__43 = paddle._C_ops.add_(conv2d_27, parameter_87)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_10 = paddle._C_ops.transpose(add__43, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu__3, parameter_88, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__44 = paddle._C_ops.add_(conv2d_28, parameter_89)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__44, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_16 = paddle.matmul(reshape__54, reshape__56, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__9 = paddle._C_ops.scale_(matmul_16, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__8 = paddle._C_ops.softmax_(scale__9, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__45 = paddle._C_ops.add_(softmax__8, parameter_90)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__3, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_17 = paddle.matmul(reshape_16, add__45, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_17, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(reshape__58, parameter_91, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__46 = paddle._C_ops.add_(conv2d_29, parameter_92)

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__47 = paddle._C_ops.add_(add__46, add__42)

        # pd_op.batch_norm_: (1x64x350x25xf16, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__47, parameter_93, parameter_94, parameter_95, parameter_96, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__48 = paddle._C_ops.add_(batch_norm__36, relu__3)

        # pd_op.relu_: (1x64x350x25xf16) <- (1x64x350x25xf16)
        relu__4 = paddle._C_ops.relu_(add__48)

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x9x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu__4, parameter_97, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__49 = paddle._C_ops.add_(conv2d_30, parameter_98)

        # pd_op.batch_norm_: (1x64x350x25xf16, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__49, parameter_99, parameter_100, parameter_101, parameter_102, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__50 = paddle._C_ops.add_(batch_norm__42, relu__3)

        # pd_op.relu_: (1x64x350x25xf16) <- (1x64x350x25xf16)
        relu__5 = paddle._C_ops.relu_(add__50)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu__5, parameter_103, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__51 = paddle._C_ops.add_(conv2d_31, parameter_104)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_11 = paddle._C_ops.transpose(add__51, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_11, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(relu__5, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__52 = paddle._C_ops.add_(conv2d_32, parameter_106)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__52, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_18 = paddle.matmul(reshape__60, reshape__62, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__10 = paddle._C_ops.scale_(matmul_18, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__9 = paddle._C_ops.softmax_(scale__10, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__53 = paddle._C_ops.add_(softmax__9, parameter_107)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__5, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_19 = paddle.matmul(reshape_18, add__53, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_19, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(reshape__64, parameter_108, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__54 = paddle._C_ops.add_(conv2d_33, parameter_109)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(relu__5, parameter_110, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__55 = paddle._C_ops.add_(conv2d_34, parameter_111)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_12 = paddle._C_ops.transpose(add__55, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_12, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu__5, parameter_112, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__56 = paddle._C_ops.add_(conv2d_35, parameter_113)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__56, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_20 = paddle.matmul(reshape__66, reshape__68, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__11 = paddle._C_ops.scale_(matmul_20, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__10 = paddle._C_ops.softmax_(scale__11, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__57 = paddle._C_ops.add_(softmax__10, parameter_114)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__5, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_21 = paddle.matmul(reshape_20, add__57, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_21, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(reshape__70, parameter_115, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__58 = paddle._C_ops.add_(conv2d_36, parameter_116)

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__59 = paddle._C_ops.add_(add__58, add__54)

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(relu__5, parameter_117, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__60 = paddle._C_ops.add_(conv2d_37, parameter_118)

        # pd_op.transpose: (1x25x16x350xf16) <- (1x16x350x25xf16)
        transpose_13 = paddle._C_ops.transpose(add__60, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x16x350xf16) <- (1x25x16x350xf16, 3xi64)
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_13, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf16) <- (1x64x350x25xf16, 16x64x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(relu__5, parameter_119, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x350x25xf16) <- (1x16x350x25xf16, 1x16x1x1xf16)
        add__61 = paddle._C_ops.add_(conv2d_38, parameter_120)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x16x350x25xf16) <- (1x16x350x25xf16, 3xi64)
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__61, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_22 = paddle.matmul(reshape__72, reshape__74, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__12 = paddle._C_ops.scale_(matmul_22, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__11 = paddle._C_ops.softmax_(scale__12, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__62 = paddle._C_ops.add_(softmax__11, parameter_121)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__5, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_23 = paddle.matmul(reshape_22, add__62, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_23, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(reshape__76, parameter_122, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__63 = paddle._C_ops.add_(conv2d_39, parameter_123)

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__64 = paddle._C_ops.add_(add__63, add__59)

        # pd_op.batch_norm_: (1x64x350x25xf16, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__64, parameter_124, parameter_125, parameter_126, parameter_127, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__65 = paddle._C_ops.add_(batch_norm__48, relu__5)

        # pd_op.relu_: (1x64x350x25xf16) <- (1x64x350x25xf16)
        relu__6 = paddle._C_ops.relu_(add__65)

        # pd_op.conv2d: (1x64x350x25xf16) <- (1x64x350x25xf16, 64x64x9x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu__6, parameter_128, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x1x1xf16)
        add__66 = paddle._C_ops.add_(conv2d_40, parameter_129)

        # pd_op.batch_norm_: (1x64x350x25xf16, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__66, parameter_130, parameter_131, parameter_132, parameter_133, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf16) <- (1x64x350x25xf16, 1x64x350x25xf16)
        add__67 = paddle._C_ops.add_(batch_norm__54, relu__5)

        # pd_op.relu_: (1x64x350x25xf16) <- (1x64x350x25xf16)
        relu__7 = paddle._C_ops.relu_(add__67)

        # pd_op.conv2d: (1x32x350x25xf16) <- (1x64x350x25xf16, 32x64x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(relu__7, parameter_134, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x350x25xf16) <- (1x32x350x25xf16, 1x32x1x1xf16)
        add__68 = paddle._C_ops.add_(conv2d_41, parameter_135)

        # pd_op.transpose: (1x25x32x350xf16) <- (1x32x350x25xf16)
        transpose_14 = paddle._C_ops.transpose(add__68, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x11200xf16, 0x1x25x32x350xf16) <- (1x25x32x350xf16, 3xi64)
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, constant_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x350x25xf16) <- (1x64x350x25xf16, 32x64x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(relu__7, parameter_136, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x350x25xf16) <- (1x32x350x25xf16, 1x32x1x1xf16)
        add__69 = paddle._C_ops.add_(conv2d_42, parameter_137)

        # pd_op.reshape_: (1x11200x25xf16, 0x1x32x350x25xf16) <- (1x32x350x25xf16, 3xi64)
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__69, constant_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x11200xf16, 1x11200x25xf16)
        matmul_24 = paddle.matmul(reshape__78, reshape__80, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__13 = paddle._C_ops.scale_(matmul_24, constant_12, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__12 = paddle._C_ops.softmax_(scale__13, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__70 = paddle._C_ops.add_(softmax__12, parameter_138)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__7, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_25 = paddle.matmul(reshape_24, add__70, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_25, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x350x25xf16) <- (1x64x350x25xf16, 128x64x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(reshape__82, parameter_139, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x350x25xf16) <- (1x128x350x25xf16, 1x128x1x1xf16)
        add__71 = paddle._C_ops.add_(conv2d_43, parameter_140)

        # pd_op.conv2d: (1x32x350x25xf16) <- (1x64x350x25xf16, 32x64x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(relu__7, parameter_141, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x350x25xf16) <- (1x32x350x25xf16, 1x32x1x1xf16)
        add__72 = paddle._C_ops.add_(conv2d_44, parameter_142)

        # pd_op.transpose: (1x25x32x350xf16) <- (1x32x350x25xf16)
        transpose_15 = paddle._C_ops.transpose(add__72, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x11200xf16, 0x1x25x32x350xf16) <- (1x25x32x350xf16, 3xi64)
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_15, constant_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x350x25xf16) <- (1x64x350x25xf16, 32x64x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(relu__7, parameter_143, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x350x25xf16) <- (1x32x350x25xf16, 1x32x1x1xf16)
        add__73 = paddle._C_ops.add_(conv2d_45, parameter_144)

        # pd_op.reshape_: (1x11200x25xf16, 0x1x32x350x25xf16) <- (1x32x350x25xf16, 3xi64)
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__73, constant_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x11200xf16, 1x11200x25xf16)
        matmul_26 = paddle.matmul(reshape__84, reshape__86, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__14 = paddle._C_ops.scale_(matmul_26, constant_12, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__13 = paddle._C_ops.softmax_(scale__14, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__74 = paddle._C_ops.add_(softmax__13, parameter_145)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__7, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_27 = paddle.matmul(reshape_26, add__74, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_27, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x350x25xf16) <- (1x64x350x25xf16, 128x64x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(reshape__88, parameter_146, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x350x25xf16) <- (1x128x350x25xf16, 1x128x1x1xf16)
        add__75 = paddle._C_ops.add_(conv2d_46, parameter_147)

        # pd_op.add_: (1x128x350x25xf16) <- (1x128x350x25xf16, 1x128x350x25xf16)
        add__76 = paddle._C_ops.add_(add__75, add__71)

        # pd_op.conv2d: (1x32x350x25xf16) <- (1x64x350x25xf16, 32x64x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu__7, parameter_148, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x350x25xf16) <- (1x32x350x25xf16, 1x32x1x1xf16)
        add__77 = paddle._C_ops.add_(conv2d_47, parameter_149)

        # pd_op.transpose: (1x25x32x350xf16) <- (1x32x350x25xf16)
        transpose_16 = paddle._C_ops.transpose(add__77, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x11200xf16, 0x1x25x32x350xf16) <- (1x25x32x350xf16, 3xi64)
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_16, constant_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x350x25xf16) <- (1x64x350x25xf16, 32x64x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(relu__7, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x350x25xf16) <- (1x32x350x25xf16, 1x32x1x1xf16)
        add__78 = paddle._C_ops.add_(conv2d_48, parameter_151)

        # pd_op.reshape_: (1x11200x25xf16, 0x1x32x350x25xf16) <- (1x32x350x25xf16, 3xi64)
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__78, constant_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x11200xf16, 1x11200x25xf16)
        matmul_28 = paddle.matmul(reshape__90, reshape__92, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__15 = paddle._C_ops.scale_(matmul_28, constant_12, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__14 = paddle._C_ops.softmax_(scale__15, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__79 = paddle._C_ops.add_(softmax__14, parameter_152)

        # pd_op.reshape: (1x22400x25xf16, 0x1x64x350x25xf16) <- (1x64x350x25xf16, 3xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__7, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_29 = paddle.matmul(reshape_28, add__79, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x64x350x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_29, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x350x25xf16) <- (1x64x350x25xf16, 128x64x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(reshape__94, parameter_153, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x350x25xf16) <- (1x128x350x25xf16, 1x128x1x1xf16)
        add__80 = paddle._C_ops.add_(conv2d_49, parameter_154)

        # pd_op.add_: (1x128x350x25xf16) <- (1x128x350x25xf16, 1x128x350x25xf16)
        add__81 = paddle._C_ops.add_(add__80, add__76)

        # pd_op.batch_norm_: (1x128x350x25xf16, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x350x25xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__81, parameter_155, parameter_156, parameter_157, parameter_158, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x128x350x25xf16) <- (1x64x350x25xf16, 128x64x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(relu__7, parameter_159, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x350x25xf16) <- (1x128x350x25xf16, 1x128x1x1xf16)
        add__82 = paddle._C_ops.add_(conv2d_50, parameter_160)

        # pd_op.batch_norm_: (1x128x350x25xf16, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x350x25xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__82, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x350x25xf16) <- (1x128x350x25xf16, 1x128x350x25xf16)
        add__83 = paddle._C_ops.add_(batch_norm__60, batch_norm__66)

        # pd_op.relu_: (1x128x350x25xf16) <- (1x128x350x25xf16)
        relu__8 = paddle._C_ops.relu_(add__83)

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x128x350x25xf16, 128x128x9x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu__8, parameter_165, [2, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__84 = paddle._C_ops.add_(conv2d_51, parameter_166)

        # pd_op.batch_norm_: (1x128x175x25xf16, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__84, parameter_167, parameter_168, parameter_169, parameter_170, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x64x350x25xf16, 128x64x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(relu__7, parameter_171, [2, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__85 = paddle._C_ops.add_(conv2d_52, parameter_172)

        # pd_op.batch_norm_: (1x128x175x25xf16, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__85, parameter_173, parameter_174, parameter_175, parameter_176, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x175x25xf16)
        add__86 = paddle._C_ops.add_(batch_norm__72, batch_norm__78)

        # pd_op.relu_: (1x128x175x25xf16) <- (1x128x175x25xf16)
        relu__9 = paddle._C_ops.relu_(add__86)

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_53 = paddle._C_ops.conv2d(relu__9, parameter_177, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__87 = paddle._C_ops.add_(conv2d_53, parameter_178)

        # pd_op.transpose: (1x25x32x175xf16) <- (1x32x175x25xf16)
        transpose_17 = paddle._C_ops.transpose(add__87, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x32x175xf16) <- (1x25x32x175xf16, 3xi64)
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_17, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(relu__9, parameter_179, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__88 = paddle._C_ops.add_(conv2d_54, parameter_180)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x32x175x25xf16) <- (1x32x175x25xf16, 3xi64)
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__88, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_30 = paddle.matmul(reshape__96, reshape__98, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__16 = paddle._C_ops.scale_(matmul_30, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__15 = paddle._C_ops.softmax_(scale__16, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__89 = paddle._C_ops.add_(softmax__15, parameter_181)

        # pd_op.reshape: (1x22400x25xf16, 0x1x128x175x25xf16) <- (1x128x175x25xf16, 3xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__9, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_31 = paddle.matmul(reshape_30, add__89, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x128x175x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_31, constant_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x128x175x25xf16, 128x128x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(reshape__100, parameter_182, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__90 = paddle._C_ops.add_(conv2d_55, parameter_183)

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_56 = paddle._C_ops.conv2d(relu__9, parameter_184, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__91 = paddle._C_ops.add_(conv2d_56, parameter_185)

        # pd_op.transpose: (1x25x32x175xf16) <- (1x32x175x25xf16)
        transpose_18 = paddle._C_ops.transpose(add__91, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x32x175xf16) <- (1x25x32x175xf16, 3xi64)
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_18, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_57 = paddle._C_ops.conv2d(relu__9, parameter_186, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__92 = paddle._C_ops.add_(conv2d_57, parameter_187)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x32x175x25xf16) <- (1x32x175x25xf16, 3xi64)
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__92, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_32 = paddle.matmul(reshape__102, reshape__104, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__17 = paddle._C_ops.scale_(matmul_32, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__16 = paddle._C_ops.softmax_(scale__17, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__93 = paddle._C_ops.add_(softmax__16, parameter_188)

        # pd_op.reshape: (1x22400x25xf16, 0x1x128x175x25xf16) <- (1x128x175x25xf16, 3xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__9, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_33 = paddle.matmul(reshape_32, add__93, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x128x175x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_33, constant_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x128x175x25xf16, 128x128x1x1xf16)
        conv2d_58 = paddle._C_ops.conv2d(reshape__106, parameter_189, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__94 = paddle._C_ops.add_(conv2d_58, parameter_190)

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x175x25xf16)
        add__95 = paddle._C_ops.add_(add__94, add__90)

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_59 = paddle._C_ops.conv2d(relu__9, parameter_191, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__96 = paddle._C_ops.add_(conv2d_59, parameter_192)

        # pd_op.transpose: (1x25x32x175xf16) <- (1x32x175x25xf16)
        transpose_19 = paddle._C_ops.transpose(add__96, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x32x175xf16) <- (1x25x32x175xf16, 3xi64)
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_19, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(relu__9, parameter_193, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__97 = paddle._C_ops.add_(conv2d_60, parameter_194)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x32x175x25xf16) <- (1x32x175x25xf16, 3xi64)
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__97, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_34 = paddle.matmul(reshape__108, reshape__110, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__18 = paddle._C_ops.scale_(matmul_34, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__17 = paddle._C_ops.softmax_(scale__18, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__98 = paddle._C_ops.add_(softmax__17, parameter_195)

        # pd_op.reshape: (1x22400x25xf16, 0x1x128x175x25xf16) <- (1x128x175x25xf16, 3xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__9, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_35 = paddle.matmul(reshape_34, add__98, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x128x175x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_35, constant_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x128x175x25xf16, 128x128x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(reshape__112, parameter_196, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__99 = paddle._C_ops.add_(conv2d_61, parameter_197)

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x175x25xf16)
        add__100 = paddle._C_ops.add_(add__99, add__95)

        # pd_op.batch_norm_: (1x128x175x25xf16, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__100, parameter_198, parameter_199, parameter_200, parameter_201, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x175x25xf16)
        add__101 = paddle._C_ops.add_(batch_norm__84, relu__9)

        # pd_op.relu_: (1x128x175x25xf16) <- (1x128x175x25xf16)
        relu__10 = paddle._C_ops.relu_(add__101)

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x128x175x25xf16, 128x128x9x1xf16)
        conv2d_62 = paddle._C_ops.conv2d(relu__10, parameter_202, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__102 = paddle._C_ops.add_(conv2d_62, parameter_203)

        # pd_op.batch_norm_: (1x128x175x25xf16, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__102, parameter_204, parameter_205, parameter_206, parameter_207, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x175x25xf16)
        add__103 = paddle._C_ops.add_(batch_norm__90, relu__9)

        # pd_op.relu_: (1x128x175x25xf16) <- (1x128x175x25xf16)
        relu__11 = paddle._C_ops.relu_(add__103)

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_63 = paddle._C_ops.conv2d(relu__11, parameter_208, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__104 = paddle._C_ops.add_(conv2d_63, parameter_209)

        # pd_op.transpose: (1x25x32x175xf16) <- (1x32x175x25xf16)
        transpose_20 = paddle._C_ops.transpose(add__104, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x32x175xf16) <- (1x25x32x175xf16, 3xi64)
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_64 = paddle._C_ops.conv2d(relu__11, parameter_210, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__105 = paddle._C_ops.add_(conv2d_64, parameter_211)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x32x175x25xf16) <- (1x32x175x25xf16, 3xi64)
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__105, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_36 = paddle.matmul(reshape__114, reshape__116, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__19 = paddle._C_ops.scale_(matmul_36, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__18 = paddle._C_ops.softmax_(scale__19, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__106 = paddle._C_ops.add_(softmax__18, parameter_212)

        # pd_op.reshape: (1x22400x25xf16, 0x1x128x175x25xf16) <- (1x128x175x25xf16, 3xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__11, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_37 = paddle.matmul(reshape_36, add__106, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x128x175x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_37, constant_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x128x175x25xf16, 128x128x1x1xf16)
        conv2d_65 = paddle._C_ops.conv2d(reshape__118, parameter_213, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__107 = paddle._C_ops.add_(conv2d_65, parameter_214)

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_66 = paddle._C_ops.conv2d(relu__11, parameter_215, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__108 = paddle._C_ops.add_(conv2d_66, parameter_216)

        # pd_op.transpose: (1x25x32x175xf16) <- (1x32x175x25xf16)
        transpose_21 = paddle._C_ops.transpose(add__108, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x32x175xf16) <- (1x25x32x175xf16, 3xi64)
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_21, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_67 = paddle._C_ops.conv2d(relu__11, parameter_217, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__109 = paddle._C_ops.add_(conv2d_67, parameter_218)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x32x175x25xf16) <- (1x32x175x25xf16, 3xi64)
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__109, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_38 = paddle.matmul(reshape__120, reshape__122, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__20 = paddle._C_ops.scale_(matmul_38, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__19 = paddle._C_ops.softmax_(scale__20, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__110 = paddle._C_ops.add_(softmax__19, parameter_219)

        # pd_op.reshape: (1x22400x25xf16, 0x1x128x175x25xf16) <- (1x128x175x25xf16, 3xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__11, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_39 = paddle.matmul(reshape_38, add__110, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x128x175x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_39, constant_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x128x175x25xf16, 128x128x1x1xf16)
        conv2d_68 = paddle._C_ops.conv2d(reshape__124, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__111 = paddle._C_ops.add_(conv2d_68, parameter_221)

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x175x25xf16)
        add__112 = paddle._C_ops.add_(add__111, add__107)

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_69 = paddle._C_ops.conv2d(relu__11, parameter_222, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__113 = paddle._C_ops.add_(conv2d_69, parameter_223)

        # pd_op.transpose: (1x25x32x175xf16) <- (1x32x175x25xf16)
        transpose_22 = paddle._C_ops.transpose(add__113, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5600xf16, 0x1x25x32x175xf16) <- (1x25x32x175xf16, 3xi64)
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_22, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf16) <- (1x128x175x25xf16, 32x128x1x1xf16)
        conv2d_70 = paddle._C_ops.conv2d(relu__11, parameter_224, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x175x25xf16) <- (1x32x175x25xf16, 1x32x1x1xf16)
        add__114 = paddle._C_ops.add_(conv2d_70, parameter_225)

        # pd_op.reshape_: (1x5600x25xf16, 0x1x32x175x25xf16) <- (1x32x175x25xf16, 3xi64)
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__114, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5600xf16, 1x5600x25xf16)
        matmul_40 = paddle.matmul(reshape__126, reshape__128, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__21 = paddle._C_ops.scale_(matmul_40, constant_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__20 = paddle._C_ops.softmax_(scale__21, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__115 = paddle._C_ops.add_(softmax__20, parameter_226)

        # pd_op.reshape: (1x22400x25xf16, 0x1x128x175x25xf16) <- (1x128x175x25xf16, 3xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__11, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_41 = paddle.matmul(reshape_40, add__115, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x128x175x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_41, constant_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x128x175x25xf16, 128x128x1x1xf16)
        conv2d_71 = paddle._C_ops.conv2d(reshape__130, parameter_227, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__116 = paddle._C_ops.add_(conv2d_71, parameter_228)

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x175x25xf16)
        add__117 = paddle._C_ops.add_(add__116, add__112)

        # pd_op.batch_norm_: (1x128x175x25xf16, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__117, parameter_229, parameter_230, parameter_231, parameter_232, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x175x25xf16)
        add__118 = paddle._C_ops.add_(batch_norm__96, relu__11)

        # pd_op.relu_: (1x128x175x25xf16) <- (1x128x175x25xf16)
        relu__12 = paddle._C_ops.relu_(add__118)

        # pd_op.conv2d: (1x128x175x25xf16) <- (1x128x175x25xf16, 128x128x9x1xf16)
        conv2d_72 = paddle._C_ops.conv2d(relu__12, parameter_233, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x1x1xf16)
        add__119 = paddle._C_ops.add_(conv2d_72, parameter_234)

        # pd_op.batch_norm_: (1x128x175x25xf16, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__119, parameter_235, parameter_236, parameter_237, parameter_238, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf16) <- (1x128x175x25xf16, 1x128x175x25xf16)
        add__120 = paddle._C_ops.add_(batch_norm__102, relu__11)

        # pd_op.relu_: (1x128x175x25xf16) <- (1x128x175x25xf16)
        relu__13 = paddle._C_ops.relu_(add__120)

        # pd_op.conv2d: (1x64x175x25xf16) <- (1x128x175x25xf16, 64x128x1x1xf16)
        conv2d_73 = paddle._C_ops.conv2d(relu__13, parameter_239, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x175x25xf16) <- (1x64x175x25xf16, 1x64x1x1xf16)
        add__121 = paddle._C_ops.add_(conv2d_73, parameter_240)

        # pd_op.transpose: (1x25x64x175xf16) <- (1x64x175x25xf16)
        transpose_23 = paddle._C_ops.transpose(add__121, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x11200xf16, 0x1x25x64x175xf16) <- (1x25x64x175xf16, 3xi64)
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_23, constant_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x175x25xf16) <- (1x128x175x25xf16, 64x128x1x1xf16)
        conv2d_74 = paddle._C_ops.conv2d(relu__13, parameter_241, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x175x25xf16) <- (1x64x175x25xf16, 1x64x1x1xf16)
        add__122 = paddle._C_ops.add_(conv2d_74, parameter_242)

        # pd_op.reshape_: (1x11200x25xf16, 0x1x64x175x25xf16) <- (1x64x175x25xf16, 3xi64)
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__122, constant_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x11200xf16, 1x11200x25xf16)
        matmul_42 = paddle.matmul(reshape__132, reshape__134, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__22 = paddle._C_ops.scale_(matmul_42, constant_12, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__21 = paddle._C_ops.softmax_(scale__22, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__123 = paddle._C_ops.add_(softmax__21, parameter_243)

        # pd_op.reshape: (1x22400x25xf16, 0x1x128x175x25xf16) <- (1x128x175x25xf16, 3xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__13, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_43 = paddle.matmul(reshape_42, add__123, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x128x175x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_43, constant_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x175x25xf16) <- (1x128x175x25xf16, 256x128x1x1xf16)
        conv2d_75 = paddle._C_ops.conv2d(reshape__136, parameter_244, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x175x25xf16) <- (1x256x175x25xf16, 1x256x1x1xf16)
        add__124 = paddle._C_ops.add_(conv2d_75, parameter_245)

        # pd_op.conv2d: (1x64x175x25xf16) <- (1x128x175x25xf16, 64x128x1x1xf16)
        conv2d_76 = paddle._C_ops.conv2d(relu__13, parameter_246, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x175x25xf16) <- (1x64x175x25xf16, 1x64x1x1xf16)
        add__125 = paddle._C_ops.add_(conv2d_76, parameter_247)

        # pd_op.transpose: (1x25x64x175xf16) <- (1x64x175x25xf16)
        transpose_24 = paddle._C_ops.transpose(add__125, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x11200xf16, 0x1x25x64x175xf16) <- (1x25x64x175xf16, 3xi64)
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, constant_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x175x25xf16) <- (1x128x175x25xf16, 64x128x1x1xf16)
        conv2d_77 = paddle._C_ops.conv2d(relu__13, parameter_248, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x175x25xf16) <- (1x64x175x25xf16, 1x64x1x1xf16)
        add__126 = paddle._C_ops.add_(conv2d_77, parameter_249)

        # pd_op.reshape_: (1x11200x25xf16, 0x1x64x175x25xf16) <- (1x64x175x25xf16, 3xi64)
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__126, constant_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x11200xf16, 1x11200x25xf16)
        matmul_44 = paddle.matmul(reshape__138, reshape__140, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__23 = paddle._C_ops.scale_(matmul_44, constant_12, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__22 = paddle._C_ops.softmax_(scale__23, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__127 = paddle._C_ops.add_(softmax__22, parameter_250)

        # pd_op.reshape: (1x22400x25xf16, 0x1x128x175x25xf16) <- (1x128x175x25xf16, 3xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__13, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_45 = paddle.matmul(reshape_44, add__127, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x128x175x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_45, constant_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x175x25xf16) <- (1x128x175x25xf16, 256x128x1x1xf16)
        conv2d_78 = paddle._C_ops.conv2d(reshape__142, parameter_251, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x175x25xf16) <- (1x256x175x25xf16, 1x256x1x1xf16)
        add__128 = paddle._C_ops.add_(conv2d_78, parameter_252)

        # pd_op.add_: (1x256x175x25xf16) <- (1x256x175x25xf16, 1x256x175x25xf16)
        add__129 = paddle._C_ops.add_(add__128, add__124)

        # pd_op.conv2d: (1x64x175x25xf16) <- (1x128x175x25xf16, 64x128x1x1xf16)
        conv2d_79 = paddle._C_ops.conv2d(relu__13, parameter_253, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x175x25xf16) <- (1x64x175x25xf16, 1x64x1x1xf16)
        add__130 = paddle._C_ops.add_(conv2d_79, parameter_254)

        # pd_op.transpose: (1x25x64x175xf16) <- (1x64x175x25xf16)
        transpose_25 = paddle._C_ops.transpose(add__130, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x11200xf16, 0x1x25x64x175xf16) <- (1x25x64x175xf16, 3xi64)
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_25, constant_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x175x25xf16) <- (1x128x175x25xf16, 64x128x1x1xf16)
        conv2d_80 = paddle._C_ops.conv2d(relu__13, parameter_255, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x175x25xf16) <- (1x64x175x25xf16, 1x64x1x1xf16)
        add__131 = paddle._C_ops.add_(conv2d_80, parameter_256)

        # pd_op.reshape_: (1x11200x25xf16, 0x1x64x175x25xf16) <- (1x64x175x25xf16, 3xi64)
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__131, constant_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x11200xf16, 1x11200x25xf16)
        matmul_46 = paddle.matmul(reshape__144, reshape__146, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__24 = paddle._C_ops.scale_(matmul_46, constant_12, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__23 = paddle._C_ops.softmax_(scale__24, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__132 = paddle._C_ops.add_(softmax__23, parameter_257)

        # pd_op.reshape: (1x22400x25xf16, 0x1x128x175x25xf16) <- (1x128x175x25xf16, 3xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__13, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf16) <- (1x22400x25xf16, 1x25x25xf16)
        matmul_47 = paddle.matmul(reshape_46, add__132, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x128x175x25xf16, 0x1x22400x25xf16) <- (1x22400x25xf16, 4xi64)
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_47, constant_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x175x25xf16) <- (1x128x175x25xf16, 256x128x1x1xf16)
        conv2d_81 = paddle._C_ops.conv2d(reshape__148, parameter_258, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x175x25xf16) <- (1x256x175x25xf16, 1x256x1x1xf16)
        add__133 = paddle._C_ops.add_(conv2d_81, parameter_259)

        # pd_op.add_: (1x256x175x25xf16) <- (1x256x175x25xf16, 1x256x175x25xf16)
        add__134 = paddle._C_ops.add_(add__133, add__129)

        # pd_op.batch_norm_: (1x256x175x25xf16, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x175x25xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__134, parameter_260, parameter_261, parameter_262, parameter_263, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x256x175x25xf16) <- (1x128x175x25xf16, 256x128x1x1xf16)
        conv2d_82 = paddle._C_ops.conv2d(relu__13, parameter_264, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x175x25xf16) <- (1x256x175x25xf16, 1x256x1x1xf16)
        add__135 = paddle._C_ops.add_(conv2d_82, parameter_265)

        # pd_op.batch_norm_: (1x256x175x25xf16, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x175x25xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__135, parameter_266, parameter_267, parameter_268, parameter_269, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x175x25xf16) <- (1x256x175x25xf16, 1x256x175x25xf16)
        add__136 = paddle._C_ops.add_(batch_norm__108, batch_norm__114)

        # pd_op.relu_: (1x256x175x25xf16) <- (1x256x175x25xf16)
        relu__14 = paddle._C_ops.relu_(add__136)

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x256x175x25xf16, 256x256x9x1xf16)
        conv2d_83 = paddle._C_ops.conv2d(relu__14, parameter_270, [2, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__137 = paddle._C_ops.add_(conv2d_83, parameter_271)

        # pd_op.batch_norm_: (1x256x88x25xf16, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__137, parameter_272, parameter_273, parameter_274, parameter_275, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x128x175x25xf16, 256x128x1x1xf16)
        conv2d_84 = paddle._C_ops.conv2d(relu__13, parameter_276, [2, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__138 = paddle._C_ops.add_(conv2d_84, parameter_277)

        # pd_op.batch_norm_: (1x256x88x25xf16, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__138, parameter_278, parameter_279, parameter_280, parameter_281, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x88x25xf16)
        add__139 = paddle._C_ops.add_(batch_norm__120, batch_norm__126)

        # pd_op.relu_: (1x256x88x25xf16) <- (1x256x88x25xf16)
        relu__15 = paddle._C_ops.relu_(add__139)

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_85 = paddle._C_ops.conv2d(relu__15, parameter_282, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__140 = paddle._C_ops.add_(conv2d_85, parameter_283)

        # pd_op.transpose: (1x25x64x88xf16) <- (1x64x88x25xf16)
        transpose_26 = paddle._C_ops.transpose(add__140, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5632xf16, 0x1x25x64x88xf16) <- (1x25x64x88xf16, 3xi64)
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_26, constant_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_86 = paddle._C_ops.conv2d(relu__15, parameter_284, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__141 = paddle._C_ops.add_(conv2d_86, parameter_285)

        # pd_op.reshape_: (1x5632x25xf16, 0x1x64x88x25xf16) <- (1x64x88x25xf16, 3xi64)
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__141, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5632xf16, 1x5632x25xf16)
        matmul_48 = paddle.matmul(reshape__150, reshape__152, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__25 = paddle._C_ops.scale_(matmul_48, constant_16, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__24 = paddle._C_ops.softmax_(scale__25, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__142 = paddle._C_ops.add_(softmax__24, parameter_286)

        # pd_op.reshape: (1x22528x25xf16, 0x1x256x88x25xf16) <- (1x256x88x25xf16, 3xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__15, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf16) <- (1x22528x25xf16, 1x25x25xf16)
        matmul_49 = paddle.matmul(reshape_48, add__142, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x256x88x25xf16, 0x1x22528x25xf16) <- (1x22528x25xf16, 4xi64)
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_49, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x256x88x25xf16, 256x256x1x1xf16)
        conv2d_87 = paddle._C_ops.conv2d(reshape__154, parameter_287, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__143 = paddle._C_ops.add_(conv2d_87, parameter_288)

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_88 = paddle._C_ops.conv2d(relu__15, parameter_289, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__144 = paddle._C_ops.add_(conv2d_88, parameter_290)

        # pd_op.transpose: (1x25x64x88xf16) <- (1x64x88x25xf16)
        transpose_27 = paddle._C_ops.transpose(add__144, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5632xf16, 0x1x25x64x88xf16) <- (1x25x64x88xf16, 3xi64)
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_27, constant_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_89 = paddle._C_ops.conv2d(relu__15, parameter_291, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__145 = paddle._C_ops.add_(conv2d_89, parameter_292)

        # pd_op.reshape_: (1x5632x25xf16, 0x1x64x88x25xf16) <- (1x64x88x25xf16, 3xi64)
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__145, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5632xf16, 1x5632x25xf16)
        matmul_50 = paddle.matmul(reshape__156, reshape__158, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__26 = paddle._C_ops.scale_(matmul_50, constant_16, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__25 = paddle._C_ops.softmax_(scale__26, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__146 = paddle._C_ops.add_(softmax__25, parameter_293)

        # pd_op.reshape: (1x22528x25xf16, 0x1x256x88x25xf16) <- (1x256x88x25xf16, 3xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__15, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf16) <- (1x22528x25xf16, 1x25x25xf16)
        matmul_51 = paddle.matmul(reshape_50, add__146, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x256x88x25xf16, 0x1x22528x25xf16) <- (1x22528x25xf16, 4xi64)
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_51, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x256x88x25xf16, 256x256x1x1xf16)
        conv2d_90 = paddle._C_ops.conv2d(reshape__160, parameter_294, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__147 = paddle._C_ops.add_(conv2d_90, parameter_295)

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x88x25xf16)
        add__148 = paddle._C_ops.add_(add__147, add__143)

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_91 = paddle._C_ops.conv2d(relu__15, parameter_296, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__149 = paddle._C_ops.add_(conv2d_91, parameter_297)

        # pd_op.transpose: (1x25x64x88xf16) <- (1x64x88x25xf16)
        transpose_28 = paddle._C_ops.transpose(add__149, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5632xf16, 0x1x25x64x88xf16) <- (1x25x64x88xf16, 3xi64)
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, constant_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_92 = paddle._C_ops.conv2d(relu__15, parameter_298, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__150 = paddle._C_ops.add_(conv2d_92, parameter_299)

        # pd_op.reshape_: (1x5632x25xf16, 0x1x64x88x25xf16) <- (1x64x88x25xf16, 3xi64)
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__150, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5632xf16, 1x5632x25xf16)
        matmul_52 = paddle.matmul(reshape__162, reshape__164, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__27 = paddle._C_ops.scale_(matmul_52, constant_16, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__26 = paddle._C_ops.softmax_(scale__27, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__151 = paddle._C_ops.add_(softmax__26, parameter_300)

        # pd_op.reshape: (1x22528x25xf16, 0x1x256x88x25xf16) <- (1x256x88x25xf16, 3xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__15, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf16) <- (1x22528x25xf16, 1x25x25xf16)
        matmul_53 = paddle.matmul(reshape_52, add__151, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x256x88x25xf16, 0x1x22528x25xf16) <- (1x22528x25xf16, 4xi64)
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_53, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x256x88x25xf16, 256x256x1x1xf16)
        conv2d_93 = paddle._C_ops.conv2d(reshape__166, parameter_301, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__152 = paddle._C_ops.add_(conv2d_93, parameter_302)

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x88x25xf16)
        add__153 = paddle._C_ops.add_(add__152, add__148)

        # pd_op.batch_norm_: (1x256x88x25xf16, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__153, parameter_303, parameter_304, parameter_305, parameter_306, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x88x25xf16)
        add__154 = paddle._C_ops.add_(batch_norm__132, relu__15)

        # pd_op.relu_: (1x256x88x25xf16) <- (1x256x88x25xf16)
        relu__16 = paddle._C_ops.relu_(add__154)

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x256x88x25xf16, 256x256x9x1xf16)
        conv2d_94 = paddle._C_ops.conv2d(relu__16, parameter_307, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__155 = paddle._C_ops.add_(conv2d_94, parameter_308)

        # pd_op.batch_norm_: (1x256x88x25xf16, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__155, parameter_309, parameter_310, parameter_311, parameter_312, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x88x25xf16)
        add__156 = paddle._C_ops.add_(batch_norm__138, relu__15)

        # pd_op.relu_: (1x256x88x25xf16) <- (1x256x88x25xf16)
        relu__17 = paddle._C_ops.relu_(add__156)

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_95 = paddle._C_ops.conv2d(relu__17, parameter_313, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__157 = paddle._C_ops.add_(conv2d_95, parameter_314)

        # pd_op.transpose: (1x25x64x88xf16) <- (1x64x88x25xf16)
        transpose_29 = paddle._C_ops.transpose(add__157, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5632xf16, 0x1x25x64x88xf16) <- (1x25x64x88xf16, 3xi64)
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_29, constant_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_96 = paddle._C_ops.conv2d(relu__17, parameter_315, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__158 = paddle._C_ops.add_(conv2d_96, parameter_316)

        # pd_op.reshape_: (1x5632x25xf16, 0x1x64x88x25xf16) <- (1x64x88x25xf16, 3xi64)
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__158, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5632xf16, 1x5632x25xf16)
        matmul_54 = paddle.matmul(reshape__168, reshape__170, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__28 = paddle._C_ops.scale_(matmul_54, constant_16, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__27 = paddle._C_ops.softmax_(scale__28, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__159 = paddle._C_ops.add_(softmax__27, parameter_317)

        # pd_op.reshape: (1x22528x25xf16, 0x1x256x88x25xf16) <- (1x256x88x25xf16, 3xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__17, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf16) <- (1x22528x25xf16, 1x25x25xf16)
        matmul_55 = paddle.matmul(reshape_54, add__159, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x256x88x25xf16, 0x1x22528x25xf16) <- (1x22528x25xf16, 4xi64)
        reshape__172, reshape__173 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_55, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x256x88x25xf16, 256x256x1x1xf16)
        conv2d_97 = paddle._C_ops.conv2d(reshape__172, parameter_318, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__160 = paddle._C_ops.add_(conv2d_97, parameter_319)

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_98 = paddle._C_ops.conv2d(relu__17, parameter_320, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__161 = paddle._C_ops.add_(conv2d_98, parameter_321)

        # pd_op.transpose: (1x25x64x88xf16) <- (1x64x88x25xf16)
        transpose_30 = paddle._C_ops.transpose(add__161, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5632xf16, 0x1x25x64x88xf16) <- (1x25x64x88xf16, 3xi64)
        reshape__174, reshape__175 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_30, constant_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_99 = paddle._C_ops.conv2d(relu__17, parameter_322, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__162 = paddle._C_ops.add_(conv2d_99, parameter_323)

        # pd_op.reshape_: (1x5632x25xf16, 0x1x64x88x25xf16) <- (1x64x88x25xf16, 3xi64)
        reshape__176, reshape__177 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__162, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5632xf16, 1x5632x25xf16)
        matmul_56 = paddle.matmul(reshape__174, reshape__176, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__29 = paddle._C_ops.scale_(matmul_56, constant_16, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__28 = paddle._C_ops.softmax_(scale__29, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__163 = paddle._C_ops.add_(softmax__28, parameter_324)

        # pd_op.reshape: (1x22528x25xf16, 0x1x256x88x25xf16) <- (1x256x88x25xf16, 3xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__17, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf16) <- (1x22528x25xf16, 1x25x25xf16)
        matmul_57 = paddle.matmul(reshape_56, add__163, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x256x88x25xf16, 0x1x22528x25xf16) <- (1x22528x25xf16, 4xi64)
        reshape__178, reshape__179 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_57, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x256x88x25xf16, 256x256x1x1xf16)
        conv2d_100 = paddle._C_ops.conv2d(reshape__178, parameter_325, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__164 = paddle._C_ops.add_(conv2d_100, parameter_326)

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x88x25xf16)
        add__165 = paddle._C_ops.add_(add__164, add__160)

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_101 = paddle._C_ops.conv2d(relu__17, parameter_327, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__166 = paddle._C_ops.add_(conv2d_101, parameter_328)

        # pd_op.transpose: (1x25x64x88xf16) <- (1x64x88x25xf16)
        transpose_31 = paddle._C_ops.transpose(add__166, [0, 3, 1, 2])

        # pd_op.reshape_: (1x25x5632xf16, 0x1x25x64x88xf16) <- (1x25x64x88xf16, 3xi64)
        reshape__180, reshape__181 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_31, constant_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf16) <- (1x256x88x25xf16, 64x256x1x1xf16)
        conv2d_102 = paddle._C_ops.conv2d(relu__17, parameter_329, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x64x88x25xf16) <- (1x64x88x25xf16, 1x64x1x1xf16)
        add__167 = paddle._C_ops.add_(conv2d_102, parameter_330)

        # pd_op.reshape_: (1x5632x25xf16, 0x1x64x88x25xf16) <- (1x64x88x25xf16, 3xi64)
        reshape__182, reshape__183 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__167, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf16) <- (1x25x5632xf16, 1x5632x25xf16)
        matmul_58 = paddle.matmul(reshape__180, reshape__182, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (1x25x25xf16) <- (1x25x25xf16, 1xf32)
        scale__30 = paddle._C_ops.scale_(matmul_58, constant_16, float('0'), True)

        # pd_op.softmax_: (1x25x25xf16) <- (1x25x25xf16)
        softmax__29 = paddle._C_ops.softmax_(scale__30, -2)

        # pd_op.add_: (1x25x25xf16) <- (1x25x25xf16, 25x25xf16)
        add__168 = paddle._C_ops.add_(softmax__29, parameter_331)

        # pd_op.reshape: (1x22528x25xf16, 0x1x256x88x25xf16) <- (1x256x88x25xf16, 3xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__17, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf16) <- (1x22528x25xf16, 1x25x25xf16)
        matmul_59 = paddle.matmul(reshape_58, add__168, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (1x256x88x25xf16, 0x1x22528x25xf16) <- (1x22528x25xf16, 4xi64)
        reshape__184, reshape__185 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_59, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x256x88x25xf16, 256x256x1x1xf16)
        conv2d_103 = paddle._C_ops.conv2d(reshape__184, parameter_332, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__169 = paddle._C_ops.add_(conv2d_103, parameter_333)

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x88x25xf16)
        add__170 = paddle._C_ops.add_(add__169, add__165)

        # pd_op.batch_norm_: (1x256x88x25xf16, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__170, parameter_334, parameter_335, parameter_336, parameter_337, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x88x25xf16)
        add__171 = paddle._C_ops.add_(batch_norm__144, relu__17)

        # pd_op.relu_: (1x256x88x25xf16) <- (1x256x88x25xf16)
        relu__18 = paddle._C_ops.relu_(add__171)

        # pd_op.conv2d: (1x256x88x25xf16) <- (1x256x88x25xf16, 256x256x9x1xf16)
        conv2d_104 = paddle._C_ops.conv2d(relu__18, parameter_338, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x1x1xf16)
        add__172 = paddle._C_ops.add_(conv2d_104, parameter_339)

        # pd_op.batch_norm_: (1x256x88x25xf16, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__172, parameter_340, parameter_341, parameter_342, parameter_343, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf16) <- (1x256x88x25xf16, 1x256x88x25xf16)
        add__173 = paddle._C_ops.add_(batch_norm__150, relu__17)

        # pd_op.relu_: (1x256x88x25xf16) <- (1x256x88x25xf16)
        relu__19 = paddle._C_ops.relu_(add__173)

        # pd_op.reshape_: (1x1x256x2200xf16, 0x1x256x88x25xf16) <- (1x256x88x25xf16, 4xi64)
        reshape__186, reshape__187 = (lambda x, f: f(x))(paddle._C_ops.reshape_(relu__19, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.mean: (1x1x256xf16) <- (1x1x256x2200xf16)
        mean_0 = paddle._C_ops.mean(reshape__186, [3], False)

        # pd_op.mean: (1x256xf16) <- (1x1x256xf16)
        mean_1 = paddle._C_ops.mean(mean_0, [1], False)

        # pd_op.matmul: (1x60xf16) <- (1x256xf16, 256x60xf16)
        matmul_60 = paddle.matmul(mean_1, parameter_344, transpose_x=False, transpose_y=False)

        # pd_op.add_: (1x60xf16) <- (1x60xf16, 60xf16)
        add__174 = paddle._C_ops.add_(matmul_60, parameter_345)

        # pd_op.cast: (1x60xf32) <- (1x60xf16)
        cast_1 = paddle._C_ops.cast(add__174, paddle.float32)
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

    def forward(self, constant_19, parameter_339, parameter_333, parameter_331, parameter_330, parameter_328, parameter_326, parameter_324, parameter_323, parameter_321, parameter_319, parameter_317, parameter_316, parameter_314, parameter_308, parameter_302, parameter_300, parameter_299, parameter_297, parameter_295, parameter_293, parameter_292, parameter_290, parameter_288, constant_18, constant_17, parameter_286, constant_16, constant_15, parameter_285, constant_14, parameter_283, parameter_277, parameter_271, parameter_265, parameter_259, parameter_257, parameter_256, parameter_254, parameter_252, parameter_250, parameter_249, parameter_247, parameter_245, parameter_243, parameter_242, parameter_240, parameter_234, parameter_228, parameter_226, parameter_225, parameter_223, parameter_221, parameter_219, parameter_218, parameter_216, parameter_214, parameter_212, parameter_211, parameter_209, parameter_203, parameter_197, parameter_195, parameter_194, parameter_192, parameter_190, parameter_188, parameter_187, parameter_185, parameter_183, constant_13, parameter_181, parameter_180, parameter_178, parameter_172, parameter_166, parameter_160, parameter_154, parameter_152, parameter_151, parameter_149, parameter_147, parameter_145, parameter_144, parameter_142, parameter_140, parameter_138, constant_12, constant_11, parameter_137, constant_10, parameter_135, parameter_129, parameter_123, parameter_121, parameter_120, parameter_118, parameter_116, parameter_114, parameter_113, parameter_111, parameter_109, parameter_107, parameter_106, parameter_104, parameter_98, parameter_92, parameter_90, parameter_89, parameter_87, parameter_85, parameter_83, parameter_82, parameter_80, parameter_78, parameter_76, parameter_75, parameter_73, parameter_67, parameter_61, parameter_59, parameter_58, parameter_56, parameter_54, parameter_52, parameter_51, parameter_49, parameter_47, constant_9, constant_8, parameter_45, parameter_44, parameter_42, constant_7, parameter_36, parameter_30, parameter_24, parameter_22, parameter_21, parameter_19, parameter_17, parameter_15, parameter_14, parameter_12, parameter_10, constant_6, parameter_8, constant_5, constant_4, parameter_7, constant_3, parameter_5, constant_2, constant_1, constant_0, parameter_3, parameter_0, parameter_2, parameter_1, parameter_4, parameter_6, parameter_9, parameter_11, parameter_13, parameter_16, parameter_18, parameter_20, parameter_23, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_40, parameter_37, parameter_39, parameter_38, parameter_41, parameter_43, parameter_46, parameter_48, parameter_50, parameter_53, parameter_55, parameter_57, parameter_60, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_74, parameter_77, parameter_79, parameter_81, parameter_84, parameter_86, parameter_88, parameter_91, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_102, parameter_99, parameter_101, parameter_100, parameter_103, parameter_105, parameter_108, parameter_110, parameter_112, parameter_115, parameter_117, parameter_119, parameter_122, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_136, parameter_139, parameter_141, parameter_143, parameter_146, parameter_148, parameter_150, parameter_153, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_179, parameter_182, parameter_184, parameter_186, parameter_189, parameter_191, parameter_193, parameter_196, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_210, parameter_213, parameter_215, parameter_217, parameter_220, parameter_222, parameter_224, parameter_227, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_241, parameter_244, parameter_246, parameter_248, parameter_251, parameter_253, parameter_255, parameter_258, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_284, parameter_287, parameter_289, parameter_291, parameter_294, parameter_296, parameter_298, parameter_301, parameter_306, parameter_303, parameter_305, parameter_304, parameter_307, parameter_312, parameter_309, parameter_311, parameter_310, parameter_313, parameter_315, parameter_318, parameter_320, parameter_322, parameter_325, parameter_327, parameter_329, parameter_332, parameter_337, parameter_334, parameter_336, parameter_335, parameter_338, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_345, feed_0):
        return self.builtin_module_3589_0_0(constant_19, parameter_339, parameter_333, parameter_331, parameter_330, parameter_328, parameter_326, parameter_324, parameter_323, parameter_321, parameter_319, parameter_317, parameter_316, parameter_314, parameter_308, parameter_302, parameter_300, parameter_299, parameter_297, parameter_295, parameter_293, parameter_292, parameter_290, parameter_288, constant_18, constant_17, parameter_286, constant_16, constant_15, parameter_285, constant_14, parameter_283, parameter_277, parameter_271, parameter_265, parameter_259, parameter_257, parameter_256, parameter_254, parameter_252, parameter_250, parameter_249, parameter_247, parameter_245, parameter_243, parameter_242, parameter_240, parameter_234, parameter_228, parameter_226, parameter_225, parameter_223, parameter_221, parameter_219, parameter_218, parameter_216, parameter_214, parameter_212, parameter_211, parameter_209, parameter_203, parameter_197, parameter_195, parameter_194, parameter_192, parameter_190, parameter_188, parameter_187, parameter_185, parameter_183, constant_13, parameter_181, parameter_180, parameter_178, parameter_172, parameter_166, parameter_160, parameter_154, parameter_152, parameter_151, parameter_149, parameter_147, parameter_145, parameter_144, parameter_142, parameter_140, parameter_138, constant_12, constant_11, parameter_137, constant_10, parameter_135, parameter_129, parameter_123, parameter_121, parameter_120, parameter_118, parameter_116, parameter_114, parameter_113, parameter_111, parameter_109, parameter_107, parameter_106, parameter_104, parameter_98, parameter_92, parameter_90, parameter_89, parameter_87, parameter_85, parameter_83, parameter_82, parameter_80, parameter_78, parameter_76, parameter_75, parameter_73, parameter_67, parameter_61, parameter_59, parameter_58, parameter_56, parameter_54, parameter_52, parameter_51, parameter_49, parameter_47, constant_9, constant_8, parameter_45, parameter_44, parameter_42, constant_7, parameter_36, parameter_30, parameter_24, parameter_22, parameter_21, parameter_19, parameter_17, parameter_15, parameter_14, parameter_12, parameter_10, constant_6, parameter_8, constant_5, constant_4, parameter_7, constant_3, parameter_5, constant_2, constant_1, constant_0, parameter_3, parameter_0, parameter_2, parameter_1, parameter_4, parameter_6, parameter_9, parameter_11, parameter_13, parameter_16, parameter_18, parameter_20, parameter_23, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_40, parameter_37, parameter_39, parameter_38, parameter_41, parameter_43, parameter_46, parameter_48, parameter_50, parameter_53, parameter_55, parameter_57, parameter_60, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_74, parameter_77, parameter_79, parameter_81, parameter_84, parameter_86, parameter_88, parameter_91, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_102, parameter_99, parameter_101, parameter_100, parameter_103, parameter_105, parameter_108, parameter_110, parameter_112, parameter_115, parameter_117, parameter_119, parameter_122, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_136, parameter_139, parameter_141, parameter_143, parameter_146, parameter_148, parameter_150, parameter_153, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_179, parameter_182, parameter_184, parameter_186, parameter_189, parameter_191, parameter_193, parameter_196, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_210, parameter_213, parameter_215, parameter_217, parameter_220, parameter_222, parameter_224, parameter_227, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_241, parameter_244, parameter_246, parameter_248, parameter_251, parameter_253, parameter_255, parameter_258, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_284, parameter_287, parameter_289, parameter_291, parameter_294, parameter_296, parameter_298, parameter_301, parameter_306, parameter_303, parameter_305, parameter_304, parameter_307, parameter_312, parameter_309, parameter_311, parameter_310, parameter_313, parameter_315, parameter_318, parameter_320, parameter_322, parameter_325, parameter_327, parameter_329, parameter_332, parameter_337, parameter_334, parameter_336, parameter_335, parameter_338, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_345, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_3589_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_19
            paddle.to_tensor([1, 1, 256, -1], dtype='int64').reshape([4]),
            # parameter_339
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_333
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_331
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_330
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_328
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_326
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_324
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_319
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_317
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_316
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_314
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_308
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_302
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_300
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_299
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_297
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_295
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_290
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_18
            paddle.to_tensor([1, 256, 88, 25], dtype='int64').reshape([4]),
            # constant_17
            paddle.to_tensor([1, 22528, 25], dtype='int64').reshape([3]),
            # parameter_286
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # constant_16
            paddle.to_tensor([0.000177557], dtype='float32').reshape([1]),
            # constant_15
            paddle.to_tensor([1, 5632, 25], dtype='int64').reshape([3]),
            # parameter_285
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_14
            paddle.to_tensor([1, 25, 5632], dtype='int64').reshape([3]),
            # parameter_283
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_265
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_257
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_252
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_250
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_247
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_245
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_242
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_240
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_228
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_226
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_216
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_212
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_197
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_195
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_192
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_187
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_185
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_13
            paddle.to_tensor([1, 128, 175, 25], dtype='int64').reshape([4]),
            # parameter_181
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_180
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_160
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_142
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_140
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # constant_12
            paddle.to_tensor([8.92857e-05], dtype='float32').reshape([1]),
            # constant_11
            paddle.to_tensor([1, 11200, 25], dtype='int64').reshape([3]),
            # parameter_137
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_10
            paddle.to_tensor([1, 25, 11200], dtype='int64').reshape([3]),
            # parameter_135
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_120
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_98
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_92
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_82
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_47
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_9
            paddle.to_tensor([1, 64, 350, 25], dtype='int64').reshape([4]),
            # constant_8
            paddle.to_tensor([1, 22400, 25], dtype='int64').reshape([3]),
            # parameter_45
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_42
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_7
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            # parameter_36
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_30
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_17
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_12
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_10
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_6
            paddle.to_tensor([1, 700, 25], dtype='int64').reshape([3]),
            # parameter_8
            paddle.uniform([25, 25], dtype='float16', min=0, max=0.5),
            # constant_5
            paddle.to_tensor([0.000178571], dtype='float32').reshape([1]),
            # constant_4
            paddle.to_tensor([1, 5600, 25], dtype='int64').reshape([3]),
            # parameter_7
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_3
            paddle.to_tensor([1, 25, 5600], dtype='int64').reshape([3]),
            # parameter_5
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_2
            paddle.to_tensor([1, 2, 350, 25], dtype='int64').reshape([4]),
            # constant_1
            paddle.to_tensor([1, 1, 25, 2, 350], dtype='int64').reshape([5]),
            # constant_0
            paddle.to_tensor([1, 50, 350], dtype='int64').reshape([3]),
            # parameter_3
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([16, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_6
            paddle.uniform([16, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([16, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([16, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([16, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([16, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([64, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64, 64, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_40
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_48
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64, 64, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_77
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([64, 64, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_108
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_110
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_112
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([16, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_122
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([64, 64, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_150
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([128, 128, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_170
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_176
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_182
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([128, 128, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_210
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_220
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_222
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([32, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_232
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([128, 128, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_246
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_248
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_255
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_258
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([256, 256, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_275
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_281
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_284
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_287
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_291
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_296
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_301
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_306
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([256, 256, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_312
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_315
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_318
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_320
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_322
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_325
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_327
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_329
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_332
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_337
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([256, 256, 9, 1], dtype='float16', min=0, max=0.5),
            # parameter_343
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([256, 60], dtype='float16', min=0, max=0.5),
            # parameter_345
            paddle.uniform([60], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 2, 350, 25, 1], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_19
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # parameter_339
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_333
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_331
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_330
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_328
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_326
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_324
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_323
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_321
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_319
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_317
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_316
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_314
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_308
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_302
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_300
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_299
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_297
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_295
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_290
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # constant_18
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_17
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_286
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_15
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_285
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # constant_14
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_283
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_265
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_257
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_252
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_250
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_247
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_245
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_242
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_240
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_228
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_226
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_216
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_212
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_197
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_195
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_192
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_187
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_185
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # constant_13
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # parameter_181
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_180
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_160
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_142
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_140
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_11
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_137
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # constant_10
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_135
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_120
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_98
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_92
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_82
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_47
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # constant_9
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_8
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_45
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_42
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_30
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_17
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_12
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_10
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # constant_6
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_8
            paddle.static.InputSpec(shape=[25, 25], dtype='float16'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_4
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_7
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # constant_3
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_5
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # constant_2
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_1
            paddle.static.InputSpec(shape=[5], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_3
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float16'),
            # parameter_6
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[64, 2, 1, 1], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float16'),
            # parameter_16
            paddle.static.InputSpec(shape=[64, 2, 1, 1], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[64, 2, 1, 1], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[64, 2, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float16'),
            # parameter_40
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_46
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_48
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_60
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_77
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_96
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_108
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_110
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_112
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float16'),
            # parameter_122
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_150
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[128, 128, 9, 1], dtype='float16'),
            # parameter_170
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_176
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_182
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[128, 128, 9, 1], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_210
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_220
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_222
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_232
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[128, 128, 9, 1], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_246
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_248
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_255
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_258
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[256, 256, 9, 1], dtype='float16'),
            # parameter_275
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_281
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_284
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_287
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_291
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_296
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_301
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_306
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[256, 256, 9, 1], dtype='float16'),
            # parameter_312
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_315
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_318
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_320
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_322
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_325
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_327
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_329
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_332
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_337
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[256, 256, 9, 1], dtype='float16'),
            # parameter_343
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[256, 60], dtype='float16'),
            # parameter_345
            paddle.static.InputSpec(shape=[60], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[1, 2, 350, 25, 1], dtype='float32'),
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