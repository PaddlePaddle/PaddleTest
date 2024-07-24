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
    return [325][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1449_0_0(self, constant_2, constant_1, parameter_355, parameter_353, parameter_336, parameter_334, parameter_317, parameter_315, parameter_298, parameter_296, parameter_279, parameter_277, parameter_260, parameter_258, parameter_241, parameter_239, parameter_222, parameter_220, parameter_203, parameter_201, parameter_184, parameter_182, parameter_165, parameter_163, parameter_146, parameter_144, parameter_127, parameter_125, parameter_108, parameter_106, parameter_89, parameter_87, parameter_70, parameter_68, parameter_51, parameter_49, parameter_32, parameter_30, parameter_13, parameter_11, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_12, parameter_14, parameter_18, parameter_15, parameter_17, parameter_16, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_31, parameter_33, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_50, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_69, parameter_71, parameter_75, parameter_72, parameter_74, parameter_73, parameter_76, parameter_80, parameter_77, parameter_79, parameter_78, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_88, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_107, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_126, parameter_128, parameter_132, parameter_129, parameter_131, parameter_130, parameter_133, parameter_137, parameter_134, parameter_136, parameter_135, parameter_138, parameter_142, parameter_139, parameter_141, parameter_140, parameter_143, parameter_145, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_164, parameter_166, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_175, parameter_172, parameter_174, parameter_173, parameter_176, parameter_180, parameter_177, parameter_179, parameter_178, parameter_181, parameter_183, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_202, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_221, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_240, parameter_242, parameter_246, parameter_243, parameter_245, parameter_244, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_259, parameter_261, parameter_265, parameter_262, parameter_264, parameter_263, parameter_266, parameter_270, parameter_267, parameter_269, parameter_268, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_278, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_297, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_316, parameter_318, parameter_322, parameter_319, parameter_321, parameter_320, parameter_323, parameter_327, parameter_324, parameter_326, parameter_325, parameter_328, parameter_332, parameter_329, parameter_331, parameter_330, parameter_333, parameter_335, parameter_337, parameter_341, parameter_338, parameter_340, parameter_339, parameter_342, parameter_346, parameter_343, parameter_345, parameter_344, parameter_347, parameter_351, parameter_348, parameter_350, parameter_349, parameter_352, parameter_354, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_365, parameter_362, parameter_364, parameter_363, parameter_366, parameter_367, feed_0):

        # pd_op.conv2d: (-1x32x96x96xf32) <- (-1x3x192x192xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x96x96xf32, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x96x96xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x32x96x96xf32) <- (-1x32x96x96xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.depthwise_conv2d: (-1x32x96x96xf32) <- (-1x32x96x96xf32, 32x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(swish_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x96x96xf32, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x96x96xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x32x96x96xf32) <- (-1x32x96x96xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.pool2d: (-1x32x1x1xf32) <- (-1x32x96x96xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(swish_1, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x8x1x1xf32) <- (-1x32x1x1xf32, 8x32x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(pool2d_0, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x8x1x1xf32) <- (-1x8x1x1xf32, 1x8x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_1, parameter_11)

        # pd_op.swish: (-1x8x1x1xf32) <- (-1x8x1x1xf32)
        swish_2 = paddle._C_ops.swish(add__0)

        # pd_op.conv2d: (-1x32x1x1xf32) <- (-1x8x1x1xf32, 32x8x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(swish_2, parameter_12, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x1x1xf32) <- (-1x32x1x1xf32, 1x32x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_2, parameter_13)

        # pd_op.sigmoid_: (-1x32x1x1xf32) <- (-1x32x1x1xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(add__1)

        # pd_op.multiply_: (-1x32x96x96xf32) <- (-1x32x96x96xf32, -1x32x1x1xf32)
        multiply__0 = paddle._C_ops.multiply_(swish_1, sigmoid__0)

        # pd_op.conv2d: (-1x16x96x96xf32) <- (-1x32x96x96xf32, 16x32x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(multiply__0, parameter_14, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x96x96xf32, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x96x96xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_15, parameter_16, parameter_17, parameter_18, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x96x96x96xf32) <- (-1x16x96x96xf32, 96x16x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(batch_norm__12, parameter_19, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x96x96xf32, 96xf32, 96xf32, 96xf32, 96xf32, None) <- (-1x96x96x96xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_20, parameter_21, parameter_22, parameter_23, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x96x96xf32) <- (-1x96x96x96xf32)
        swish_3 = paddle._C_ops.swish(batch_norm__18)

        # pd_op.depthwise_conv2d: (-1x96x48x48xf32) <- (-1x96x96x96xf32, 96x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(swish_3, parameter_24, [2, 2], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x48x48xf32, 96xf32, 96xf32, 96xf32, 96xf32, None) <- (-1x96x48x48xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_25, parameter_26, parameter_27, parameter_28, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x48x48xf32) <- (-1x96x48x48xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__24)

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x48x48xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(swish_4, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x4x1x1xf32) <- (-1x96x1x1xf32, 4x96x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(pool2d_1, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x4x1x1xf32) <- (-1x4x1x1xf32, 1x4x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_5, parameter_30)

        # pd_op.swish: (-1x4x1x1xf32) <- (-1x4x1x1xf32)
        swish_5 = paddle._C_ops.swish(add__2)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x4x1x1xf32, 96x4x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(swish_5, parameter_31, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_6, parameter_32)

        # pd_op.sigmoid_: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(add__3)

        # pd_op.multiply_: (-1x96x48x48xf32) <- (-1x96x48x48xf32, -1x96x1x1xf32)
        multiply__1 = paddle._C_ops.multiply_(swish_4, sigmoid__1)

        # pd_op.conv2d: (-1x24x48x48xf32) <- (-1x96x48x48xf32, 24x96x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(multiply__1, parameter_33, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x48x48xf32, 24xf32, 24xf32, 24xf32, 24xf32, None) <- (-1x24x48x48xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_34, parameter_35, parameter_36, parameter_37, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x144x48x48xf32) <- (-1x24x48x48xf32, 144x24x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(batch_norm__30, parameter_38, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x48x48xf32, 144xf32, 144xf32, 144xf32, 144xf32, None) <- (-1x144x48x48xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_39, parameter_40, parameter_41, parameter_42, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x144x48x48xf32) <- (-1x144x48x48xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__36)

        # pd_op.depthwise_conv2d: (-1x144x48x48xf32) <- (-1x144x48x48xf32, 144x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(swish_6, parameter_43, [1, 1], [1, 1], 'EXPLICIT', 144, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x144x48x48xf32, 144xf32, 144xf32, 144xf32, 144xf32, None) <- (-1x144x48x48xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_44, parameter_45, parameter_46, parameter_47, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x144x48x48xf32) <- (-1x144x48x48xf32)
        swish_7 = paddle._C_ops.swish(batch_norm__42)

        # pd_op.pool2d: (-1x144x1x1xf32) <- (-1x144x48x48xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(swish_7, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf32) <- (-1x144x1x1xf32, 6x144x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(pool2d_2, parameter_48, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x6x1x1xf32) <- (-1x6x1x1xf32, 1x6x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_9, parameter_49)

        # pd_op.swish: (-1x6x1x1xf32) <- (-1x6x1x1xf32)
        swish_8 = paddle._C_ops.swish(add__4)

        # pd_op.conv2d: (-1x144x1x1xf32) <- (-1x6x1x1xf32, 144x6x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(swish_8, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x144x1x1xf32) <- (-1x144x1x1xf32, 1x144x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_10, parameter_51)

        # pd_op.sigmoid_: (-1x144x1x1xf32) <- (-1x144x1x1xf32)
        sigmoid__2 = paddle._C_ops.sigmoid_(add__5)

        # pd_op.multiply_: (-1x144x48x48xf32) <- (-1x144x48x48xf32, -1x144x1x1xf32)
        multiply__2 = paddle._C_ops.multiply_(swish_7, sigmoid__2)

        # pd_op.conv2d: (-1x24x48x48xf32) <- (-1x144x48x48xf32, 24x144x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(multiply__2, parameter_52, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x48x48xf32, 24xf32, 24xf32, 24xf32, 24xf32, None) <- (-1x24x48x48xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_53, parameter_54, parameter_55, parameter_56, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x24x48x48xf32) <- (-1x24x48x48xf32, -1x24x48x48xf32)
        add__6 = paddle._C_ops.add_(batch_norm__48, batch_norm__30)

        # pd_op.conv2d: (-1x144x48x48xf32) <- (-1x24x48x48xf32, 144x24x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(add__6, parameter_57, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x48x48xf32, 144xf32, 144xf32, 144xf32, 144xf32, None) <- (-1x144x48x48xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_58, parameter_59, parameter_60, parameter_61, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x144x48x48xf32) <- (-1x144x48x48xf32)
        swish_9 = paddle._C_ops.swish(batch_norm__54)

        # pd_op.depthwise_conv2d: (-1x144x24x24xf32) <- (-1x144x48x48xf32, 144x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(swish_9, parameter_62, [2, 2], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x144x24x24xf32, 144xf32, 144xf32, 144xf32, 144xf32, None) <- (-1x144x24x24xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_63, parameter_64, parameter_65, parameter_66, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x144x24x24xf32) <- (-1x144x24x24xf32)
        swish_10 = paddle._C_ops.swish(batch_norm__60)

        # pd_op.pool2d: (-1x144x1x1xf32) <- (-1x144x24x24xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(swish_10, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf32) <- (-1x144x1x1xf32, 6x144x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(pool2d_3, parameter_67, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x6x1x1xf32) <- (-1x6x1x1xf32, 1x6x1x1xf32)
        add__7 = paddle._C_ops.add_(conv2d_13, parameter_68)

        # pd_op.swish: (-1x6x1x1xf32) <- (-1x6x1x1xf32)
        swish_11 = paddle._C_ops.swish(add__7)

        # pd_op.conv2d: (-1x144x1x1xf32) <- (-1x6x1x1xf32, 144x6x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(swish_11, parameter_69, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x144x1x1xf32) <- (-1x144x1x1xf32, 1x144x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_14, parameter_70)

        # pd_op.sigmoid_: (-1x144x1x1xf32) <- (-1x144x1x1xf32)
        sigmoid__3 = paddle._C_ops.sigmoid_(add__8)

        # pd_op.multiply_: (-1x144x24x24xf32) <- (-1x144x24x24xf32, -1x144x1x1xf32)
        multiply__3 = paddle._C_ops.multiply_(swish_10, sigmoid__3)

        # pd_op.conv2d: (-1x40x24x24xf32) <- (-1x144x24x24xf32, 40x144x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(multiply__3, parameter_71, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x24x24xf32, 40xf32, 40xf32, 40xf32, 40xf32, None) <- (-1x40x24x24xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_72, parameter_73, parameter_74, parameter_75, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x240x24x24xf32) <- (-1x40x24x24xf32, 240x40x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(batch_norm__66, parameter_76, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x240x24x24xf32, 240xf32, 240xf32, 240xf32, 240xf32, None) <- (-1x240x24x24xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_77, parameter_78, parameter_79, parameter_80, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x24x24xf32) <- (-1x240x24x24xf32)
        swish_12 = paddle._C_ops.swish(batch_norm__72)

        # pd_op.depthwise_conv2d: (-1x240x24x24xf32) <- (-1x240x24x24xf32, 240x1x5x5xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(swish_12, parameter_81, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x240x24x24xf32, 240xf32, 240xf32, 240xf32, 240xf32, None) <- (-1x240x24x24xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_82, parameter_83, parameter_84, parameter_85, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x24x24xf32) <- (-1x240x24x24xf32)
        swish_13 = paddle._C_ops.swish(batch_norm__78)

        # pd_op.pool2d: (-1x240x1x1xf32) <- (-1x240x24x24xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(swish_13, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x10x1x1xf32) <- (-1x240x1x1xf32, 10x240x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(pool2d_4, parameter_86, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x10x1x1xf32) <- (-1x10x1x1xf32, 1x10x1x1xf32)
        add__9 = paddle._C_ops.add_(conv2d_17, parameter_87)

        # pd_op.swish: (-1x10x1x1xf32) <- (-1x10x1x1xf32)
        swish_14 = paddle._C_ops.swish(add__9)

        # pd_op.conv2d: (-1x240x1x1xf32) <- (-1x10x1x1xf32, 240x10x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(swish_14, parameter_88, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x240x1x1xf32) <- (-1x240x1x1xf32, 1x240x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_18, parameter_89)

        # pd_op.sigmoid_: (-1x240x1x1xf32) <- (-1x240x1x1xf32)
        sigmoid__4 = paddle._C_ops.sigmoid_(add__10)

        # pd_op.multiply_: (-1x240x24x24xf32) <- (-1x240x24x24xf32, -1x240x1x1xf32)
        multiply__4 = paddle._C_ops.multiply_(swish_13, sigmoid__4)

        # pd_op.conv2d: (-1x40x24x24xf32) <- (-1x240x24x24xf32, 40x240x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(multiply__4, parameter_90, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x24x24xf32, 40xf32, 40xf32, 40xf32, 40xf32, None) <- (-1x40x24x24xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x40x24x24xf32) <- (-1x40x24x24xf32, -1x40x24x24xf32)
        add__11 = paddle._C_ops.add_(batch_norm__84, batch_norm__66)

        # pd_op.conv2d: (-1x240x24x24xf32) <- (-1x40x24x24xf32, 240x40x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(add__11, parameter_95, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x240x24x24xf32, 240xf32, 240xf32, 240xf32, 240xf32, None) <- (-1x240x24x24xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x24x24xf32) <- (-1x240x24x24xf32)
        swish_15 = paddle._C_ops.swish(batch_norm__90)

        # pd_op.depthwise_conv2d: (-1x240x12x12xf32) <- (-1x240x24x24xf32, 240x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(swish_15, parameter_100, [2, 2], [1, 1], 'EXPLICIT', 240, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x240x12x12xf32, 240xf32, 240xf32, 240xf32, 240xf32, None) <- (-1x240x12x12xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x12x12xf32) <- (-1x240x12x12xf32)
        swish_16 = paddle._C_ops.swish(batch_norm__96)

        # pd_op.pool2d: (-1x240x1x1xf32) <- (-1x240x12x12xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(swish_16, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x10x1x1xf32) <- (-1x240x1x1xf32, 10x240x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(pool2d_5, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x10x1x1xf32) <- (-1x10x1x1xf32, 1x10x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_21, parameter_106)

        # pd_op.swish: (-1x10x1x1xf32) <- (-1x10x1x1xf32)
        swish_17 = paddle._C_ops.swish(add__12)

        # pd_op.conv2d: (-1x240x1x1xf32) <- (-1x10x1x1xf32, 240x10x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(swish_17, parameter_107, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x240x1x1xf32) <- (-1x240x1x1xf32, 1x240x1x1xf32)
        add__13 = paddle._C_ops.add_(conv2d_22, parameter_108)

        # pd_op.sigmoid_: (-1x240x1x1xf32) <- (-1x240x1x1xf32)
        sigmoid__5 = paddle._C_ops.sigmoid_(add__13)

        # pd_op.multiply_: (-1x240x12x12xf32) <- (-1x240x12x12xf32, -1x240x1x1xf32)
        multiply__5 = paddle._C_ops.multiply_(swish_16, sigmoid__5)

        # pd_op.conv2d: (-1x80x12x12xf32) <- (-1x240x12x12xf32, 80x240x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(multiply__5, parameter_109, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x12x12xf32, 80xf32, 80xf32, 80xf32, 80xf32, None) <- (-1x80x12x12xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x480x12x12xf32) <- (-1x80x12x12xf32, 480x80x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(batch_norm__102, parameter_114, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32, None) <- (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_115, parameter_116, parameter_117, parameter_118, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x12x12xf32) <- (-1x480x12x12xf32)
        swish_18 = paddle._C_ops.swish(batch_norm__108)

        # pd_op.depthwise_conv2d: (-1x480x12x12xf32) <- (-1x480x12x12xf32, 480x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(swish_18, parameter_119, [1, 1], [1, 1], 'EXPLICIT', 480, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32, None) <- (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_120, parameter_121, parameter_122, parameter_123, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x12x12xf32) <- (-1x480x12x12xf32)
        swish_19 = paddle._C_ops.swish(batch_norm__114)

        # pd_op.pool2d: (-1x480x1x1xf32) <- (-1x480x12x12xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(swish_19, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x480x1x1xf32, 20x480x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(pool2d_6, parameter_124, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__14 = paddle._C_ops.add_(conv2d_25, parameter_125)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_20 = paddle._C_ops.swish(add__14)

        # pd_op.conv2d: (-1x480x1x1xf32) <- (-1x20x1x1xf32, 480x20x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(swish_20, parameter_126, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x480x1x1xf32) <- (-1x480x1x1xf32, 1x480x1x1xf32)
        add__15 = paddle._C_ops.add_(conv2d_26, parameter_127)

        # pd_op.sigmoid_: (-1x480x1x1xf32) <- (-1x480x1x1xf32)
        sigmoid__6 = paddle._C_ops.sigmoid_(add__15)

        # pd_op.multiply_: (-1x480x12x12xf32) <- (-1x480x12x12xf32, -1x480x1x1xf32)
        multiply__6 = paddle._C_ops.multiply_(swish_19, sigmoid__6)

        # pd_op.conv2d: (-1x80x12x12xf32) <- (-1x480x12x12xf32, 80x480x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(multiply__6, parameter_128, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x12x12xf32, 80xf32, 80xf32, 80xf32, 80xf32, None) <- (-1x80x12x12xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_129, parameter_130, parameter_131, parameter_132, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x12x12xf32) <- (-1x80x12x12xf32, -1x80x12x12xf32)
        add__16 = paddle._C_ops.add_(batch_norm__120, batch_norm__102)

        # pd_op.conv2d: (-1x480x12x12xf32) <- (-1x80x12x12xf32, 480x80x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(add__16, parameter_133, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32, None) <- (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_134, parameter_135, parameter_136, parameter_137, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x12x12xf32) <- (-1x480x12x12xf32)
        swish_21 = paddle._C_ops.swish(batch_norm__126)

        # pd_op.depthwise_conv2d: (-1x480x12x12xf32) <- (-1x480x12x12xf32, 480x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(swish_21, parameter_138, [1, 1], [1, 1], 'EXPLICIT', 480, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32, None) <- (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_7, parameter_139, parameter_140, parameter_141, parameter_142, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x12x12xf32) <- (-1x480x12x12xf32)
        swish_22 = paddle._C_ops.swish(batch_norm__132)

        # pd_op.pool2d: (-1x480x1x1xf32) <- (-1x480x12x12xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(swish_22, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x480x1x1xf32, 20x480x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(pool2d_7, parameter_143, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__17 = paddle._C_ops.add_(conv2d_29, parameter_144)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_23 = paddle._C_ops.swish(add__17)

        # pd_op.conv2d: (-1x480x1x1xf32) <- (-1x20x1x1xf32, 480x20x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(swish_23, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x480x1x1xf32) <- (-1x480x1x1xf32, 1x480x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_30, parameter_146)

        # pd_op.sigmoid_: (-1x480x1x1xf32) <- (-1x480x1x1xf32)
        sigmoid__7 = paddle._C_ops.sigmoid_(add__18)

        # pd_op.multiply_: (-1x480x12x12xf32) <- (-1x480x12x12xf32, -1x480x1x1xf32)
        multiply__7 = paddle._C_ops.multiply_(swish_22, sigmoid__7)

        # pd_op.conv2d: (-1x80x12x12xf32) <- (-1x480x12x12xf32, 80x480x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(multiply__7, parameter_147, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x12x12xf32, 80xf32, 80xf32, 80xf32, 80xf32, None) <- (-1x80x12x12xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_148, parameter_149, parameter_150, parameter_151, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x12x12xf32) <- (-1x80x12x12xf32, -1x80x12x12xf32)
        add__19 = paddle._C_ops.add_(batch_norm__138, add__16)

        # pd_op.conv2d: (-1x480x12x12xf32) <- (-1x80x12x12xf32, 480x80x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(add__19, parameter_152, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32, None) <- (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_153, parameter_154, parameter_155, parameter_156, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x12x12xf32) <- (-1x480x12x12xf32)
        swish_24 = paddle._C_ops.swish(batch_norm__144)

        # pd_op.depthwise_conv2d: (-1x480x12x12xf32) <- (-1x480x12x12xf32, 480x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(swish_24, parameter_157, [1, 1], [1, 1], 'EXPLICIT', 480, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32, None) <- (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_8, parameter_158, parameter_159, parameter_160, parameter_161, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x12x12xf32) <- (-1x480x12x12xf32)
        swish_25 = paddle._C_ops.swish(batch_norm__150)

        # pd_op.pool2d: (-1x480x1x1xf32) <- (-1x480x12x12xf32, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(swish_25, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x480x1x1xf32, 20x480x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(pool2d_8, parameter_162, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__20 = paddle._C_ops.add_(conv2d_33, parameter_163)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_26 = paddle._C_ops.swish(add__20)

        # pd_op.conv2d: (-1x480x1x1xf32) <- (-1x20x1x1xf32, 480x20x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(swish_26, parameter_164, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x480x1x1xf32) <- (-1x480x1x1xf32, 1x480x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_34, parameter_165)

        # pd_op.sigmoid_: (-1x480x1x1xf32) <- (-1x480x1x1xf32)
        sigmoid__8 = paddle._C_ops.sigmoid_(add__21)

        # pd_op.multiply_: (-1x480x12x12xf32) <- (-1x480x12x12xf32, -1x480x1x1xf32)
        multiply__8 = paddle._C_ops.multiply_(swish_25, sigmoid__8)

        # pd_op.conv2d: (-1x80x12x12xf32) <- (-1x480x12x12xf32, 80x480x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(multiply__8, parameter_166, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x12x12xf32, 80xf32, 80xf32, 80xf32, 80xf32, None) <- (-1x80x12x12xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_167, parameter_168, parameter_169, parameter_170, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x12x12xf32) <- (-1x80x12x12xf32, -1x80x12x12xf32)
        add__22 = paddle._C_ops.add_(batch_norm__156, add__19)

        # pd_op.conv2d: (-1x480x12x12xf32) <- (-1x80x12x12xf32, 480x80x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(add__22, parameter_171, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32, None) <- (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_172, parameter_173, parameter_174, parameter_175, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x12x12xf32) <- (-1x480x12x12xf32)
        swish_27 = paddle._C_ops.swish(batch_norm__162)

        # pd_op.depthwise_conv2d: (-1x480x12x12xf32) <- (-1x480x12x12xf32, 480x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(swish_27, parameter_176, [1, 1], [2, 2], 'EXPLICIT', 480, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32, None) <- (-1x480x12x12xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_9, parameter_177, parameter_178, parameter_179, parameter_180, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x12x12xf32) <- (-1x480x12x12xf32)
        swish_28 = paddle._C_ops.swish(batch_norm__168)

        # pd_op.pool2d: (-1x480x1x1xf32) <- (-1x480x12x12xf32, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(swish_28, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x480x1x1xf32, 20x480x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(pool2d_9, parameter_181, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__23 = paddle._C_ops.add_(conv2d_37, parameter_182)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_29 = paddle._C_ops.swish(add__23)

        # pd_op.conv2d: (-1x480x1x1xf32) <- (-1x20x1x1xf32, 480x20x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(swish_29, parameter_183, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x480x1x1xf32) <- (-1x480x1x1xf32, 1x480x1x1xf32)
        add__24 = paddle._C_ops.add_(conv2d_38, parameter_184)

        # pd_op.sigmoid_: (-1x480x1x1xf32) <- (-1x480x1x1xf32)
        sigmoid__9 = paddle._C_ops.sigmoid_(add__24)

        # pd_op.multiply_: (-1x480x12x12xf32) <- (-1x480x12x12xf32, -1x480x1x1xf32)
        multiply__9 = paddle._C_ops.multiply_(swish_28, sigmoid__9)

        # pd_op.conv2d: (-1x112x12x12xf32) <- (-1x480x12x12xf32, 112x480x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(multiply__9, parameter_185, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x112x12x12xf32, 112xf32, 112xf32, 112xf32, 112xf32, None) <- (-1x112x12x12xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x672x12x12xf32) <- (-1x112x12x12xf32, 672x112x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(batch_norm__174, parameter_190, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_191, parameter_192, parameter_193, parameter_194, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x672x12x12xf32) <- (-1x672x12x12xf32)
        swish_30 = paddle._C_ops.swish(batch_norm__180)

        # pd_op.depthwise_conv2d: (-1x672x12x12xf32) <- (-1x672x12x12xf32, 672x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(swish_30, parameter_195, [1, 1], [2, 2], 'EXPLICIT', 672, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_10, parameter_196, parameter_197, parameter_198, parameter_199, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x672x12x12xf32) <- (-1x672x12x12xf32)
        swish_31 = paddle._C_ops.swish(batch_norm__186)

        # pd_op.pool2d: (-1x672x1x1xf32) <- (-1x672x12x12xf32, 2xi64)
        pool2d_10 = paddle._C_ops.pool2d(swish_31, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x28x1x1xf32) <- (-1x672x1x1xf32, 28x672x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(pool2d_10, parameter_200, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x28x1x1xf32) <- (-1x28x1x1xf32, 1x28x1x1xf32)
        add__25 = paddle._C_ops.add_(conv2d_41, parameter_201)

        # pd_op.swish: (-1x28x1x1xf32) <- (-1x28x1x1xf32)
        swish_32 = paddle._C_ops.swish(add__25)

        # pd_op.conv2d: (-1x672x1x1xf32) <- (-1x28x1x1xf32, 672x28x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(swish_32, parameter_202, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x672x1x1xf32) <- (-1x672x1x1xf32, 1x672x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_42, parameter_203)

        # pd_op.sigmoid_: (-1x672x1x1xf32) <- (-1x672x1x1xf32)
        sigmoid__10 = paddle._C_ops.sigmoid_(add__26)

        # pd_op.multiply_: (-1x672x12x12xf32) <- (-1x672x12x12xf32, -1x672x1x1xf32)
        multiply__10 = paddle._C_ops.multiply_(swish_31, sigmoid__10)

        # pd_op.conv2d: (-1x112x12x12xf32) <- (-1x672x12x12xf32, 112x672x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(multiply__10, parameter_204, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x112x12x12xf32, 112xf32, 112xf32, 112xf32, 112xf32, None) <- (-1x112x12x12xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_205, parameter_206, parameter_207, parameter_208, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x112x12x12xf32) <- (-1x112x12x12xf32, -1x112x12x12xf32)
        add__27 = paddle._C_ops.add_(batch_norm__192, batch_norm__174)

        # pd_op.conv2d: (-1x672x12x12xf32) <- (-1x112x12x12xf32, 672x112x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(add__27, parameter_209, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x672x12x12xf32) <- (-1x672x12x12xf32)
        swish_33 = paddle._C_ops.swish(batch_norm__198)

        # pd_op.depthwise_conv2d: (-1x672x12x12xf32) <- (-1x672x12x12xf32, 672x1x5x5xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(swish_33, parameter_214, [1, 1], [2, 2], 'EXPLICIT', 672, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_11, parameter_215, parameter_216, parameter_217, parameter_218, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x672x12x12xf32) <- (-1x672x12x12xf32)
        swish_34 = paddle._C_ops.swish(batch_norm__204)

        # pd_op.pool2d: (-1x672x1x1xf32) <- (-1x672x12x12xf32, 2xi64)
        pool2d_11 = paddle._C_ops.pool2d(swish_34, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x28x1x1xf32) <- (-1x672x1x1xf32, 28x672x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(pool2d_11, parameter_219, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x28x1x1xf32) <- (-1x28x1x1xf32, 1x28x1x1xf32)
        add__28 = paddle._C_ops.add_(conv2d_45, parameter_220)

        # pd_op.swish: (-1x28x1x1xf32) <- (-1x28x1x1xf32)
        swish_35 = paddle._C_ops.swish(add__28)

        # pd_op.conv2d: (-1x672x1x1xf32) <- (-1x28x1x1xf32, 672x28x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(swish_35, parameter_221, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x672x1x1xf32) <- (-1x672x1x1xf32, 1x672x1x1xf32)
        add__29 = paddle._C_ops.add_(conv2d_46, parameter_222)

        # pd_op.sigmoid_: (-1x672x1x1xf32) <- (-1x672x1x1xf32)
        sigmoid__11 = paddle._C_ops.sigmoid_(add__29)

        # pd_op.multiply_: (-1x672x12x12xf32) <- (-1x672x12x12xf32, -1x672x1x1xf32)
        multiply__11 = paddle._C_ops.multiply_(swish_34, sigmoid__11)

        # pd_op.conv2d: (-1x112x12x12xf32) <- (-1x672x12x12xf32, 112x672x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(multiply__11, parameter_223, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x112x12x12xf32, 112xf32, 112xf32, 112xf32, 112xf32, None) <- (-1x112x12x12xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_224, parameter_225, parameter_226, parameter_227, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x112x12x12xf32) <- (-1x112x12x12xf32, -1x112x12x12xf32)
        add__30 = paddle._C_ops.add_(batch_norm__210, add__27)

        # pd_op.conv2d: (-1x672x12x12xf32) <- (-1x112x12x12xf32, 672x112x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(add__30, parameter_228, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_229, parameter_230, parameter_231, parameter_232, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x672x12x12xf32) <- (-1x672x12x12xf32)
        swish_36 = paddle._C_ops.swish(batch_norm__216)

        # pd_op.depthwise_conv2d: (-1x672x12x12xf32) <- (-1x672x12x12xf32, 672x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(swish_36, parameter_233, [1, 1], [2, 2], 'EXPLICIT', 672, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_12, parameter_234, parameter_235, parameter_236, parameter_237, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x672x12x12xf32) <- (-1x672x12x12xf32)
        swish_37 = paddle._C_ops.swish(batch_norm__222)

        # pd_op.pool2d: (-1x672x1x1xf32) <- (-1x672x12x12xf32, 2xi64)
        pool2d_12 = paddle._C_ops.pool2d(swish_37, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x28x1x1xf32) <- (-1x672x1x1xf32, 28x672x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(pool2d_12, parameter_238, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x28x1x1xf32) <- (-1x28x1x1xf32, 1x28x1x1xf32)
        add__31 = paddle._C_ops.add_(conv2d_49, parameter_239)

        # pd_op.swish: (-1x28x1x1xf32) <- (-1x28x1x1xf32)
        swish_38 = paddle._C_ops.swish(add__31)

        # pd_op.conv2d: (-1x672x1x1xf32) <- (-1x28x1x1xf32, 672x28x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(swish_38, parameter_240, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x672x1x1xf32) <- (-1x672x1x1xf32, 1x672x1x1xf32)
        add__32 = paddle._C_ops.add_(conv2d_50, parameter_241)

        # pd_op.sigmoid_: (-1x672x1x1xf32) <- (-1x672x1x1xf32)
        sigmoid__12 = paddle._C_ops.sigmoid_(add__32)

        # pd_op.multiply_: (-1x672x12x12xf32) <- (-1x672x12x12xf32, -1x672x1x1xf32)
        multiply__12 = paddle._C_ops.multiply_(swish_37, sigmoid__12)

        # pd_op.conv2d: (-1x112x12x12xf32) <- (-1x672x12x12xf32, 112x672x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(multiply__12, parameter_242, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x112x12x12xf32, 112xf32, 112xf32, 112xf32, 112xf32, None) <- (-1x112x12x12xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_243, parameter_244, parameter_245, parameter_246, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x112x12x12xf32) <- (-1x112x12x12xf32, -1x112x12x12xf32)
        add__33 = paddle._C_ops.add_(batch_norm__228, add__30)

        # pd_op.conv2d: (-1x672x12x12xf32) <- (-1x112x12x12xf32, 672x112x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(add__33, parameter_247, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x12x12xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_248, parameter_249, parameter_250, parameter_251, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x672x12x12xf32) <- (-1x672x12x12xf32)
        swish_39 = paddle._C_ops.swish(batch_norm__234)

        # pd_op.depthwise_conv2d: (-1x672x6x6xf32) <- (-1x672x12x12xf32, 672x1x5x5xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(swish_39, parameter_252, [2, 2], [2, 2], 'EXPLICIT', 672, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x672x6x6xf32, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x6x6xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_13, parameter_253, parameter_254, parameter_255, parameter_256, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x672x6x6xf32) <- (-1x672x6x6xf32)
        swish_40 = paddle._C_ops.swish(batch_norm__240)

        # pd_op.pool2d: (-1x672x1x1xf32) <- (-1x672x6x6xf32, 2xi64)
        pool2d_13 = paddle._C_ops.pool2d(swish_40, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x28x1x1xf32) <- (-1x672x1x1xf32, 28x672x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(pool2d_13, parameter_257, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x28x1x1xf32) <- (-1x28x1x1xf32, 1x28x1x1xf32)
        add__34 = paddle._C_ops.add_(conv2d_53, parameter_258)

        # pd_op.swish: (-1x28x1x1xf32) <- (-1x28x1x1xf32)
        swish_41 = paddle._C_ops.swish(add__34)

        # pd_op.conv2d: (-1x672x1x1xf32) <- (-1x28x1x1xf32, 672x28x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(swish_41, parameter_259, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x672x1x1xf32) <- (-1x672x1x1xf32, 1x672x1x1xf32)
        add__35 = paddle._C_ops.add_(conv2d_54, parameter_260)

        # pd_op.sigmoid_: (-1x672x1x1xf32) <- (-1x672x1x1xf32)
        sigmoid__13 = paddle._C_ops.sigmoid_(add__35)

        # pd_op.multiply_: (-1x672x6x6xf32) <- (-1x672x6x6xf32, -1x672x1x1xf32)
        multiply__13 = paddle._C_ops.multiply_(swish_40, sigmoid__13)

        # pd_op.conv2d: (-1x192x6x6xf32) <- (-1x672x6x6xf32, 192x672x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(multiply__13, parameter_261, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32, None) <- (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_262, parameter_263, parameter_264, parameter_265, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x6x6xf32) <- (-1x192x6x6xf32, 1152x192x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(batch_norm__246, parameter_266, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_267, parameter_268, parameter_269, parameter_270, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_42 = paddle._C_ops.swish(batch_norm__252)

        # pd_op.depthwise_conv2d: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, 1152x1x5x5xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(swish_42, parameter_271, [1, 1], [2, 2], 'EXPLICIT', 1152, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_14, parameter_272, parameter_273, parameter_274, parameter_275, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_43 = paddle._C_ops.swish(batch_norm__258)

        # pd_op.pool2d: (-1x1152x1x1xf32) <- (-1x1152x6x6xf32, 2xi64)
        pool2d_14 = paddle._C_ops.pool2d(swish_43, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x1152x1x1xf32, 48x1152x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(pool2d_14, parameter_276, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__36 = paddle._C_ops.add_(conv2d_57, parameter_277)

        # pd_op.swish: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        swish_44 = paddle._C_ops.swish(add__36)

        # pd_op.conv2d: (-1x1152x1x1xf32) <- (-1x48x1x1xf32, 1152x48x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(swish_44, parameter_278, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32, 1x1152x1x1xf32)
        add__37 = paddle._C_ops.add_(conv2d_58, parameter_279)

        # pd_op.sigmoid_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32)
        sigmoid__14 = paddle._C_ops.sigmoid_(add__37)

        # pd_op.multiply_: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, -1x1152x1x1xf32)
        multiply__14 = paddle._C_ops.multiply_(swish_43, sigmoid__14)

        # pd_op.conv2d: (-1x192x6x6xf32) <- (-1x1152x6x6xf32, 192x1152x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(multiply__14, parameter_280, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32, None) <- (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_281, parameter_282, parameter_283, parameter_284, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x192x6x6xf32) <- (-1x192x6x6xf32, -1x192x6x6xf32)
        add__38 = paddle._C_ops.add_(batch_norm__264, batch_norm__246)

        # pd_op.conv2d: (-1x1152x6x6xf32) <- (-1x192x6x6xf32, 1152x192x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(add__38, parameter_285, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_286, parameter_287, parameter_288, parameter_289, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_45 = paddle._C_ops.swish(batch_norm__270)

        # pd_op.depthwise_conv2d: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, 1152x1x5x5xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(swish_45, parameter_290, [1, 1], [2, 2], 'EXPLICIT', 1152, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_15, parameter_291, parameter_292, parameter_293, parameter_294, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_46 = paddle._C_ops.swish(batch_norm__276)

        # pd_op.pool2d: (-1x1152x1x1xf32) <- (-1x1152x6x6xf32, 2xi64)
        pool2d_15 = paddle._C_ops.pool2d(swish_46, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x1152x1x1xf32, 48x1152x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(pool2d_15, parameter_295, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__39 = paddle._C_ops.add_(conv2d_61, parameter_296)

        # pd_op.swish: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        swish_47 = paddle._C_ops.swish(add__39)

        # pd_op.conv2d: (-1x1152x1x1xf32) <- (-1x48x1x1xf32, 1152x48x1x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(swish_47, parameter_297, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32, 1x1152x1x1xf32)
        add__40 = paddle._C_ops.add_(conv2d_62, parameter_298)

        # pd_op.sigmoid_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32)
        sigmoid__15 = paddle._C_ops.sigmoid_(add__40)

        # pd_op.multiply_: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, -1x1152x1x1xf32)
        multiply__15 = paddle._C_ops.multiply_(swish_46, sigmoid__15)

        # pd_op.conv2d: (-1x192x6x6xf32) <- (-1x1152x6x6xf32, 192x1152x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(multiply__15, parameter_299, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32, None) <- (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_300, parameter_301, parameter_302, parameter_303, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x192x6x6xf32) <- (-1x192x6x6xf32, -1x192x6x6xf32)
        add__41 = paddle._C_ops.add_(batch_norm__282, add__38)

        # pd_op.conv2d: (-1x1152x6x6xf32) <- (-1x192x6x6xf32, 1152x192x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(add__41, parameter_304, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_305, parameter_306, parameter_307, parameter_308, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_48 = paddle._C_ops.swish(batch_norm__288)

        # pd_op.depthwise_conv2d: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, 1152x1x5x5xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(swish_48, parameter_309, [1, 1], [2, 2], 'EXPLICIT', 1152, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_16, parameter_310, parameter_311, parameter_312, parameter_313, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_49 = paddle._C_ops.swish(batch_norm__294)

        # pd_op.pool2d: (-1x1152x1x1xf32) <- (-1x1152x6x6xf32, 2xi64)
        pool2d_16 = paddle._C_ops.pool2d(swish_49, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x1152x1x1xf32, 48x1152x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(pool2d_16, parameter_314, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__42 = paddle._C_ops.add_(conv2d_65, parameter_315)

        # pd_op.swish: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        swish_50 = paddle._C_ops.swish(add__42)

        # pd_op.conv2d: (-1x1152x1x1xf32) <- (-1x48x1x1xf32, 1152x48x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(swish_50, parameter_316, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32, 1x1152x1x1xf32)
        add__43 = paddle._C_ops.add_(conv2d_66, parameter_317)

        # pd_op.sigmoid_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32)
        sigmoid__16 = paddle._C_ops.sigmoid_(add__43)

        # pd_op.multiply_: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, -1x1152x1x1xf32)
        multiply__16 = paddle._C_ops.multiply_(swish_49, sigmoid__16)

        # pd_op.conv2d: (-1x192x6x6xf32) <- (-1x1152x6x6xf32, 192x1152x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(multiply__16, parameter_318, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32, None) <- (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_319, parameter_320, parameter_321, parameter_322, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x192x6x6xf32) <- (-1x192x6x6xf32, -1x192x6x6xf32)
        add__44 = paddle._C_ops.add_(batch_norm__300, add__41)

        # pd_op.conv2d: (-1x1152x6x6xf32) <- (-1x192x6x6xf32, 1152x192x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(add__44, parameter_323, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_324, parameter_325, parameter_326, parameter_327, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_51 = paddle._C_ops.swish(batch_norm__306)

        # pd_op.depthwise_conv2d: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, 1152x1x5x5xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(swish_51, parameter_328, [1, 1], [2, 2], 'EXPLICIT', 1152, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_17, parameter_329, parameter_330, parameter_331, parameter_332, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_52 = paddle._C_ops.swish(batch_norm__312)

        # pd_op.pool2d: (-1x1152x1x1xf32) <- (-1x1152x6x6xf32, 2xi64)
        pool2d_17 = paddle._C_ops.pool2d(swish_52, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x1152x1x1xf32, 48x1152x1x1xf32)
        conv2d_69 = paddle._C_ops.conv2d(pool2d_17, parameter_333, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__45 = paddle._C_ops.add_(conv2d_69, parameter_334)

        # pd_op.swish: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        swish_53 = paddle._C_ops.swish(add__45)

        # pd_op.conv2d: (-1x1152x1x1xf32) <- (-1x48x1x1xf32, 1152x48x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(swish_53, parameter_335, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32, 1x1152x1x1xf32)
        add__46 = paddle._C_ops.add_(conv2d_70, parameter_336)

        # pd_op.sigmoid_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32)
        sigmoid__17 = paddle._C_ops.sigmoid_(add__46)

        # pd_op.multiply_: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, -1x1152x1x1xf32)
        multiply__17 = paddle._C_ops.multiply_(swish_52, sigmoid__17)

        # pd_op.conv2d: (-1x192x6x6xf32) <- (-1x1152x6x6xf32, 192x1152x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(multiply__17, parameter_337, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32, None) <- (-1x192x6x6xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_338, parameter_339, parameter_340, parameter_341, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x192x6x6xf32) <- (-1x192x6x6xf32, -1x192x6x6xf32)
        add__47 = paddle._C_ops.add_(batch_norm__318, add__44)

        # pd_op.conv2d: (-1x1152x6x6xf32) <- (-1x192x6x6xf32, 1152x192x1x1xf32)
        conv2d_72 = paddle._C_ops.conv2d(add__47, parameter_342, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_343, parameter_344, parameter_345, parameter_346, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_54 = paddle._C_ops.swish(batch_norm__324)

        # pd_op.depthwise_conv2d: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, 1152x1x3x3xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(swish_54, parameter_347, [1, 1], [1, 1], 'EXPLICIT', 1152, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x6x6xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_18, parameter_348, parameter_349, parameter_350, parameter_351, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32)
        swish_55 = paddle._C_ops.swish(batch_norm__330)

        # pd_op.pool2d: (-1x1152x1x1xf32) <- (-1x1152x6x6xf32, 2xi64)
        pool2d_18 = paddle._C_ops.pool2d(swish_55, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x1152x1x1xf32, 48x1152x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(pool2d_18, parameter_352, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__48 = paddle._C_ops.add_(conv2d_73, parameter_353)

        # pd_op.swish: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        swish_56 = paddle._C_ops.swish(add__48)

        # pd_op.conv2d: (-1x1152x1x1xf32) <- (-1x48x1x1xf32, 1152x48x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(swish_56, parameter_354, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32, 1x1152x1x1xf32)
        add__49 = paddle._C_ops.add_(conv2d_74, parameter_355)

        # pd_op.sigmoid_: (-1x1152x1x1xf32) <- (-1x1152x1x1xf32)
        sigmoid__18 = paddle._C_ops.sigmoid_(add__49)

        # pd_op.multiply_: (-1x1152x6x6xf32) <- (-1x1152x6x6xf32, -1x1152x1x1xf32)
        multiply__18 = paddle._C_ops.multiply_(swish_55, sigmoid__18)

        # pd_op.conv2d: (-1x320x6x6xf32) <- (-1x1152x6x6xf32, 320x1152x1x1xf32)
        conv2d_75 = paddle._C_ops.conv2d(multiply__18, parameter_356, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x6x6xf32, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x6x6xf32, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_357, parameter_358, parameter_359, parameter_360, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1280x6x6xf32) <- (-1x320x6x6xf32, 1280x320x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(batch_norm__336, parameter_361, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1280x6x6xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32, None) <- (-1x1280x6x6xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_362, parameter_363, parameter_364, parameter_365, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1280x6x6xf32) <- (-1x1280x6x6xf32)
        swish_57 = paddle._C_ops.swish(batch_norm__342)

        # pd_op.pool2d: (-1x1280x1x1xf32) <- (-1x1280x6x6xf32, 2xi64)
        pool2d_19 = paddle._C_ops.pool2d(swish_57, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.dropout: (-1x1280x1x1xf32, None) <- (-1x1280x1x1xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(pool2d_19, None, constant_1, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.squeeze_: (-1x1280xf32, None) <- (-1x1280x1x1xf32, 2xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(dropout_0, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf32) <- (-1x1280xf32, 1280x1000xf32)
        matmul_0 = paddle.matmul(squeeze__0, parameter_366, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__50 = paddle._C_ops.add_(matmul_0, parameter_367)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__0 = paddle._C_ops.softmax_(add__50, -1)
        return softmax__0



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

    def forward(self, constant_2, constant_1, parameter_355, parameter_353, parameter_336, parameter_334, parameter_317, parameter_315, parameter_298, parameter_296, parameter_279, parameter_277, parameter_260, parameter_258, parameter_241, parameter_239, parameter_222, parameter_220, parameter_203, parameter_201, parameter_184, parameter_182, parameter_165, parameter_163, parameter_146, parameter_144, parameter_127, parameter_125, parameter_108, parameter_106, parameter_89, parameter_87, parameter_70, parameter_68, parameter_51, parameter_49, parameter_32, parameter_30, parameter_13, parameter_11, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_12, parameter_14, parameter_18, parameter_15, parameter_17, parameter_16, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_31, parameter_33, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_50, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_69, parameter_71, parameter_75, parameter_72, parameter_74, parameter_73, parameter_76, parameter_80, parameter_77, parameter_79, parameter_78, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_88, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_107, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_126, parameter_128, parameter_132, parameter_129, parameter_131, parameter_130, parameter_133, parameter_137, parameter_134, parameter_136, parameter_135, parameter_138, parameter_142, parameter_139, parameter_141, parameter_140, parameter_143, parameter_145, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_164, parameter_166, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_175, parameter_172, parameter_174, parameter_173, parameter_176, parameter_180, parameter_177, parameter_179, parameter_178, parameter_181, parameter_183, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_202, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_221, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_240, parameter_242, parameter_246, parameter_243, parameter_245, parameter_244, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_259, parameter_261, parameter_265, parameter_262, parameter_264, parameter_263, parameter_266, parameter_270, parameter_267, parameter_269, parameter_268, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_278, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_297, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_316, parameter_318, parameter_322, parameter_319, parameter_321, parameter_320, parameter_323, parameter_327, parameter_324, parameter_326, parameter_325, parameter_328, parameter_332, parameter_329, parameter_331, parameter_330, parameter_333, parameter_335, parameter_337, parameter_341, parameter_338, parameter_340, parameter_339, parameter_342, parameter_346, parameter_343, parameter_345, parameter_344, parameter_347, parameter_351, parameter_348, parameter_350, parameter_349, parameter_352, parameter_354, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_365, parameter_362, parameter_364, parameter_363, parameter_366, parameter_367, feed_0):
        return self.builtin_module_1449_0_0(constant_2, constant_1, parameter_355, parameter_353, parameter_336, parameter_334, parameter_317, parameter_315, parameter_298, parameter_296, parameter_279, parameter_277, parameter_260, parameter_258, parameter_241, parameter_239, parameter_222, parameter_220, parameter_203, parameter_201, parameter_184, parameter_182, parameter_165, parameter_163, parameter_146, parameter_144, parameter_127, parameter_125, parameter_108, parameter_106, parameter_89, parameter_87, parameter_70, parameter_68, parameter_51, parameter_49, parameter_32, parameter_30, parameter_13, parameter_11, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_12, parameter_14, parameter_18, parameter_15, parameter_17, parameter_16, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_31, parameter_33, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_50, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_69, parameter_71, parameter_75, parameter_72, parameter_74, parameter_73, parameter_76, parameter_80, parameter_77, parameter_79, parameter_78, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_88, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_107, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_126, parameter_128, parameter_132, parameter_129, parameter_131, parameter_130, parameter_133, parameter_137, parameter_134, parameter_136, parameter_135, parameter_138, parameter_142, parameter_139, parameter_141, parameter_140, parameter_143, parameter_145, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_164, parameter_166, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_175, parameter_172, parameter_174, parameter_173, parameter_176, parameter_180, parameter_177, parameter_179, parameter_178, parameter_181, parameter_183, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_202, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_221, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_240, parameter_242, parameter_246, parameter_243, parameter_245, parameter_244, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_259, parameter_261, parameter_265, parameter_262, parameter_264, parameter_263, parameter_266, parameter_270, parameter_267, parameter_269, parameter_268, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_278, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_297, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_316, parameter_318, parameter_322, parameter_319, parameter_321, parameter_320, parameter_323, parameter_327, parameter_324, parameter_326, parameter_325, parameter_328, parameter_332, parameter_329, parameter_331, parameter_330, parameter_333, parameter_335, parameter_337, parameter_341, parameter_338, parameter_340, parameter_339, parameter_342, parameter_346, parameter_343, parameter_345, parameter_344, parameter_347, parameter_351, parameter_348, parameter_350, parameter_349, parameter_352, parameter_354, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_365, parameter_362, parameter_364, parameter_363, parameter_366, parameter_367, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1449_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_2
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            # constant_1
            paddle.to_tensor([0.2], dtype='float32').reshape([1]),
            # parameter_355
            paddle.uniform([1, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([1, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([1, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([1, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([1, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([1, 28, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([1, 28, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([1, 28, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1, 28, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([1, 10, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([1, 10, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([1, 6, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([1, 6, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([1, 4, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([1, 8, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_0
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # parameter_0
            paddle.uniform([32, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([16, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([96, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([144, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([144, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([24, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([144, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([144, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([40, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([40, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([240, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([80, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([480, 80, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([480, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([80, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([480, 80, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([480, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([80, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([480, 80, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([480, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([80, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([480, 80, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([480, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([112, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([672, 112, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([672, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([112, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([672, 112, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([672, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([112, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([672, 112, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([672, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([112, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([672, 112, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([672, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([192, 672, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([1152, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([1152, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([192, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([1152, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([1152, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([192, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([1152, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([1152, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([192, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1152, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([1152, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([192, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([1152, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([1152, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([320, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([1280, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 192, 192], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_2
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[1, 1152, 1, 1], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[1, 1152, 1, 1], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[1, 1152, 1, 1], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[1, 1152, 1, 1], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[1, 1152, 1, 1], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[1, 28, 1, 1], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[1, 28, 1, 1], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[1, 28, 1, 1], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[1, 28, 1, 1], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[1, 6, 1, 1], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[1, 6, 1, 1], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[1, 4, 1, 1], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[1, 8, 1, 1], dtype='float32'),
            # constant_0
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[8, 32, 1, 1], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[32, 8, 1, 1], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[16, 32, 1, 1], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[96, 16, 1, 1], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[4, 96, 1, 1], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[96, 4, 1, 1], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[144, 24, 1, 1], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[144, 1, 3, 3], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[6, 144, 1, 1], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[144, 6, 1, 1], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[24, 144, 1, 1], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[144, 24, 1, 1], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[144, 1, 5, 5], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[6, 144, 1, 1], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[144, 6, 1, 1], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[40, 144, 1, 1], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[240, 1, 5, 5], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[10, 240, 1, 1], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[240, 10, 1, 1], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[40, 240, 1, 1], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[240, 1, 3, 3], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[10, 240, 1, 1], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[240, 10, 1, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[80, 240, 1, 1], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[480, 80, 1, 1], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[480, 1, 3, 3], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[20, 480, 1, 1], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[480, 20, 1, 1], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[80, 480, 1, 1], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[480, 80, 1, 1], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[480, 1, 3, 3], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[20, 480, 1, 1], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[480, 20, 1, 1], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[80, 480, 1, 1], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[480, 80, 1, 1], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[480, 1, 3, 3], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[20, 480, 1, 1], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[480, 20, 1, 1], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[80, 480, 1, 1], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[480, 80, 1, 1], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[480, 1, 5, 5], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[20, 480, 1, 1], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[480, 20, 1, 1], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[112, 480, 1, 1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[672, 112, 1, 1], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[672, 1, 5, 5], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[28, 672, 1, 1], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[672, 28, 1, 1], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[112, 672, 1, 1], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[672, 112, 1, 1], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[672, 1, 5, 5], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[28, 672, 1, 1], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[672, 28, 1, 1], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[112, 672, 1, 1], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[672, 112, 1, 1], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[672, 1, 5, 5], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[28, 672, 1, 1], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[672, 28, 1, 1], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[112, 672, 1, 1], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[672, 112, 1, 1], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[672, 1, 5, 5], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[28, 672, 1, 1], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[672, 28, 1, 1], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[192, 672, 1, 1], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[1152, 192, 1, 1], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[1152, 1, 5, 5], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[48, 1152, 1, 1], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[1152, 48, 1, 1], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[192, 1152, 1, 1], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[1152, 192, 1, 1], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[1152, 1, 5, 5], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[48, 1152, 1, 1], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[1152, 48, 1, 1], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[192, 1152, 1, 1], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[1152, 192, 1, 1], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[1152, 1, 5, 5], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[48, 1152, 1, 1], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[1152, 48, 1, 1], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[192, 1152, 1, 1], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[1152, 192, 1, 1], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[1152, 1, 5, 5], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[48, 1152, 1, 1], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[1152, 48, 1, 1], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[192, 1152, 1, 1], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[1152, 192, 1, 1], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[1152, 1, 3, 3], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[48, 1152, 1, 1], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[1152, 48, 1, 1], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[320, 1152, 1, 1], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[1280, 320, 1, 1], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 192, 192], dtype='float32'),
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