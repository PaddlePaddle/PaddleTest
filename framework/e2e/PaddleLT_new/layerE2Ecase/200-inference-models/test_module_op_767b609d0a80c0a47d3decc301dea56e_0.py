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
    return [939][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2060_0_0(self, parameter_410, parameter_409, constant_20, constant_19, constant_18, constant_17, parameter_347, constant_16, parameter_346, constant_15, constant_14, constant_13, constant_12, parameter_343, constant_11, parameter_342, constant_10, constant_9, parameter_336, parameter_335, parameter_330, parameter_324, parameter_318, parameter_312, parameter_306, parameter_300, parameter_294, parameter_288, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_251, parameter_252, parameter_253, parameter_254, parameter_256, parameter_255, parameter_257, parameter_258, parameter_259, parameter_260, parameter_262, parameter_261, parameter_263, parameter_264, parameter_265, parameter_266, parameter_268, parameter_267, parameter_269, parameter_270, parameter_271, parameter_272, parameter_274, parameter_273, parameter_275, parameter_276, parameter_277, parameter_278, parameter_280, parameter_279, parameter_281, parameter_282, parameter_283, parameter_284, parameter_286, parameter_285, parameter_287, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_310, parameter_307, parameter_309, parameter_308, parameter_311, parameter_316, parameter_313, parameter_315, parameter_314, parameter_317, parameter_322, parameter_319, parameter_321, parameter_320, parameter_323, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_334, parameter_331, parameter_333, parameter_332, parameter_337, parameter_338, parameter_339, parameter_340, parameter_341, parameter_344, parameter_345, parameter_348, parameter_349, parameter_350, parameter_351, parameter_352, parameter_353, parameter_354, parameter_356, parameter_355, parameter_357, parameter_358, parameter_359, parameter_360, parameter_362, parameter_361, parameter_363, parameter_364, parameter_365, parameter_366, parameter_367, parameter_368, parameter_370, parameter_369, parameter_371, parameter_372, parameter_373, parameter_374, parameter_376, parameter_375, parameter_377, parameter_378, parameter_379, parameter_380, parameter_381, parameter_382, parameter_384, parameter_383, parameter_385, parameter_386, parameter_387, parameter_388, parameter_390, parameter_389, parameter_391, parameter_392, parameter_393, parameter_394, parameter_395, parameter_396, parameter_398, parameter_397, parameter_399, parameter_400, parameter_401, parameter_402, parameter_404, parameter_403, parameter_405, parameter_406, parameter_407, parameter_408, feed_0):

        # pd_op.cast: (-1x3x32x128xf16) <- (-1x3x32x128xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x32x32x128xf16) <- (-1x3x32x128xf16, 32x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x32x128xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x32x128xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x32x128xf16) <- (-1x32x32x128xf16)
        relu__0 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.conv2d: (-1x32x32x128xf16) <- (-1x32x32x128xf16, 32x32x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_5, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x32x128xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x32x128xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x32x128xf16) <- (-1x32x32x128xf16)
        relu__1 = paddle._C_ops.relu(batch_norm__6)

        # pd_op.conv2d: (-1x32x16x64xf16) <- (-1x32x32x128xf16, 32x32x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_10, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x16x64xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x16x64xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x16x64xf16) <- (-1x32x32x128xf16, 32x32x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(relu__0, parameter_15, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x16x64xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x16x64xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x32x16x64xf16) <- (-1x32x16x64xf16, -1x32x16x64xf16)
        add__0 = paddle._C_ops.add(batch_norm__12, batch_norm__18)

        # pd_op.relu_: (-1x32x16x64xf16) <- (-1x32x16x64xf16)
        relu__2 = paddle._C_ops.relu(add__0)

        # pd_op.conv2d: (-1x32x16x64xf16) <- (-1x32x16x64xf16, 32x32x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__2, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x16x64xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x16x64xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x16x64xf16) <- (-1x32x16x64xf16)
        relu__3 = paddle._C_ops.relu(batch_norm__24)

        # pd_op.conv2d: (-1x32x16x64xf16) <- (-1x32x16x64xf16, 32x32x3x3xf16)
        conv2d_5 = paddle._C_ops.conv2d(relu__3, parameter_25, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x16x64xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x16x64xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x32x16x64xf16) <- (-1x32x16x64xf16, -1x32x16x64xf16)
        add__1 = paddle._C_ops.add(batch_norm__30, relu__2)

        # pd_op.relu_: (-1x32x16x64xf16) <- (-1x32x16x64xf16)
        relu__4 = paddle._C_ops.relu(add__1)

        # pd_op.conv2d: (-1x32x16x64xf16) <- (-1x32x16x64xf16, 32x32x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(relu__4, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x16x64xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x16x64xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x16x64xf16) <- (-1x32x16x64xf16)
        relu__5 = paddle._C_ops.relu(batch_norm__36)

        # pd_op.conv2d: (-1x32x16x64xf16) <- (-1x32x16x64xf16, 32x32x3x3xf16)
        conv2d_7 = paddle._C_ops.conv2d(relu__5, parameter_35, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x16x64xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x16x64xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x32x16x64xf16) <- (-1x32x16x64xf16, -1x32x16x64xf16)
        add__2 = paddle._C_ops.add(batch_norm__42, relu__4)

        # pd_op.relu_: (-1x32x16x64xf16) <- (-1x32x16x64xf16)
        relu__6 = paddle._C_ops.relu(add__2)

        # pd_op.conv2d: (-1x64x16x64xf16) <- (-1x32x16x64xf16, 64x32x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu__6, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x16x64xf16) <- (-1x64x16x64xf16)
        relu__7 = paddle._C_ops.relu(batch_norm__48)

        # pd_op.conv2d: (-1x64x16x64xf16) <- (-1x64x16x64xf16, 64x64x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__7, parameter_45, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x16x64xf16) <- (-1x32x16x64xf16, 64x32x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu__6, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x16x64xf16) <- (-1x64x16x64xf16, -1x64x16x64xf16)
        add__3 = paddle._C_ops.add(batch_norm__54, batch_norm__60)

        # pd_op.relu_: (-1x64x16x64xf16) <- (-1x64x16x64xf16)
        relu__8 = paddle._C_ops.relu(add__3)

        # pd_op.conv2d: (-1x64x16x64xf16) <- (-1x64x16x64xf16, 64x64x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(relu__8, parameter_55, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x16x64xf16) <- (-1x64x16x64xf16)
        relu__9 = paddle._C_ops.relu(batch_norm__66)

        # pd_op.conv2d: (-1x64x16x64xf16) <- (-1x64x16x64xf16, 64x64x3x3xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu__9, parameter_60, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x16x64xf16) <- (-1x64x16x64xf16, -1x64x16x64xf16)
        add__4 = paddle._C_ops.add(batch_norm__72, relu__8)

        # pd_op.relu_: (-1x64x16x64xf16) <- (-1x64x16x64xf16)
        relu__10 = paddle._C_ops.relu(add__4)

        # pd_op.conv2d: (-1x64x16x64xf16) <- (-1x64x16x64xf16, 64x64x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu__10, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x16x64xf16) <- (-1x64x16x64xf16)
        relu__11 = paddle._C_ops.relu(batch_norm__78)

        # pd_op.conv2d: (-1x64x16x64xf16) <- (-1x64x16x64xf16, 64x64x3x3xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu__11, parameter_70, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x16x64xf16) <- (-1x64x16x64xf16, -1x64x16x64xf16)
        add__5 = paddle._C_ops.add(batch_norm__84, relu__10)

        # pd_op.relu_: (-1x64x16x64xf16) <- (-1x64x16x64xf16)
        relu__12 = paddle._C_ops.relu(add__5)

        # pd_op.conv2d: (-1x64x16x64xf16) <- (-1x64x16x64xf16, 64x64x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu__12, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x16x64xf16) <- (-1x64x16x64xf16)
        relu__13 = paddle._C_ops.relu(batch_norm__90)

        # pd_op.conv2d: (-1x64x16x64xf16) <- (-1x64x16x64xf16, 64x64x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(relu__13, parameter_80, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x16x64xf16) <- (-1x64x16x64xf16, -1x64x16x64xf16)
        add__6 = paddle._C_ops.add(batch_norm__96, relu__12)

        # pd_op.relu_: (-1x64x16x64xf16) <- (-1x64x16x64xf16)
        relu__14 = paddle._C_ops.relu(add__6)

        # pd_op.conv2d: (-1x128x16x64xf16) <- (-1x64x16x64xf16, 128x64x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu__14, parameter_85, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x64xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x64xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x16x64xf16) <- (-1x128x16x64xf16)
        relu__15 = paddle._C_ops.relu(batch_norm__102)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x16x64xf16, 128x128x3x3xf16)
        conv2d_18 = paddle._C_ops.conv2d(relu__15, parameter_90, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x64x16x64xf16, 128x64x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu__14, parameter_95, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x8x32xf16) <- (-1x128x8x32xf16, -1x128x8x32xf16)
        add__7 = paddle._C_ops.add(batch_norm__108, batch_norm__114)

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__16 = paddle._C_ops.relu(add__7)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(relu__16, parameter_100, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__17 = paddle._C_ops.relu(batch_norm__120)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x3x3xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu__17, parameter_105, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x8x32xf16) <- (-1x128x8x32xf16, -1x128x8x32xf16)
        add__8 = paddle._C_ops.add(batch_norm__126, relu__16)

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__18 = paddle._C_ops.relu(add__8)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(relu__18, parameter_110, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__19 = paddle._C_ops.relu(batch_norm__132)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x3x3xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu__19, parameter_115, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x8x32xf16) <- (-1x128x8x32xf16, -1x128x8x32xf16)
        add__9 = paddle._C_ops.add(batch_norm__138, relu__18)

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__20 = paddle._C_ops.relu(add__9)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(relu__20, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__21 = paddle._C_ops.relu(batch_norm__144)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x3x3xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__21, parameter_125, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x8x32xf16) <- (-1x128x8x32xf16, -1x128x8x32xf16)
        add__10 = paddle._C_ops.add(batch_norm__150, relu__20)

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__22 = paddle._C_ops.relu(add__10)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(relu__22, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__23 = paddle._C_ops.relu(batch_norm__156)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x3x3xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu__23, parameter_135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x8x32xf16) <- (-1x128x8x32xf16, -1x128x8x32xf16)
        add__11 = paddle._C_ops.add(batch_norm__162, relu__22)

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__24 = paddle._C_ops.relu(add__11)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu__24, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__25 = paddle._C_ops.relu(batch_norm__168)

        # pd_op.conv2d: (-1x128x8x32xf16) <- (-1x128x8x32xf16, 128x128x3x3xf16)
        conv2d_29 = paddle._C_ops.conv2d(relu__25, parameter_145, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x8x32xf16) <- (-1x128x8x32xf16, -1x128x8x32xf16)
        add__12 = paddle._C_ops.add(batch_norm__174, relu__24)

        # pd_op.relu_: (-1x128x8x32xf16) <- (-1x128x8x32xf16)
        relu__26 = paddle._C_ops.relu(add__12)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x128x8x32xf16, 256x128x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu__26, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__27 = paddle._C_ops.relu(batch_norm__180)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu__27, parameter_155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x128x8x32xf16, 256x128x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(relu__26, parameter_160, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x32xf16) <- (-1x256x8x32xf16, -1x256x8x32xf16)
        add__13 = paddle._C_ops.add(batch_norm__186, batch_norm__192)

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__28 = paddle._C_ops.relu(add__13)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(relu__28, parameter_165, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__29 = paddle._C_ops.relu(batch_norm__198)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x3x3xf16)
        conv2d_34 = paddle._C_ops.conv2d(relu__29, parameter_170, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x32xf16) <- (-1x256x8x32xf16, -1x256x8x32xf16)
        add__14 = paddle._C_ops.add(batch_norm__204, relu__28)

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__30 = paddle._C_ops.relu(add__14)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu__30, parameter_175, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__31 = paddle._C_ops.relu(batch_norm__210)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x3x3xf16)
        conv2d_36 = paddle._C_ops.conv2d(relu__31, parameter_180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_181, parameter_182, parameter_183, parameter_184, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x32xf16) <- (-1x256x8x32xf16, -1x256x8x32xf16)
        add__15 = paddle._C_ops.add(batch_norm__216, relu__30)

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__32 = paddle._C_ops.relu(add__15)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(relu__32, parameter_185, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__33 = paddle._C_ops.relu(batch_norm__222)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x3x3xf16)
        conv2d_38 = paddle._C_ops.conv2d(relu__33, parameter_190, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_191, parameter_192, parameter_193, parameter_194, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x32xf16) <- (-1x256x8x32xf16, -1x256x8x32xf16)
        add__16 = paddle._C_ops.add(batch_norm__228, relu__32)

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__34 = paddle._C_ops.relu(add__16)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(relu__34, parameter_195, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_196, parameter_197, parameter_198, parameter_199, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__35 = paddle._C_ops.relu(batch_norm__234)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x3x3xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu__35, parameter_200, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_201, parameter_202, parameter_203, parameter_204, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x32xf16) <- (-1x256x8x32xf16, -1x256x8x32xf16)
        add__17 = paddle._C_ops.add(batch_norm__240, relu__34)

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__36 = paddle._C_ops.relu(add__17)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(relu__36, parameter_205, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_206, parameter_207, parameter_208, parameter_209, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__37 = paddle._C_ops.relu(batch_norm__246)

        # pd_op.conv2d: (-1x256x8x32xf16) <- (-1x256x8x32xf16, 256x256x3x3xf16)
        conv2d_42 = paddle._C_ops.conv2d(relu__37, parameter_210, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_211, parameter_212, parameter_213, parameter_214, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x32xf16) <- (-1x256x8x32xf16, -1x256x8x32xf16)
        add__18 = paddle._C_ops.add(batch_norm__252, relu__36)

        # pd_op.relu_: (-1x256x8x32xf16) <- (-1x256x8x32xf16)
        relu__38 = paddle._C_ops.relu(add__18)

        # pd_op.conv2d: (-1x512x8x32xf16) <- (-1x256x8x32xf16, 512x256x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(relu__38, parameter_215, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x32xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x32xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x8x32xf16) <- (-1x512x8x32xf16)
        relu__39 = paddle._C_ops.relu(batch_norm__258)

        # pd_op.conv2d: (-1x512x8x32xf16) <- (-1x512x8x32xf16, 512x512x3x3xf16)
        conv2d_44 = paddle._C_ops.conv2d(relu__39, parameter_220, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x32xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x32xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_221, parameter_222, parameter_223, parameter_224, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x8x32xf16) <- (-1x256x8x32xf16, 512x256x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(relu__38, parameter_225, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x32xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x32xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_226, parameter_227, parameter_228, parameter_229, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x8x32xf16) <- (-1x512x8x32xf16, -1x512x8x32xf16)
        add__19 = paddle._C_ops.add(batch_norm__264, batch_norm__270)

        # pd_op.relu_: (-1x512x8x32xf16) <- (-1x512x8x32xf16)
        relu__40 = paddle._C_ops.relu(add__19)

        # pd_op.conv2d: (-1x512x8x32xf16) <- (-1x512x8x32xf16, 512x512x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(relu__40, parameter_230, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x32xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x32xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x8x32xf16) <- (-1x512x8x32xf16)
        relu__41 = paddle._C_ops.relu(batch_norm__276)

        # pd_op.conv2d: (-1x512x8x32xf16) <- (-1x512x8x32xf16, 512x512x3x3xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu__41, parameter_235, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x32xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x32xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x8x32xf16) <- (-1x512x8x32xf16, -1x512x8x32xf16)
        add__20 = paddle._C_ops.add(batch_norm__282, relu__40)

        # pd_op.relu_: (-1x512x8x32xf16) <- (-1x512x8x32xf16)
        relu__42 = paddle._C_ops.relu(add__20)

        # pd_op.conv2d: (-1x512x8x32xf16) <- (-1x512x8x32xf16, 512x512x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(relu__42, parameter_240, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x32xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x32xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x8x32xf16) <- (-1x512x8x32xf16)
        relu__43 = paddle._C_ops.relu(batch_norm__288)

        # pd_op.conv2d: (-1x512x8x32xf16) <- (-1x512x8x32xf16, 512x512x3x3xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu__43, parameter_245, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x32xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x32xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x8x32xf16) <- (-1x512x8x32xf16, -1x512x8x32xf16)
        add__21 = paddle._C_ops.add(batch_norm__294, relu__42)

        # pd_op.relu_: (-1x512x8x32xf16) <- (-1x512x8x32xf16)
        relu__44 = paddle._C_ops.relu(add__21)

        # pd_op.transpose: (-1x8x32x512xf16) <- (-1x512x8x32xf16)
        transpose_0 = paddle._C_ops.transpose(relu__44, [0, 2, 3, 1])

        # pd_op.flatten_: (-1x256x512xf16, None) <- (-1x8x32x512xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten(transpose_0, 1, 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (256x-1x512xf16) <- (-1x256x512xf16)
        transpose_1 = paddle._C_ops.transpose(flatten__0, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (256x-1x512xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(transpose_1, 'float32'))

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_0 = [slice_0]

        # pd_op.slice: (-1x1x512xf16) <- (256x1x512xf16, 1xi64, [xi32])
        slice_1 = paddle._C_ops.slice(parameter_250, [0], constant_0, combine_0, [-1], [])

        # pd_op.add: (256x-1x512xf16) <- (256x-1x512xf16, -1x1x512xf16)
        add_0 = transpose_1 + slice_1

        # pd_op.dropout: (256x-1x512xf16, None) <- (256x-1x512xf16, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_0, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x256x512xf16) <- (256x-1x512xf16)
        transpose_2 = paddle._C_ops.transpose(dropout_0, [1, 0, 2])

        # pd_op.matmul: (-1x256x1536xf16) <- (-1x256x512xf16, 512x1536xf16)
        matmul_0 = paddle.matmul(transpose_2, parameter_251, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x1536xf16) <- (-1x256x1536xf16, 1536xf16)
        add__22 = paddle._C_ops.add(matmul_0, parameter_252)

        # pd_op.reshape_: (-1x256x3x8x64xf16, 0x-1x256x1536xf16) <- (-1x256x1536xf16, 5xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__22, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x256x64xf16) <- (-1x256x3x8x64xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__0, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x256x64xf16) <- (3x-1x8x256x64xf16, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_3, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x256x64xf16) <- (3x-1x8x256x64xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_3, [0], constant_1, constant_4, [1], [0])

        # pd_op.slice: (-1x8x256x64xf16) <- (3x-1x8x256x64xf16, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(transpose_3, [0], constant_4, constant_5, [1], [0])

        # pd_op.transpose: (-1x8x64x256xf16) <- (-1x8x256x64xf16)
        transpose_4 = paddle._C_ops.transpose(slice_3, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x256x256xf16) <- (-1x8x256x64xf16, -1x8x64x256xf16)
        matmul_1 = paddle.matmul(slice_2, transpose_4, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x256x256xf16) <- (-1x8x256x256xf16, 1xf32)
        scale__0 = paddle._C_ops.scale(matmul_1, constant_6, float('0'), True)

        # pd_op.softmax_: (-1x8x256x256xf16) <- (-1x8x256x256xf16)
        softmax__0 = paddle._C_ops.softmax(scale__0, -1)

        # pd_op.dropout: (-1x8x256x256xf16, None) <- (-1x8x256x256xf16, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__0, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x256x64xf16) <- (-1x8x256x256xf16, -1x8x256x64xf16)
        matmul_2 = paddle.matmul(dropout_2, slice_4, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x256x8x64xf16) <- (-1x8x256x64xf16)
        transpose_5 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x256x512xf16, 0x-1x256x8x64xf16) <- (-1x256x8x64xf16, 3xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_5, constant_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x512xf16) <- (-1x256x512xf16, 512x512xf16)
        matmul_3 = paddle.matmul(reshape__2, parameter_253, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, 512xf16)
        add__23 = paddle._C_ops.add(matmul_3, parameter_254)

        # pd_op.dropout: (-1x256x512xf16, None) <- (-1x256x512xf16, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__23, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, -1x256x512xf16)
        add__24 = paddle._C_ops.add(transpose_2, dropout_4)

        # pd_op.layer_norm: (-1x256x512xf16, -1x256xf32, -1x256xf32) <- (-1x256x512xf16, 512xf32, 512xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__24, parameter_255, parameter_256, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x256x2048xf16) <- (-1x256x512xf16, 512x2048xf16)
        matmul_4 = paddle.matmul(layer_norm_0, parameter_257, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x2048xf16) <- (-1x256x2048xf16, 2048xf16)
        add__25 = paddle._C_ops.add(matmul_4, parameter_258)

        # pd_op.relu_: (-1x256x2048xf16) <- (-1x256x2048xf16)
        relu__45 = paddle._C_ops.relu(add__25)

        # pd_op.dropout: (-1x256x2048xf16, None) <- (-1x256x2048xf16, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__45, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x512xf16) <- (-1x256x2048xf16, 2048x512xf16)
        matmul_5 = paddle.matmul(dropout_6, parameter_259, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, 512xf16)
        add__26 = paddle._C_ops.add(matmul_5, parameter_260)

        # pd_op.dropout: (-1x256x512xf16, None) <- (-1x256x512xf16, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__26, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x256x512xf16, None) <- (-1x256x512xf16, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_8, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, -1x256x512xf16)
        add__27 = paddle._C_ops.add(layer_norm_0, dropout_10)

        # pd_op.layer_norm: (-1x256x512xf16, -1x256xf32, -1x256xf32) <- (-1x256x512xf16, 512xf32, 512xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__27, parameter_261, parameter_262, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x256x1536xf16) <- (-1x256x512xf16, 512x1536xf16)
        matmul_6 = paddle.matmul(layer_norm_3, parameter_263, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x1536xf16) <- (-1x256x1536xf16, 1536xf16)
        add__28 = paddle._C_ops.add(matmul_6, parameter_264)

        # pd_op.reshape_: (-1x256x3x8x64xf16, 0x-1x256x1536xf16) <- (-1x256x1536xf16, 5xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__28, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x256x64xf16) <- (-1x256x3x8x64xf16)
        transpose_6 = paddle._C_ops.transpose(reshape__4, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x256x64xf16) <- (3x-1x8x256x64xf16, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(transpose_6, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x256x64xf16) <- (3x-1x8x256x64xf16, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_6, [0], constant_1, constant_4, [1], [0])

        # pd_op.slice: (-1x8x256x64xf16) <- (3x-1x8x256x64xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_6, [0], constant_4, constant_5, [1], [0])

        # pd_op.transpose: (-1x8x64x256xf16) <- (-1x8x256x64xf16)
        transpose_7 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x256x256xf16) <- (-1x8x256x64xf16, -1x8x64x256xf16)
        matmul_7 = paddle.matmul(slice_5, transpose_7, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x256x256xf16) <- (-1x8x256x256xf16, 1xf32)
        scale__1 = paddle._C_ops.scale(matmul_7, constant_6, float('0'), True)

        # pd_op.softmax_: (-1x8x256x256xf16) <- (-1x8x256x256xf16)
        softmax__1 = paddle._C_ops.softmax(scale__1, -1)

        # pd_op.dropout: (-1x8x256x256xf16, None) <- (-1x8x256x256xf16, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__1, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x256x64xf16) <- (-1x8x256x256xf16, -1x8x256x64xf16)
        matmul_8 = paddle.matmul(dropout_12, slice_7, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x256x8x64xf16) <- (-1x8x256x64xf16)
        transpose_8 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x256x512xf16, 0x-1x256x8x64xf16) <- (-1x256x8x64xf16, 3xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_8, constant_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x512xf16) <- (-1x256x512xf16, 512x512xf16)
        matmul_9 = paddle.matmul(reshape__6, parameter_265, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, 512xf16)
        add__29 = paddle._C_ops.add(matmul_9, parameter_266)

        # pd_op.dropout: (-1x256x512xf16, None) <- (-1x256x512xf16, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__29, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, -1x256x512xf16)
        add__30 = paddle._C_ops.add(layer_norm_3, dropout_14)

        # pd_op.layer_norm: (-1x256x512xf16, -1x256xf32, -1x256xf32) <- (-1x256x512xf16, 512xf32, 512xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__30, parameter_267, parameter_268, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x256x2048xf16) <- (-1x256x512xf16, 512x2048xf16)
        matmul_10 = paddle.matmul(layer_norm_6, parameter_269, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x2048xf16) <- (-1x256x2048xf16, 2048xf16)
        add__31 = paddle._C_ops.add(matmul_10, parameter_270)

        # pd_op.relu_: (-1x256x2048xf16) <- (-1x256x2048xf16)
        relu__46 = paddle._C_ops.relu(add__31)

        # pd_op.dropout: (-1x256x2048xf16, None) <- (-1x256x2048xf16, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__46, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x512xf16) <- (-1x256x2048xf16, 2048x512xf16)
        matmul_11 = paddle.matmul(dropout_16, parameter_271, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, 512xf16)
        add__32 = paddle._C_ops.add(matmul_11, parameter_272)

        # pd_op.dropout: (-1x256x512xf16, None) <- (-1x256x512xf16, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__32, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x256x512xf16, None) <- (-1x256x512xf16, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_18, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, -1x256x512xf16)
        add__33 = paddle._C_ops.add(layer_norm_6, dropout_20)

        # pd_op.layer_norm: (-1x256x512xf16, -1x256xf32, -1x256xf32) <- (-1x256x512xf16, 512xf32, 512xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__33, parameter_273, parameter_274, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x256x1536xf16) <- (-1x256x512xf16, 512x1536xf16)
        matmul_12 = paddle.matmul(layer_norm_9, parameter_275, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x1536xf16) <- (-1x256x1536xf16, 1536xf16)
        add__34 = paddle._C_ops.add(matmul_12, parameter_276)

        # pd_op.reshape_: (-1x256x3x8x64xf16, 0x-1x256x1536xf16) <- (-1x256x1536xf16, 5xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__34, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x256x64xf16) <- (-1x256x3x8x64xf16)
        transpose_9 = paddle._C_ops.transpose(reshape__8, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x256x64xf16) <- (3x-1x8x256x64xf16, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(transpose_9, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x256x64xf16) <- (3x-1x8x256x64xf16, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_9, [0], constant_1, constant_4, [1], [0])

        # pd_op.slice: (-1x8x256x64xf16) <- (3x-1x8x256x64xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_9, [0], constant_4, constant_5, [1], [0])

        # pd_op.transpose: (-1x8x64x256xf16) <- (-1x8x256x64xf16)
        transpose_10 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x256x256xf16) <- (-1x8x256x64xf16, -1x8x64x256xf16)
        matmul_13 = paddle.matmul(slice_8, transpose_10, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x256x256xf16) <- (-1x8x256x256xf16, 1xf32)
        scale__2 = paddle._C_ops.scale(matmul_13, constant_6, float('0'), True)

        # pd_op.softmax_: (-1x8x256x256xf16) <- (-1x8x256x256xf16)
        softmax__2 = paddle._C_ops.softmax(scale__2, -1)

        # pd_op.dropout: (-1x8x256x256xf16, None) <- (-1x8x256x256xf16, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__2, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x256x64xf16) <- (-1x8x256x256xf16, -1x8x256x64xf16)
        matmul_14 = paddle.matmul(dropout_22, slice_10, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x256x8x64xf16) <- (-1x8x256x64xf16)
        transpose_11 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x256x512xf16, 0x-1x256x8x64xf16) <- (-1x256x8x64xf16, 3xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_11, constant_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x512xf16) <- (-1x256x512xf16, 512x512xf16)
        matmul_15 = paddle.matmul(reshape__10, parameter_277, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, 512xf16)
        add__35 = paddle._C_ops.add(matmul_15, parameter_278)

        # pd_op.dropout: (-1x256x512xf16, None) <- (-1x256x512xf16, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__35, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, -1x256x512xf16)
        add__36 = paddle._C_ops.add(layer_norm_9, dropout_24)

        # pd_op.layer_norm: (-1x256x512xf16, -1x256xf32, -1x256xf32) <- (-1x256x512xf16, 512xf32, 512xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__36, parameter_279, parameter_280, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x256x2048xf16) <- (-1x256x512xf16, 512x2048xf16)
        matmul_16 = paddle.matmul(layer_norm_12, parameter_281, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x2048xf16) <- (-1x256x2048xf16, 2048xf16)
        add__37 = paddle._C_ops.add(matmul_16, parameter_282)

        # pd_op.relu_: (-1x256x2048xf16) <- (-1x256x2048xf16)
        relu__47 = paddle._C_ops.relu(add__37)

        # pd_op.dropout: (-1x256x2048xf16, None) <- (-1x256x2048xf16, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__47, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x512xf16) <- (-1x256x2048xf16, 2048x512xf16)
        matmul_17 = paddle.matmul(dropout_26, parameter_283, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, 512xf16)
        add__38 = paddle._C_ops.add(matmul_17, parameter_284)

        # pd_op.dropout: (-1x256x512xf16, None) <- (-1x256x512xf16, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__38, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x256x512xf16, None) <- (-1x256x512xf16, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_28, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x512xf16) <- (-1x256x512xf16, -1x256x512xf16)
        add__39 = paddle._C_ops.add(layer_norm_12, dropout_30)

        # pd_op.layer_norm: (-1x256x512xf16, -1x256xf32, -1x256xf32) <- (-1x256x512xf16, 512xf32, 512xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__39, parameter_285, parameter_286, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.reshape_: (-1x8x32x512xf16, 0x-1x256x512xf16) <- (-1x256x512xf16, 4xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(layer_norm_15, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x512x8x32xf16) <- (-1x8x32x512xf16)
        transpose_12 = paddle._C_ops.transpose(reshape__12, [0, 3, 1, 2])

        # pd_op.shape: (4xi32) <- (-1x512x8x32xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(transpose_12, 'float32'))

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_1, [0], constant_0, constant_1, [1], [0])

        # pd_op.conv2d: (-1x64x8x16xf16) <- (-1x512x8x32xf16, 64x512x3x3xf16)
        conv2d_50 = paddle._C_ops.conv2d(transpose_12, parameter_287, [1, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x8x16xf16) <- (-1x64x8x16xf16, 1x64x1x1xf16)
        add__40 = paddle._C_ops.add(conv2d_50, parameter_288)

        # pd_op.batch_norm_: (-1x64x8x16xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x8x16xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__40, parameter_289, parameter_290, parameter_291, parameter_292, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x8x16xf16) <- (-1x64x8x16xf16)
        relu__48 = paddle._C_ops.relu(batch_norm__300)

        # pd_op.conv2d: (-1x64x4x8xf16) <- (-1x64x8x16xf16, 64x64x3x3xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu__48, parameter_293, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x4x8xf16) <- (-1x64x4x8xf16, 1x64x1x1xf16)
        add__41 = paddle._C_ops.add(conv2d_51, parameter_294)

        # pd_op.batch_norm_: (-1x64x4x8xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x4x8xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__41, parameter_295, parameter_296, parameter_297, parameter_298, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x4x8xf16) <- (-1x64x4x8xf16)
        relu__49 = paddle._C_ops.relu(batch_norm__306)

        # pd_op.conv2d: (-1x64x2x4xf16) <- (-1x64x4x8xf16, 64x64x3x3xf16)
        conv2d_52 = paddle._C_ops.conv2d(relu__49, parameter_299, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x2x4xf16) <- (-1x64x2x4xf16, 1x64x1x1xf16)
        add__42 = paddle._C_ops.add(conv2d_52, parameter_300)

        # pd_op.batch_norm_: (-1x64x2x4xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x2x4xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__42, parameter_301, parameter_302, parameter_303, parameter_304, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x2x4xf16) <- (-1x64x2x4xf16)
        relu__50 = paddle._C_ops.relu(batch_norm__312)

        # pd_op.conv2d: (-1x64x1x2xf16) <- (-1x64x2x4xf16, 64x64x3x3xf16)
        conv2d_53 = paddle._C_ops.conv2d(relu__50, parameter_305, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x1x2xf16) <- (-1x64x1x2xf16, 1x64x1x1xf16)
        add__43 = paddle._C_ops.add(conv2d_53, parameter_306)

        # pd_op.batch_norm_: (-1x64x1x2xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x1x2xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__43, parameter_307, parameter_308, parameter_309, parameter_310, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x2xf16) <- (-1x64x1x2xf16)
        relu__51 = paddle._C_ops.relu(batch_norm__318)

        # pd_op.nearest_interp: (-1x64x2x4xf16) <- (-1x64x1x2xf16, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(relu__51, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.conv2d: (-1x64x2x4xf16) <- (-1x64x2x4xf16, 64x64x3x3xf16)
        conv2d_54 = paddle._C_ops.conv2d(nearest_interp_0, parameter_311, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x2x4xf16) <- (-1x64x2x4xf16, 1x64x1x1xf16)
        add__44 = paddle._C_ops.add(conv2d_54, parameter_312)

        # pd_op.batch_norm_: (-1x64x2x4xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x2x4xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__44, parameter_313, parameter_314, parameter_315, parameter_316, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x2x4xf16) <- (-1x64x2x4xf16)
        relu__52 = paddle._C_ops.relu(batch_norm__324)

        # pd_op.add_: (-1x64x2x4xf16) <- (-1x64x2x4xf16, -1x64x2x4xf16)
        add__45 = paddle._C_ops.add(relu__52, relu__50)

        # pd_op.nearest_interp: (-1x64x4x8xf16) <- (-1x64x2x4xf16, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(add__45, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.conv2d: (-1x64x4x8xf16) <- (-1x64x4x8xf16, 64x64x3x3xf16)
        conv2d_55 = paddle._C_ops.conv2d(nearest_interp_1, parameter_317, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x4x8xf16) <- (-1x64x4x8xf16, 1x64x1x1xf16)
        add__46 = paddle._C_ops.add(conv2d_55, parameter_318)

        # pd_op.batch_norm_: (-1x64x4x8xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x4x8xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__46, parameter_319, parameter_320, parameter_321, parameter_322, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x4x8xf16) <- (-1x64x4x8xf16)
        relu__53 = paddle._C_ops.relu(batch_norm__330)

        # pd_op.add_: (-1x64x4x8xf16) <- (-1x64x4x8xf16, -1x64x4x8xf16)
        add__47 = paddle._C_ops.add(relu__53, relu__49)

        # pd_op.nearest_interp: (-1x64x8x16xf16) <- (-1x64x4x8xf16, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(add__47, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.conv2d: (-1x64x8x16xf16) <- (-1x64x8x16xf16, 64x64x3x3xf16)
        conv2d_56 = paddle._C_ops.conv2d(nearest_interp_2, parameter_323, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x8x16xf16) <- (-1x64x8x16xf16, 1x64x1x1xf16)
        add__48 = paddle._C_ops.add(conv2d_56, parameter_324)

        # pd_op.batch_norm_: (-1x64x8x16xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x8x16xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__48, parameter_325, parameter_326, parameter_327, parameter_328, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x8x16xf16) <- (-1x64x8x16xf16)
        relu__54 = paddle._C_ops.relu(batch_norm__336)

        # pd_op.add_: (-1x64x8x16xf16) <- (-1x64x8x16xf16, -1x64x8x16xf16)
        add__49 = paddle._C_ops.add(relu__54, relu__48)

        # pd_op.nearest_interp: (-1x64x8x32xf16) <- (-1x64x8x16xf16, None, None, None)
        nearest_interp_3 = paddle._C_ops.nearest_interp(add__49, None, None, None, 'NCHW', -1, 8, 32, [], 'nearest', False, 0)

        # pd_op.conv2d: (-1x512x8x32xf16) <- (-1x64x8x32xf16, 512x64x3x3xf16)
        conv2d_57 = paddle._C_ops.conv2d(nearest_interp_3, parameter_329, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x8x32xf16) <- (-1x512x8x32xf16, 1x512x1x1xf16)
        add__50 = paddle._C_ops.add(conv2d_57, parameter_330)

        # pd_op.batch_norm_: (-1x512x8x32xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x32xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__50, parameter_331, parameter_332, parameter_333, parameter_334, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x8x32xf16) <- (-1x512x8x32xf16)
        relu__55 = paddle._C_ops.relu(batch_norm__342)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_1 = [slice_11, parameter_335, parameter_336]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(slice_11, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_2 = [memcpy_h2d_0, parameter_335, parameter_336]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_0 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.full_with_tensor: (-1x26x512xf16) <- (1xf32, 3xi32)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(full_0, stack_0, paddle.float16)

        # pd_op.transpose: (26x-1x512xf16) <- (-1x26x512xf16)
        transpose_13 = paddle._C_ops.transpose(full_with_tensor_0, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (26x-1x512xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(transpose_13, 'float32'))

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_2, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_3 = [slice_12]

        # pd_op.slice: (-1x1x512xf16) <- (26x1x512xf16, 1xi64, [xi32])
        slice_13 = paddle._C_ops.slice(parameter_337, [0], constant_0, combine_3, [-1], [])

        # pd_op.add: (26x-1x512xf16) <- (26x-1x512xf16, -1x1x512xf16)
        add_1 = transpose_13 + slice_13

        # pd_op.transpose: (-1x26x512xf16) <- (26x-1x512xf16)
        transpose_14 = paddle._C_ops.transpose(add_1, [1, 0, 2])

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_18 = paddle.matmul(transpose_14, parameter_338, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__51 = paddle._C_ops.add(matmul_18, parameter_339)

        # pd_op.flatten_: (-1x512x256xf16, None) <- (-1x512x8x32xf16)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten(relu__55, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x256xf16) <- (-1x26x512xf16, -1x512x256xf16)
        matmul_19 = paddle.matmul(add__51, flatten__2, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x26x256xf16) <- (-1x26x256xf16, 1xf32)
        scale__3 = paddle._C_ops.scale(matmul_19, constant_9, float('0'), True)

        # pd_op.softmax_: (-1x26x256xf16) <- (-1x26x256xf16)
        softmax__3 = paddle._C_ops.softmax(scale__3, -1)

        # pd_op.flatten_: (-1x512x256xf16, None) <- (-1x512x8x32xf16)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten(transpose_12, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x256x512xf16) <- (-1x512x256xf16)
        transpose_15 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x256xf16, -1x256x512xf16)
        matmul_20 = paddle.matmul(softmax__3, transpose_15, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x26x37xf16) <- (-1x26x512xf16, 512x37xf16)
        matmul_21 = paddle.matmul(matmul_20, parameter_340, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x37xf16) <- (-1x26x37xf16, 37xf16)
        add__52 = paddle._C_ops.add(matmul_21, parameter_341)

        # pd_op.argmax: (-1x26xi64) <- (-1x26x37xf16, 1xi64)
        argmax_0 = paddle._C_ops.argmax(add__52, constant_10, False, False, paddle.int64)

        # pd_op.equal: (-1x26xb) <- (-1x26xi64, xi64)
        equal_0 = paddle._C_ops.equal(argmax_0, parameter_342)

        # pd_op.any: (-1xb) <- (-1x26xb)
        any_0 = paddle._C_ops.any(equal_0, [-1], False)

        # pd_op.cast: (-1x26xi32) <- (-1x26xb)
        cast_1 = paddle._C_ops.cast(equal_0, paddle.int32)

        # pd_op.cumsum_: (-1x26xi32) <- (-1x26xi32, 1xi32)
        cumsum__0 = paddle._C_ops.cumsum(cast_1, constant_11, False, False, False)

        # pd_op.equal: (-1x26xb) <- (-1x26xi32, xi32)
        equal_1 = paddle._C_ops.equal(cumsum__0, parameter_343)

        # pd_op.bitwise_and_: (-1x26xb) <- (-1x26xb, -1x26xb)
        bitwise_and__0 = paddle._C_ops.bitwise_and(equal_1, equal_0)

        # pd_op.cast: (-1x26xi32) <- (-1x26xb)
        cast_2 = paddle._C_ops.cast(bitwise_and__0, paddle.int32)

        # pd_op.argmax: (-1xi64) <- (-1x26xi32, 1xi64)
        argmax_1 = paddle._C_ops.argmax(cast_2, constant_10, False, False, paddle.int64)

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(argmax_1, constant_12, float('1'), True)

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(scale_0, full_0, paddle.int64, paddle.framework._current_expected_place())

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_like_0, constant_12, float('26'), True)

        # pd_op.where: (-1xi64) <- (-1xb, -1xi64, -1xi64)
        where_0 = paddle._C_ops.where(any_0, scale_0, scale_1)

        # pd_op.softmax_: (-1x26x37xf16) <- (-1x26x37xf16)
        softmax__4 = paddle._C_ops.softmax(add__52, -1)

        # pd_op.clip: (-1xi64) <- (-1xi64, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(where_0, constant_13, constant_14)

        # pd_op.share_data_: (-1x26x37xf16) <- (-1x26x37xf16)
        share_data__0 = softmax__4.detach()

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x37xf16, 37x512xf16)
        matmul_22 = paddle.matmul(share_data__0, parameter_344, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (26x-1x512xf16) <- (-1x26x512xf16)
        transpose_16 = paddle._C_ops.transpose(matmul_22, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (26x-1x512xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(transpose_16, 'float32'))

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_3, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_4 = [slice_14]

        # pd_op.slice: (-1x1x512xf16) <- (26x1x512xf16, 1xi64, [xi32])
        slice_15 = paddle._C_ops.slice(parameter_345, [0], constant_0, combine_4, [-1], [])

        # pd_op.add: (26x-1x512xf16) <- (26x-1x512xf16, -1x1x512xf16)
        add_2 = transpose_16 + slice_15

        # pd_op.dropout: (26x-1x512xf16, None) <- (26x-1x512xf16, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_2, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x26x512xf16) <- (26x-1x512xf16)
        transpose_17 = paddle._C_ops.transpose(dropout_32, [1, 0, 2])

        # pd_op.unsqueeze_: (-1x1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(clip_0, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (2xi32) <- (-1x1xi64)
        shape_4 = paddle._C_ops.shape(unsqueeze__0)

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_4, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32, 1xi32]) <- (xi32, 1xi32)
        combine_5 = [slice_16, constant_16]

        # pd_op.tile: (-1x26xi64) <- (1x26xi64, [xi32, 1xi32])
        tile_0 = paddle._C_ops.tile(parameter_346, combine_5)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_6 = [slice_16, parameter_335]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_1 = paddle._C_ops.memcpy_h2d(slice_16, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_7 = [memcpy_h2d_1, parameter_335]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_1 = paddle._C_ops.stack(combine_7, 0)

        # pd_op.full_with_tensor: (-1x26xf16) <- (1xf32, 2xi32)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_0, stack_1, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('-inf'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_8 = [slice_16, parameter_335]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_2 = paddle._C_ops.memcpy_h2d(slice_16, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_9 = [memcpy_h2d_2, parameter_335]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_2 = paddle._C_ops.stack(combine_9, 0)

        # pd_op.full_with_tensor: (-1x26xf16) <- (1xf32, 2xi32)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(full_1, stack_2, paddle.float16)

        # pd_op.greater_equal: (-1x26xb) <- (-1x26xi64, -1x1xi64)
        greater_equal_0 = paddle._C_ops.greater_equal(tile_0, unsqueeze__0)

        # pd_op.where_: (-1x26xf16) <- (-1x26xb, -1x26xf16, -1x26xf16)
        where__0 = paddle._C_ops.where(greater_equal_0, full_with_tensor_2, full_with_tensor_1)

        # pd_op.unsqueeze_: (-1x1x26xf16, None) <- (-1x26xf16, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(where__0, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x26x26xf16) <- (-1x1x26xf16, 26x26xf16)
        add_3 = unsqueeze__2 + parameter_347

        # pd_op.unsqueeze_: (-1x1x26x26xf16, None) <- (-1x26x26xf16, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_3, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_like: (-1x26x512xf16) <- (-1x26x512xf16, 1xf32)
        full_like_1 = paddle._C_ops.full_like(transpose_17, full_0, paddle.float16, paddle.framework._current_expected_place())

        # pd_op.transpose: (26x-1x512xf16) <- (-1x26x512xf16)
        transpose_18 = paddle._C_ops.transpose(full_like_1, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (26x-1x512xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(transpose_18, 'float32'))

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_5, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_10 = [slice_17]

        # pd_op.slice: (-1x1x512xf16) <- (26x1x512xf16, 1xi64, [xi32])
        slice_18 = paddle._C_ops.slice(parameter_348, [0], constant_0, combine_10, [-1], [])

        # pd_op.add: (26x-1x512xf16) <- (26x-1x512xf16, -1x1x512xf16)
        add_4 = transpose_18 + slice_18

        # pd_op.transpose: (-1x26x512xf16) <- (26x-1x512xf16)
        transpose_19 = paddle._C_ops.transpose(add_4, [1, 0, 2])

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_23 = paddle.matmul(transpose_19, parameter_349, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__53 = paddle._C_ops.add(matmul_23, parameter_350)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__53, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_20 = paddle._C_ops.transpose(reshape__14, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_24 = paddle.matmul(transpose_17, parameter_351, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__54 = paddle._C_ops.add(matmul_24, parameter_352)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__54, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_21 = paddle._C_ops.transpose(reshape__16, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_21, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(transpose_21, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_22 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_25 = paddle.matmul(transpose_20, transpose_22, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__4 = paddle._C_ops.scale(matmul_25, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__55 = paddle._C_ops.add(scale__4, unsqueeze__4)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__5 = paddle._C_ops.softmax(add__55, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__5, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_26 = paddle.matmul(dropout_34, slice_20, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_23 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_23, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_27 = paddle.matmul(reshape__18, parameter_353, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__56 = paddle._C_ops.add(matmul_27, parameter_354)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__56, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__57 = paddle._C_ops.add(transpose_19, dropout_36)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__57, parameter_355, parameter_356, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_28 = paddle.matmul(layer_norm_18, parameter_357, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__58 = paddle._C_ops.add(matmul_28, parameter_358)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__56 = paddle._C_ops.relu(add__58)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__56, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_29 = paddle.matmul(dropout_38, parameter_359, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__59 = paddle._C_ops.add(matmul_29, parameter_360)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__59, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_40, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__60 = paddle._C_ops.add(layer_norm_18, dropout_42)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__60, parameter_361, parameter_362, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_30 = paddle.matmul(layer_norm_21, parameter_363, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__61 = paddle._C_ops.add(matmul_30, parameter_364)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__61, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_24 = paddle._C_ops.transpose(reshape__20, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_31 = paddle.matmul(transpose_17, parameter_365, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__62 = paddle._C_ops.add(matmul_31, parameter_366)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__62, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_25 = paddle._C_ops.transpose(reshape__22, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(transpose_25, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_25, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_26 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_32 = paddle.matmul(transpose_24, transpose_26, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__5 = paddle._C_ops.scale(matmul_32, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__63 = paddle._C_ops.add(scale__5, unsqueeze__4)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__6 = paddle._C_ops.softmax(add__63, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__6, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_33 = paddle.matmul(dropout_44, slice_22, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_27 = paddle._C_ops.transpose(matmul_33, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_27, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_34 = paddle.matmul(reshape__24, parameter_367, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__64 = paddle._C_ops.add(matmul_34, parameter_368)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__64, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__65 = paddle._C_ops.add(layer_norm_21, dropout_46)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__65, parameter_369, parameter_370, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_35 = paddle.matmul(layer_norm_24, parameter_371, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__66 = paddle._C_ops.add(matmul_35, parameter_372)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__57 = paddle._C_ops.relu(add__66)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__57, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_36 = paddle.matmul(dropout_48, parameter_373, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__67 = paddle._C_ops.add(matmul_36, parameter_374)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__67, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_50, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__68 = paddle._C_ops.add(layer_norm_24, dropout_52)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__68, parameter_375, parameter_376, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_37 = paddle.matmul(layer_norm_27, parameter_377, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__69 = paddle._C_ops.add(matmul_37, parameter_378)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__69, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_28 = paddle._C_ops.transpose(reshape__26, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_38 = paddle.matmul(transpose_17, parameter_379, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__70 = paddle._C_ops.add(matmul_38, parameter_380)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__70, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_29 = paddle._C_ops.transpose(reshape__28, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_29, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_29, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_30 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_39 = paddle.matmul(transpose_28, transpose_30, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__6 = paddle._C_ops.scale(matmul_39, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__71 = paddle._C_ops.add(scale__6, unsqueeze__4)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__7 = paddle._C_ops.softmax(add__71, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__7, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_40 = paddle.matmul(dropout_54, slice_24, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_31 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_31, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_41 = paddle.matmul(reshape__30, parameter_381, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__72 = paddle._C_ops.add(matmul_41, parameter_382)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__72, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__73 = paddle._C_ops.add(layer_norm_27, dropout_56)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__73, parameter_383, parameter_384, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_42 = paddle.matmul(layer_norm_30, parameter_385, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__74 = paddle._C_ops.add(matmul_42, parameter_386)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__58 = paddle._C_ops.relu(add__74)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__58, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_43 = paddle.matmul(dropout_58, parameter_387, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__75 = paddle._C_ops.add(matmul_43, parameter_388)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__75, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_60, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__76 = paddle._C_ops.add(layer_norm_30, dropout_62)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__76, parameter_389, parameter_390, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_44 = paddle.matmul(layer_norm_33, parameter_391, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__77 = paddle._C_ops.add(matmul_44, parameter_392)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__77, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_32 = paddle._C_ops.transpose(reshape__32, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_45 = paddle.matmul(transpose_17, parameter_393, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__78 = paddle._C_ops.add(matmul_45, parameter_394)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__78, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_33 = paddle._C_ops.transpose(reshape__34, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_33, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_33, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_34 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_46 = paddle.matmul(transpose_32, transpose_34, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__7 = paddle._C_ops.scale(matmul_46, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__79 = paddle._C_ops.add(scale__7, unsqueeze__4)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__8 = paddle._C_ops.softmax(add__79, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__8, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_47 = paddle.matmul(dropout_64, slice_26, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_35 = paddle._C_ops.transpose(matmul_47, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_35, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_48 = paddle.matmul(reshape__36, parameter_395, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__80 = paddle._C_ops.add(matmul_48, parameter_396)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__80, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__81 = paddle._C_ops.add(layer_norm_33, dropout_66)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__81, parameter_397, parameter_398, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_49 = paddle.matmul(layer_norm_36, parameter_399, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__82 = paddle._C_ops.add(matmul_49, parameter_400)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__59 = paddle._C_ops.relu(add__82)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__59, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_50 = paddle.matmul(dropout_68, parameter_401, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__83 = paddle._C_ops.add(matmul_50, parameter_402)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__83, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_70, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__84 = paddle._C_ops.add(layer_norm_36, dropout_72)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__84, parameter_403, parameter_404, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([-1x26x512xf16, -1x26x512xf16]) <- (-1x26x512xf16, -1x26x512xf16)
        combine_11 = [layer_norm_39, matmul_20]

        # pd_op.concat: (-1x26x1024xf16) <- ([-1x26x512xf16, -1x26x512xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_11, constant_11)

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x1024xf16, 1024x512xf16)
        matmul_51 = paddle.matmul(concat_0, parameter_405, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__85 = paddle._C_ops.add(matmul_51, parameter_406)

        # pd_op.sigmoid_: (-1x26x512xf16) <- (-1x26x512xf16)
        sigmoid__0 = paddle._C_ops.sigmoid(add__85)

        # pd_op.multiply: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        multiply_0 = sigmoid__0 * matmul_20

        # pd_op.scale_: (-1x26x512xf16) <- (-1x26x512xf16, 1xf32)
        scale__8 = paddle._C_ops.scale(sigmoid__0, constant_20, float('1'), True)

        # pd_op.multiply_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        multiply__0 = paddle._C_ops.multiply(scale__8, layer_norm_39)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__86 = paddle._C_ops.add(multiply_0, multiply__0)

        # pd_op.matmul: (-1x26x37xf16) <- (-1x26x512xf16, 512x37xf16)
        matmul_52 = paddle.matmul(add__86, parameter_407, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x37xf16) <- (-1x26x37xf16, 37xf16)
        add__87 = paddle._C_ops.add(matmul_52, parameter_408)

        # pd_op.argmax: (-1x26xi64) <- (-1x26x37xf16, 1xi64)
        argmax_2 = paddle._C_ops.argmax(add__87, constant_10, False, False, paddle.int64)

        # pd_op.equal: (-1x26xb) <- (-1x26xi64, xi64)
        equal_2 = paddle._C_ops.equal(argmax_2, parameter_342)

        # pd_op.any: (-1xb) <- (-1x26xb)
        any_1 = paddle._C_ops.any(equal_2, [-1], False)

        # pd_op.cast: (-1x26xi32) <- (-1x26xb)
        cast_3 = paddle._C_ops.cast(equal_2, paddle.int32)

        # pd_op.cumsum_: (-1x26xi32) <- (-1x26xi32, 1xi32)
        cumsum__1 = paddle._C_ops.cumsum(cast_3, constant_11, False, False, False)

        # pd_op.equal: (-1x26xb) <- (-1x26xi32, xi32)
        equal_3 = paddle._C_ops.equal(cumsum__1, parameter_343)

        # pd_op.bitwise_and_: (-1x26xb) <- (-1x26xb, -1x26xb)
        bitwise_and__1 = paddle._C_ops.bitwise_and(equal_3, equal_2)

        # pd_op.cast: (-1x26xi32) <- (-1x26xb)
        cast_4 = paddle._C_ops.cast(bitwise_and__1, paddle.int32)

        # pd_op.argmax: (-1xi64) <- (-1x26xi32, 1xi64)
        argmax_3 = paddle._C_ops.argmax(cast_4, constant_10, False, False, paddle.int64)

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(argmax_3, constant_12, float('1'), True)

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_2 = paddle._C_ops.full_like(scale_2, full_0, paddle.int64, paddle.framework._current_expected_place())

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_like_2, constant_12, float('26'), True)

        # pd_op.where: (-1xi64) <- (-1xb, -1xi64, -1xi64)
        where_1 = paddle._C_ops.where(any_1, scale_2, scale_3)

        # pd_op.softmax_: (-1x26x37xf16) <- (-1x26x37xf16)
        softmax__9 = paddle._C_ops.softmax(add__87, -1)

        # pd_op.clip: (-1xi64) <- (-1xi64, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(where_1, constant_13, constant_14)

        # pd_op.share_data_: (-1x26x37xf16) <- (-1x26x37xf16)
        share_data__1 = softmax__9.detach()

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x37xf16, 37x512xf16)
        matmul_53 = paddle.matmul(share_data__1, parameter_344, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (26x-1x512xf16) <- (-1x26x512xf16)
        transpose_36 = paddle._C_ops.transpose(matmul_53, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (26x-1x512xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(transpose_36, 'float32'))

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(shape_6, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_12 = [slice_27]

        # pd_op.slice: (-1x1x512xf16) <- (26x1x512xf16, 1xi64, [xi32])
        slice_28 = paddle._C_ops.slice(parameter_345, [0], constant_0, combine_12, [-1], [])

        # pd_op.add: (26x-1x512xf16) <- (26x-1x512xf16, -1x1x512xf16)
        add_5 = transpose_36 + slice_28

        # pd_op.dropout: (26x-1x512xf16, None) <- (26x-1x512xf16, None, 1xf32)
        dropout_74, dropout_75 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_5, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x26x512xf16) <- (26x-1x512xf16)
        transpose_37 = paddle._C_ops.transpose(dropout_74, [1, 0, 2])

        # pd_op.unsqueeze_: (-1x1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(clip_1, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (2xi32) <- (-1x1xi64)
        shape_7 = paddle._C_ops.shape(unsqueeze__6)

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_7, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32, 1xi32]) <- (xi32, 1xi32)
        combine_13 = [slice_29, constant_16]

        # pd_op.tile: (-1x26xi64) <- (1x26xi64, [xi32, 1xi32])
        tile_1 = paddle._C_ops.tile(parameter_409, combine_13)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_14 = [slice_29, parameter_335]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_3 = paddle._C_ops.memcpy_h2d(slice_29, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_15 = [memcpy_h2d_3, parameter_335]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_3 = paddle._C_ops.stack(combine_15, 0)

        # pd_op.full_with_tensor: (-1x26xf16) <- (1xf32, 2xi32)
        full_with_tensor_3 = paddle._C_ops.full_with_tensor(full_0, stack_3, paddle.float16)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_16 = [slice_29, parameter_335]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_4 = paddle._C_ops.memcpy_h2d(slice_29, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_17 = [memcpy_h2d_4, parameter_335]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_4 = paddle._C_ops.stack(combine_17, 0)

        # pd_op.full_with_tensor: (-1x26xf16) <- (1xf32, 2xi32)
        full_with_tensor_4 = paddle._C_ops.full_with_tensor(full_1, stack_4, paddle.float16)

        # pd_op.greater_equal: (-1x26xb) <- (-1x26xi64, -1x1xi64)
        greater_equal_1 = paddle._C_ops.greater_equal(tile_1, unsqueeze__6)

        # pd_op.where_: (-1x26xf16) <- (-1x26xb, -1x26xf16, -1x26xf16)
        where__1 = paddle._C_ops.where(greater_equal_1, full_with_tensor_4, full_with_tensor_3)

        # pd_op.unsqueeze_: (-1x1x26xf16, None) <- (-1x26xf16, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(where__1, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x26x26xf16) <- (-1x1x26xf16, 26x26xf16)
        add_6 = unsqueeze__8 + parameter_347

        # pd_op.unsqueeze_: (-1x1x26x26xf16, None) <- (-1x26x26xf16, 1xi64)
        unsqueeze__10, unsqueeze__11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_6, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_like: (-1x26x512xf16) <- (-1x26x512xf16, 1xf32)
        full_like_3 = paddle._C_ops.full_like(transpose_37, full_0, paddle.float16, paddle.framework._current_expected_place())

        # pd_op.transpose: (26x-1x512xf16) <- (-1x26x512xf16)
        transpose_38 = paddle._C_ops.transpose(full_like_3, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (26x-1x512xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(transpose_38, 'float32'))

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(shape_8, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_18 = [slice_30]

        # pd_op.slice: (-1x1x512xf16) <- (26x1x512xf16, 1xi64, [xi32])
        slice_31 = paddle._C_ops.slice(parameter_348, [0], constant_0, combine_18, [-1], [])

        # pd_op.add: (26x-1x512xf16) <- (26x-1x512xf16, -1x1x512xf16)
        add_7 = transpose_38 + slice_31

        # pd_op.transpose: (-1x26x512xf16) <- (26x-1x512xf16)
        transpose_39 = paddle._C_ops.transpose(add_7, [1, 0, 2])

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_54 = paddle.matmul(transpose_39, parameter_349, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__88 = paddle._C_ops.add(matmul_54, parameter_350)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__88, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_40 = paddle._C_ops.transpose(reshape__38, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_55 = paddle.matmul(transpose_37, parameter_351, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__89 = paddle._C_ops.add(matmul_55, parameter_352)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__89, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_41 = paddle._C_ops.transpose(reshape__40, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(transpose_41, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_41, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_42 = paddle._C_ops.transpose(slice_32, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_56 = paddle.matmul(transpose_40, transpose_42, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__9 = paddle._C_ops.scale(matmul_56, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__90 = paddle._C_ops.add(scale__9, unsqueeze__10)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__10 = paddle._C_ops.softmax(add__90, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_76, dropout_77 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__10, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_57 = paddle.matmul(dropout_76, slice_33, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_43 = paddle._C_ops.transpose(matmul_57, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_43, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_58 = paddle.matmul(reshape__42, parameter_353, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__91 = paddle._C_ops.add(matmul_58, parameter_354)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_78, dropout_79 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__91, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__92 = paddle._C_ops.add(transpose_39, dropout_78)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__92, parameter_355, parameter_356, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_59 = paddle.matmul(layer_norm_42, parameter_357, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__93 = paddle._C_ops.add(matmul_59, parameter_358)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__60 = paddle._C_ops.relu(add__93)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_80, dropout_81 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__60, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_60 = paddle.matmul(dropout_80, parameter_359, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__94 = paddle._C_ops.add(matmul_60, parameter_360)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_82, dropout_83 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__94, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_84, dropout_85 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_82, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__95 = paddle._C_ops.add(layer_norm_42, dropout_84)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__95, parameter_361, parameter_362, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_61 = paddle.matmul(layer_norm_45, parameter_363, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__96 = paddle._C_ops.add(matmul_61, parameter_364)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__96, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_44 = paddle._C_ops.transpose(reshape__44, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_62 = paddle.matmul(transpose_37, parameter_365, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__97 = paddle._C_ops.add(matmul_62, parameter_366)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__97, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_45 = paddle._C_ops.transpose(reshape__46, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_45, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_45, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_46 = paddle._C_ops.transpose(slice_34, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_63 = paddle.matmul(transpose_44, transpose_46, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__10 = paddle._C_ops.scale(matmul_63, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__98 = paddle._C_ops.add(scale__10, unsqueeze__10)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__11 = paddle._C_ops.softmax(add__98, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_86, dropout_87 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__11, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_64 = paddle.matmul(dropout_86, slice_35, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_47 = paddle._C_ops.transpose(matmul_64, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_47, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_65 = paddle.matmul(reshape__48, parameter_367, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__99 = paddle._C_ops.add(matmul_65, parameter_368)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_88, dropout_89 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__99, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__100 = paddle._C_ops.add(layer_norm_45, dropout_88)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__100, parameter_369, parameter_370, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_66 = paddle.matmul(layer_norm_48, parameter_371, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__101 = paddle._C_ops.add(matmul_66, parameter_372)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__61 = paddle._C_ops.relu(add__101)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_90, dropout_91 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__61, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_67 = paddle.matmul(dropout_90, parameter_373, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__102 = paddle._C_ops.add(matmul_67, parameter_374)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_92, dropout_93 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__102, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_94, dropout_95 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_92, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__103 = paddle._C_ops.add(layer_norm_48, dropout_94)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__103, parameter_375, parameter_376, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_68 = paddle.matmul(layer_norm_51, parameter_377, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__104 = paddle._C_ops.add(matmul_68, parameter_378)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__104, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_48 = paddle._C_ops.transpose(reshape__50, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_69 = paddle.matmul(transpose_37, parameter_379, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__105 = paddle._C_ops.add(matmul_69, parameter_380)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__105, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_49 = paddle._C_ops.transpose(reshape__52, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(transpose_49, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(transpose_49, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_50 = paddle._C_ops.transpose(slice_36, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_70 = paddle.matmul(transpose_48, transpose_50, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__11 = paddle._C_ops.scale(matmul_70, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__106 = paddle._C_ops.add(scale__11, unsqueeze__10)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__12 = paddle._C_ops.softmax(add__106, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_96, dropout_97 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__12, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_71 = paddle.matmul(dropout_96, slice_37, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_51 = paddle._C_ops.transpose(matmul_71, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_51, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_72 = paddle.matmul(reshape__54, parameter_381, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__107 = paddle._C_ops.add(matmul_72, parameter_382)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_98, dropout_99 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__107, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__108 = paddle._C_ops.add(layer_norm_51, dropout_98)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__108, parameter_383, parameter_384, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_73 = paddle.matmul(layer_norm_54, parameter_385, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__109 = paddle._C_ops.add(matmul_73, parameter_386)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__62 = paddle._C_ops.relu(add__109)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_100, dropout_101 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__62, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_74 = paddle.matmul(dropout_100, parameter_387, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__110 = paddle._C_ops.add(matmul_74, parameter_388)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_102, dropout_103 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__110, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_104, dropout_105 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_102, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__111 = paddle._C_ops.add(layer_norm_54, dropout_104)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__111, parameter_389, parameter_390, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_75 = paddle.matmul(layer_norm_57, parameter_391, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__112 = paddle._C_ops.add(matmul_75, parameter_392)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__112, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_52 = paddle._C_ops.transpose(reshape__56, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_76 = paddle.matmul(transpose_37, parameter_393, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__113 = paddle._C_ops.add(matmul_76, parameter_394)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__113, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_53 = paddle._C_ops.transpose(reshape__58, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(transpose_53, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(transpose_53, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_54 = paddle._C_ops.transpose(slice_38, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_77 = paddle.matmul(transpose_52, transpose_54, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__12 = paddle._C_ops.scale(matmul_77, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__114 = paddle._C_ops.add(scale__12, unsqueeze__10)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__13 = paddle._C_ops.softmax(add__114, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_106, dropout_107 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__13, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_78 = paddle.matmul(dropout_106, slice_39, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_55 = paddle._C_ops.transpose(matmul_78, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_55, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_79 = paddle.matmul(reshape__60, parameter_395, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__115 = paddle._C_ops.add(matmul_79, parameter_396)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_108, dropout_109 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__115, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__116 = paddle._C_ops.add(layer_norm_57, dropout_108)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__116, parameter_397, parameter_398, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_80 = paddle.matmul(layer_norm_60, parameter_399, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__117 = paddle._C_ops.add(matmul_80, parameter_400)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__63 = paddle._C_ops.relu(add__117)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_110, dropout_111 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__63, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_81 = paddle.matmul(dropout_110, parameter_401, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__118 = paddle._C_ops.add(matmul_81, parameter_402)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_112, dropout_113 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__118, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_114, dropout_115 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_112, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__119 = paddle._C_ops.add(layer_norm_60, dropout_114)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__119, parameter_403, parameter_404, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([-1x26x512xf16, -1x26x512xf16]) <- (-1x26x512xf16, -1x26x512xf16)
        combine_19 = [layer_norm_63, matmul_20]

        # pd_op.concat: (-1x26x1024xf16) <- ([-1x26x512xf16, -1x26x512xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_19, constant_11)

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x1024xf16, 1024x512xf16)
        matmul_82 = paddle.matmul(concat_1, parameter_405, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__120 = paddle._C_ops.add(matmul_82, parameter_406)

        # pd_op.sigmoid_: (-1x26x512xf16) <- (-1x26x512xf16)
        sigmoid__1 = paddle._C_ops.sigmoid(add__120)

        # pd_op.multiply: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        multiply_1 = sigmoid__1 * matmul_20

        # pd_op.scale_: (-1x26x512xf16) <- (-1x26x512xf16, 1xf32)
        scale__13 = paddle._C_ops.scale(sigmoid__1, constant_20, float('1'), True)

        # pd_op.multiply_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        multiply__1 = paddle._C_ops.multiply(scale__13, layer_norm_63)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__121 = paddle._C_ops.add(multiply_1, multiply__1)

        # pd_op.matmul: (-1x26x37xf16) <- (-1x26x512xf16, 512x37xf16)
        matmul_83 = paddle.matmul(add__121, parameter_407, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x37xf16) <- (-1x26x37xf16, 37xf16)
        add__122 = paddle._C_ops.add(matmul_83, parameter_408)

        # pd_op.argmax: (-1x26xi64) <- (-1x26x37xf16, 1xi64)
        argmax_4 = paddle._C_ops.argmax(add__122, constant_10, False, False, paddle.int64)

        # pd_op.equal: (-1x26xb) <- (-1x26xi64, xi64)
        equal_4 = paddle._C_ops.equal(argmax_4, parameter_342)

        # pd_op.any: (-1xb) <- (-1x26xb)
        any_2 = paddle._C_ops.any(equal_4, [-1], False)

        # pd_op.cast: (-1x26xi32) <- (-1x26xb)
        cast_5 = paddle._C_ops.cast(equal_4, paddle.int32)

        # pd_op.cumsum_: (-1x26xi32) <- (-1x26xi32, 1xi32)
        cumsum__2 = paddle._C_ops.cumsum(cast_5, constant_11, False, False, False)

        # pd_op.equal: (-1x26xb) <- (-1x26xi32, xi32)
        equal_5 = paddle._C_ops.equal(cumsum__2, parameter_343)

        # pd_op.bitwise_and_: (-1x26xb) <- (-1x26xb, -1x26xb)
        bitwise_and__2 = paddle._C_ops.bitwise_and(equal_5, equal_4)

        # pd_op.cast: (-1x26xi32) <- (-1x26xb)
        cast_6 = paddle._C_ops.cast(bitwise_and__2, paddle.int32)

        # pd_op.argmax: (-1xi64) <- (-1x26xi32, 1xi64)
        argmax_5 = paddle._C_ops.argmax(cast_6, constant_10, False, False, paddle.int64)

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_4 = paddle._C_ops.scale(argmax_5, constant_12, float('1'), True)

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_4 = paddle._C_ops.full_like(scale_4, full_0, paddle.int64, paddle.framework._current_expected_place())

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_5 = paddle._C_ops.scale(full_like_4, constant_12, float('26'), True)

        # pd_op.where: (-1xi64) <- (-1xb, -1xi64, -1xi64)
        where_2 = paddle._C_ops.where(any_2, scale_4, scale_5)

        # pd_op.softmax_: (-1x26x37xf16) <- (-1x26x37xf16)
        softmax__14 = paddle._C_ops.softmax(add__122, -1)

        # pd_op.clip: (-1xi64) <- (-1xi64, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(where_2, constant_13, constant_14)

        # pd_op.share_data_: (-1x26x37xf16) <- (-1x26x37xf16)
        share_data__2 = softmax__14.detach()

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x37xf16, 37x512xf16)
        matmul_84 = paddle.matmul(share_data__2, parameter_344, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (26x-1x512xf16) <- (-1x26x512xf16)
        transpose_56 = paddle._C_ops.transpose(matmul_84, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (26x-1x512xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(transpose_56, 'float32'))

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(shape_9, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_20 = [slice_40]

        # pd_op.slice: (-1x1x512xf16) <- (26x1x512xf16, 1xi64, [xi32])
        slice_41 = paddle._C_ops.slice(parameter_345, [0], constant_0, combine_20, [-1], [])

        # pd_op.add: (26x-1x512xf16) <- (26x-1x512xf16, -1x1x512xf16)
        add_8 = transpose_56 + slice_41

        # pd_op.dropout: (26x-1x512xf16, None) <- (26x-1x512xf16, None, 1xf32)
        dropout_116, dropout_117 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_8, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x26x512xf16) <- (26x-1x512xf16)
        transpose_57 = paddle._C_ops.transpose(dropout_116, [1, 0, 2])

        # pd_op.unsqueeze_: (-1x1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__12, unsqueeze__13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(clip_2, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (2xi32) <- (-1x1xi64)
        shape_10 = paddle._C_ops.shape(unsqueeze__12)

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(shape_10, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32, 1xi32]) <- (xi32, 1xi32)
        combine_21 = [slice_42, constant_16]

        # pd_op.tile: (-1x26xi64) <- (1x26xi64, [xi32, 1xi32])
        tile_2 = paddle._C_ops.tile(parameter_410, combine_21)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_22 = [slice_42, parameter_335]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_5 = paddle._C_ops.memcpy_h2d(slice_42, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_23 = [memcpy_h2d_5, parameter_335]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_5 = paddle._C_ops.stack(combine_23, 0)

        # pd_op.full_with_tensor: (-1x26xf16) <- (1xf32, 2xi32)
        full_with_tensor_5 = paddle._C_ops.full_with_tensor(full_0, stack_5, paddle.float16)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_24 = [slice_42, parameter_335]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_6 = paddle._C_ops.memcpy_h2d(slice_42, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_25 = [memcpy_h2d_6, parameter_335]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_6 = paddle._C_ops.stack(combine_25, 0)

        # pd_op.full_with_tensor: (-1x26xf16) <- (1xf32, 2xi32)
        full_with_tensor_6 = paddle._C_ops.full_with_tensor(full_1, stack_6, paddle.float16)

        # pd_op.greater_equal: (-1x26xb) <- (-1x26xi64, -1x1xi64)
        greater_equal_2 = paddle._C_ops.greater_equal(tile_2, unsqueeze__12)

        # pd_op.where_: (-1x26xf16) <- (-1x26xb, -1x26xf16, -1x26xf16)
        where__2 = paddle._C_ops.where(greater_equal_2, full_with_tensor_6, full_with_tensor_5)

        # pd_op.unsqueeze_: (-1x1x26xf16, None) <- (-1x26xf16, 1xi64)
        unsqueeze__14, unsqueeze__15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(where__2, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x26x26xf16) <- (-1x1x26xf16, 26x26xf16)
        add_9 = unsqueeze__14 + parameter_347

        # pd_op.unsqueeze_: (-1x1x26x26xf16, None) <- (-1x26x26xf16, 1xi64)
        unsqueeze__16, unsqueeze__17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_9, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_like: (-1x26x512xf16) <- (-1x26x512xf16, 1xf32)
        full_like_5 = paddle._C_ops.full_like(transpose_57, full_0, paddle.float16, paddle.framework._current_expected_place())

        # pd_op.transpose: (26x-1x512xf16) <- (-1x26x512xf16)
        transpose_58 = paddle._C_ops.transpose(full_like_5, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (26x-1x512xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(transpose_58, 'float32'))

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(shape_11, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_26 = [slice_43]

        # pd_op.slice: (-1x1x512xf16) <- (26x1x512xf16, 1xi64, [xi32])
        slice_44 = paddle._C_ops.slice(parameter_348, [0], constant_0, combine_26, [-1], [])

        # pd_op.add: (26x-1x512xf16) <- (26x-1x512xf16, -1x1x512xf16)
        add_10 = transpose_58 + slice_44

        # pd_op.transpose: (-1x26x512xf16) <- (26x-1x512xf16)
        transpose_59 = paddle._C_ops.transpose(add_10, [1, 0, 2])

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_85 = paddle.matmul(transpose_59, parameter_349, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__123 = paddle._C_ops.add(matmul_85, parameter_350)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__123, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_60 = paddle._C_ops.transpose(reshape__62, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_86 = paddle.matmul(transpose_57, parameter_351, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__124 = paddle._C_ops.add(matmul_86, parameter_352)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__124, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_61 = paddle._C_ops.transpose(reshape__64, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(transpose_61, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(transpose_61, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_62 = paddle._C_ops.transpose(slice_45, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_87 = paddle.matmul(transpose_60, transpose_62, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__14 = paddle._C_ops.scale(matmul_87, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__125 = paddle._C_ops.add(scale__14, unsqueeze__16)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__15 = paddle._C_ops.softmax(add__125, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_118, dropout_119 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__15, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_88 = paddle.matmul(dropout_118, slice_46, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_63 = paddle._C_ops.transpose(matmul_88, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_63, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_89 = paddle.matmul(reshape__66, parameter_353, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__126 = paddle._C_ops.add(matmul_89, parameter_354)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_120, dropout_121 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__126, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__127 = paddle._C_ops.add(transpose_59, dropout_120)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__127, parameter_355, parameter_356, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_90 = paddle.matmul(layer_norm_66, parameter_357, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__128 = paddle._C_ops.add(matmul_90, parameter_358)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__64 = paddle._C_ops.relu(add__128)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_122, dropout_123 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__64, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_91 = paddle.matmul(dropout_122, parameter_359, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__129 = paddle._C_ops.add(matmul_91, parameter_360)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_124, dropout_125 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__129, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_126, dropout_127 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_124, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__130 = paddle._C_ops.add(layer_norm_66, dropout_126)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__130, parameter_361, parameter_362, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_92 = paddle.matmul(layer_norm_69, parameter_363, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__131 = paddle._C_ops.add(matmul_92, parameter_364)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__131, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_64 = paddle._C_ops.transpose(reshape__68, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_93 = paddle.matmul(transpose_57, parameter_365, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__132 = paddle._C_ops.add(matmul_93, parameter_366)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__132, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_65 = paddle._C_ops.transpose(reshape__70, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(transpose_65, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(transpose_65, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_66 = paddle._C_ops.transpose(slice_47, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_94 = paddle.matmul(transpose_64, transpose_66, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__15 = paddle._C_ops.scale(matmul_94, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__133 = paddle._C_ops.add(scale__15, unsqueeze__16)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__16 = paddle._C_ops.softmax(add__133, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_128, dropout_129 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__16, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_95 = paddle.matmul(dropout_128, slice_48, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_67 = paddle._C_ops.transpose(matmul_95, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_67, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_96 = paddle.matmul(reshape__72, parameter_367, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__134 = paddle._C_ops.add(matmul_96, parameter_368)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_130, dropout_131 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__134, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__135 = paddle._C_ops.add(layer_norm_69, dropout_130)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__135, parameter_369, parameter_370, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_97 = paddle.matmul(layer_norm_72, parameter_371, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__136 = paddle._C_ops.add(matmul_97, parameter_372)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__65 = paddle._C_ops.relu(add__136)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_132, dropout_133 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__65, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_98 = paddle.matmul(dropout_132, parameter_373, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__137 = paddle._C_ops.add(matmul_98, parameter_374)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_134, dropout_135 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__137, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_136, dropout_137 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_134, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__138 = paddle._C_ops.add(layer_norm_72, dropout_136)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__138, parameter_375, parameter_376, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_99 = paddle.matmul(layer_norm_75, parameter_377, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__139 = paddle._C_ops.add(matmul_99, parameter_378)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__139, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_68 = paddle._C_ops.transpose(reshape__74, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_100 = paddle.matmul(transpose_57, parameter_379, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__140 = paddle._C_ops.add(matmul_100, parameter_380)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__140, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_69 = paddle._C_ops.transpose(reshape__76, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(transpose_69, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(transpose_69, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_70 = paddle._C_ops.transpose(slice_49, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_101 = paddle.matmul(transpose_68, transpose_70, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__16 = paddle._C_ops.scale(matmul_101, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__141 = paddle._C_ops.add(scale__16, unsqueeze__16)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__17 = paddle._C_ops.softmax(add__141, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_138, dropout_139 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__17, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_102 = paddle.matmul(dropout_138, slice_50, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_71 = paddle._C_ops.transpose(matmul_102, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_71, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_103 = paddle.matmul(reshape__78, parameter_381, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__142 = paddle._C_ops.add(matmul_103, parameter_382)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_140, dropout_141 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__142, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__143 = paddle._C_ops.add(layer_norm_75, dropout_140)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__143, parameter_383, parameter_384, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_104 = paddle.matmul(layer_norm_78, parameter_385, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__144 = paddle._C_ops.add(matmul_104, parameter_386)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__66 = paddle._C_ops.relu(add__144)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_142, dropout_143 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__66, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_105 = paddle.matmul(dropout_142, parameter_387, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__145 = paddle._C_ops.add(matmul_105, parameter_388)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_144, dropout_145 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__145, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_146, dropout_147 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_144, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__146 = paddle._C_ops.add(layer_norm_78, dropout_146)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__146, parameter_389, parameter_390, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_106 = paddle.matmul(layer_norm_81, parameter_391, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__147 = paddle._C_ops.add(matmul_106, parameter_392)

        # pd_op.reshape_: (-1x26x8x64xf16, 0x-1x26x512xf16) <- (-1x26x512xf16, 4xi64)
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__147, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x26x64xf16) <- (-1x26x8x64xf16)
        transpose_72 = paddle._C_ops.transpose(reshape__80, [0, 2, 1, 3])

        # pd_op.matmul: (-1x26x1024xf16) <- (-1x26x512xf16, 512x1024xf16)
        matmul_107 = paddle.matmul(transpose_57, parameter_393, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x1024xf16) <- (-1x26x1024xf16, 1024xf16)
        add__148 = paddle._C_ops.add(matmul_107, parameter_394)

        # pd_op.reshape_: (-1x26x2x8x64xf16, 0x-1x26x1024xf16) <- (-1x26x1024xf16, 5xi64)
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__148, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x26x64xf16) <- (-1x26x2x8x64xf16)
        transpose_73 = paddle._C_ops.transpose(reshape__82, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(transpose_73, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x26x64xf16) <- (2x-1x8x26x64xf16, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(transpose_73, [0], constant_1, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x26xf16) <- (-1x8x26x64xf16)
        transpose_74 = paddle._C_ops.transpose(slice_51, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x26x26xf16) <- (-1x8x26x64xf16, -1x8x64x26xf16)
        matmul_108 = paddle.matmul(transpose_72, transpose_74, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, 1xf32)
        scale__17 = paddle._C_ops.scale(matmul_108, constant_6, float('0'), True)

        # pd_op.add_: (-1x8x26x26xf16) <- (-1x8x26x26xf16, -1x1x26x26xf16)
        add__149 = paddle._C_ops.add(scale__17, unsqueeze__16)

        # pd_op.softmax_: (-1x8x26x26xf16) <- (-1x8x26x26xf16)
        softmax__18 = paddle._C_ops.softmax(add__149, -1)

        # pd_op.dropout: (-1x8x26x26xf16, None) <- (-1x8x26x26xf16, None, 1xf32)
        dropout_148, dropout_149 = (lambda x, f: f(x))(paddle._C_ops.dropout(softmax__18, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x26x64xf16) <- (-1x8x26x26xf16, -1x8x26x64xf16)
        matmul_109 = paddle.matmul(dropout_148, slice_52, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x26x8x64xf16) <- (-1x8x26x64xf16)
        transpose_75 = paddle._C_ops.transpose(matmul_109, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x26x512xf16, 0x-1x26x8x64xf16) <- (-1x26x8x64xf16, 3xi64)
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_75, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x512xf16, 512x512xf16)
        matmul_110 = paddle.matmul(reshape__84, parameter_395, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__150 = paddle._C_ops.add(matmul_110, parameter_396)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_150, dropout_151 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__150, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__151 = paddle._C_ops.add(layer_norm_81, dropout_150)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__151, parameter_397, parameter_398, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x26x2048xf16) <- (-1x26x512xf16, 512x2048xf16)
        matmul_111 = paddle.matmul(layer_norm_84, parameter_399, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x2048xf16) <- (-1x26x2048xf16, 2048xf16)
        add__152 = paddle._C_ops.add(matmul_111, parameter_400)

        # pd_op.relu_: (-1x26x2048xf16) <- (-1x26x2048xf16)
        relu__67 = paddle._C_ops.relu(add__152)

        # pd_op.dropout: (-1x26x2048xf16, None) <- (-1x26x2048xf16, None, 1xf32)
        dropout_152, dropout_153 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__67, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x2048xf16, 2048x512xf16)
        matmul_112 = paddle.matmul(dropout_152, parameter_401, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__153 = paddle._C_ops.add(matmul_112, parameter_402)

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_154, dropout_155 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__153, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x26x512xf16, None) <- (-1x26x512xf16, None, 1xf32)
        dropout_156, dropout_157 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_154, None, constant_2, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__154 = paddle._C_ops.add(layer_norm_84, dropout_156)

        # pd_op.layer_norm: (-1x26x512xf16, -1x26xf32, -1x26xf32) <- (-1x26x512xf16, 512xf32, 512xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__154, parameter_403, parameter_404, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([-1x26x512xf16, -1x26x512xf16]) <- (-1x26x512xf16, -1x26x512xf16)
        combine_27 = [layer_norm_87, matmul_20]

        # pd_op.concat: (-1x26x1024xf16) <- ([-1x26x512xf16, -1x26x512xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_27, constant_11)

        # pd_op.matmul: (-1x26x512xf16) <- (-1x26x1024xf16, 1024x512xf16)
        matmul_113 = paddle.matmul(concat_2, parameter_405, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, 512xf16)
        add__155 = paddle._C_ops.add(matmul_113, parameter_406)

        # pd_op.sigmoid_: (-1x26x512xf16) <- (-1x26x512xf16)
        sigmoid__2 = paddle._C_ops.sigmoid(add__155)

        # pd_op.multiply: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        multiply_2 = sigmoid__2 * matmul_20

        # pd_op.scale_: (-1x26x512xf16) <- (-1x26x512xf16, 1xf32)
        scale__18 = paddle._C_ops.scale(sigmoid__2, constant_20, float('1'), True)

        # pd_op.multiply_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        multiply__2 = paddle._C_ops.multiply(scale__18, layer_norm_87)

        # pd_op.add_: (-1x26x512xf16) <- (-1x26x512xf16, -1x26x512xf16)
        add__156 = paddle._C_ops.add(multiply_2, multiply__2)

        # pd_op.matmul: (-1x26x37xf16) <- (-1x26x512xf16, 512x37xf16)
        matmul_114 = paddle.matmul(add__156, parameter_407, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x37xf16) <- (-1x26x37xf16, 37xf16)
        add__157 = paddle._C_ops.add(matmul_114, parameter_408)

        # pd_op.softmax_: (-1x26x37xf16) <- (-1x26x37xf16)
        softmax__19 = paddle._C_ops.softmax(add__157, -1)

        # pd_op.scale_: (-1x26x37xf16) <- (-1x26x37xf16, 1xf32)
        scale__19 = paddle._C_ops.scale(softmax__19, constant_12, float('0'), True)

        # pd_op.cast: (-1x26x37xf32) <- (-1x26x37xf16)
        cast_7 = paddle._C_ops.cast(scale__19, paddle.float32)
        return cast_7



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

    def forward(self, parameter_410, parameter_409, constant_20, constant_19, constant_18, constant_17, parameter_347, constant_16, parameter_346, constant_15, constant_14, constant_13, constant_12, parameter_343, constant_11, parameter_342, constant_10, constant_9, parameter_336, parameter_335, parameter_330, parameter_324, parameter_318, parameter_312, parameter_306, parameter_300, parameter_294, parameter_288, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_251, parameter_252, parameter_253, parameter_254, parameter_256, parameter_255, parameter_257, parameter_258, parameter_259, parameter_260, parameter_262, parameter_261, parameter_263, parameter_264, parameter_265, parameter_266, parameter_268, parameter_267, parameter_269, parameter_270, parameter_271, parameter_272, parameter_274, parameter_273, parameter_275, parameter_276, parameter_277, parameter_278, parameter_280, parameter_279, parameter_281, parameter_282, parameter_283, parameter_284, parameter_286, parameter_285, parameter_287, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_310, parameter_307, parameter_309, parameter_308, parameter_311, parameter_316, parameter_313, parameter_315, parameter_314, parameter_317, parameter_322, parameter_319, parameter_321, parameter_320, parameter_323, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_334, parameter_331, parameter_333, parameter_332, parameter_337, parameter_338, parameter_339, parameter_340, parameter_341, parameter_344, parameter_345, parameter_348, parameter_349, parameter_350, parameter_351, parameter_352, parameter_353, parameter_354, parameter_356, parameter_355, parameter_357, parameter_358, parameter_359, parameter_360, parameter_362, parameter_361, parameter_363, parameter_364, parameter_365, parameter_366, parameter_367, parameter_368, parameter_370, parameter_369, parameter_371, parameter_372, parameter_373, parameter_374, parameter_376, parameter_375, parameter_377, parameter_378, parameter_379, parameter_380, parameter_381, parameter_382, parameter_384, parameter_383, parameter_385, parameter_386, parameter_387, parameter_388, parameter_390, parameter_389, parameter_391, parameter_392, parameter_393, parameter_394, parameter_395, parameter_396, parameter_398, parameter_397, parameter_399, parameter_400, parameter_401, parameter_402, parameter_404, parameter_403, parameter_405, parameter_406, parameter_407, parameter_408, feed_0):
        return self.builtin_module_2060_0_0(parameter_410, parameter_409, constant_20, constant_19, constant_18, constant_17, parameter_347, constant_16, parameter_346, constant_15, constant_14, constant_13, constant_12, parameter_343, constant_11, parameter_342, constant_10, constant_9, parameter_336, parameter_335, parameter_330, parameter_324, parameter_318, parameter_312, parameter_306, parameter_300, parameter_294, parameter_288, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_251, parameter_252, parameter_253, parameter_254, parameter_256, parameter_255, parameter_257, parameter_258, parameter_259, parameter_260, parameter_262, parameter_261, parameter_263, parameter_264, parameter_265, parameter_266, parameter_268, parameter_267, parameter_269, parameter_270, parameter_271, parameter_272, parameter_274, parameter_273, parameter_275, parameter_276, parameter_277, parameter_278, parameter_280, parameter_279, parameter_281, parameter_282, parameter_283, parameter_284, parameter_286, parameter_285, parameter_287, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_310, parameter_307, parameter_309, parameter_308, parameter_311, parameter_316, parameter_313, parameter_315, parameter_314, parameter_317, parameter_322, parameter_319, parameter_321, parameter_320, parameter_323, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_334, parameter_331, parameter_333, parameter_332, parameter_337, parameter_338, parameter_339, parameter_340, parameter_341, parameter_344, parameter_345, parameter_348, parameter_349, parameter_350, parameter_351, parameter_352, parameter_353, parameter_354, parameter_356, parameter_355, parameter_357, parameter_358, parameter_359, parameter_360, parameter_362, parameter_361, parameter_363, parameter_364, parameter_365, parameter_366, parameter_367, parameter_368, parameter_370, parameter_369, parameter_371, parameter_372, parameter_373, parameter_374, parameter_376, parameter_375, parameter_377, parameter_378, parameter_379, parameter_380, parameter_381, parameter_382, parameter_384, parameter_383, parameter_385, parameter_386, parameter_387, parameter_388, parameter_390, parameter_389, parameter_391, parameter_392, parameter_393, parameter_394, parameter_395, parameter_396, parameter_398, parameter_397, parameter_399, parameter_400, parameter_401, parameter_402, parameter_404, parameter_403, parameter_405, parameter_406, parameter_407, parameter_408, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2060_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_410
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype='int64').reshape([1, 26]),
            # parameter_409
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype='int64').reshape([1, 26]),
            # constant_20
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_19
            paddle.to_tensor([0, 26, 512], dtype='int64').reshape([3]),
            # constant_18
            paddle.to_tensor([0, 26, 2, 8, 64], dtype='int64').reshape([5]),
            # constant_17
            paddle.to_tensor([0, 26, 8, 64], dtype='int64').reshape([4]),
            # parameter_347
            paddle.uniform([26, 26], dtype='float16', min=0, max=0.5),
            # constant_16
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # parameter_346
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype='int64').reshape([1, 26]),
            # constant_15
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            # constant_14
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_13
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_12
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.to_tensor([1], dtype='int32').reshape([]),
            # constant_11
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            # parameter_342
            paddle.to_tensor([0], dtype='int64').reshape([]),
            # constant_10
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            # constant_9
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.to_tensor([512], dtype='int32').reshape([]),
            # parameter_335
            paddle.to_tensor([26], dtype='int32').reshape([]),
            # parameter_330
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_324
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_318
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_312
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_306
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_300
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_8
            paddle.to_tensor([0, 8, 32, 512], dtype='int64').reshape([4]),
            # constant_7
            paddle.to_tensor([0, 256, 512], dtype='int64').reshape([3]),
            # constant_6
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_5
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            # constant_4
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_3
            paddle.to_tensor([0, 256, 3, 8, 64], dtype='int64').reshape([5]),
            # constant_2
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_0
            paddle.uniform([32, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([64, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_55
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
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_75
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
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([256, 1, 512], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_252
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_258
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_260
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_262
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_265
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_266
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_270
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_272
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_274
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_276
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_280
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_282
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_283
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_284
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_286
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([64, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_310
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_316
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_322
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_328
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([512, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([26, 1, 512], dtype='float16', min=0, max=0.5),
            # parameter_338
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_340
            paddle.uniform([512, 37], dtype='float16', min=0, max=0.5),
            # parameter_341
            paddle.uniform([37], dtype='float16', min=0, max=0.5),
            # parameter_344
            paddle.uniform([37, 512], dtype='float16', min=0, max=0.5),
            # parameter_345
            paddle.uniform([26, 1, 512], dtype='float16', min=0, max=0.5),
            # parameter_348
            paddle.uniform([26, 1, 512], dtype='float16', min=0, max=0.5),
            # parameter_349
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_350
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_351
            paddle.uniform([512, 1024], dtype='float16', min=0, max=0.5),
            # parameter_352
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_353
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_354
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_356
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_358
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_359
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_360
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_362
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_365
            paddle.uniform([512, 1024], dtype='float16', min=0, max=0.5),
            # parameter_366
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_367
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_368
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_370
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_372
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_373
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_374
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_376
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_378
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_379
            paddle.uniform([512, 1024], dtype='float16', min=0, max=0.5),
            # parameter_380
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_381
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_382
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_384
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_386
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_387
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_388
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_390
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_392
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_393
            paddle.uniform([512, 1024], dtype='float16', min=0, max=0.5),
            # parameter_394
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_395
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_396
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_398
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_400
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_401
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_402
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_404
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([1024, 512], dtype='float16', min=0, max=0.5),
            # parameter_406
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_407
            paddle.uniform([512, 37], dtype='float16', min=0, max=0.5),
            # parameter_408
            paddle.uniform([37], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 32, 128], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_410
            paddle.static.InputSpec(shape=[1, 26], dtype='int64'),
            # parameter_409
            paddle.static.InputSpec(shape=[1, 26], dtype='int64'),
            # constant_20
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_19
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_18
            paddle.static.InputSpec(shape=[5], dtype='int64'),
            # constant_17
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # parameter_347
            paddle.static.InputSpec(shape=[26, 26], dtype='float16'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_346
            paddle.static.InputSpec(shape=[1, 26], dtype='int64'),
            # constant_15
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_342
            paddle.static.InputSpec(shape=[], dtype='int64'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # parameter_335
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # parameter_330
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_324
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_318
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_312
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_306
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_300
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # constant_8
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_7
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_3
            paddle.static.InputSpec(shape=[5], dtype='int64'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[64, 32, 1, 1], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64, 32, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_55
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
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_75
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
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[256, 1, 512], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_252
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_258
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_260
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_262
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_265
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_266
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_270
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_272
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_274
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_276
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_280
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_282
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_283
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_284
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_286
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[64, 512, 3, 3], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_310
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_316
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_322
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_328
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[512, 64, 3, 3], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[26, 1, 512], dtype='float16'),
            # parameter_338
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_340
            paddle.static.InputSpec(shape=[512, 37], dtype='float16'),
            # parameter_341
            paddle.static.InputSpec(shape=[37], dtype='float16'),
            # parameter_344
            paddle.static.InputSpec(shape=[37, 512], dtype='float16'),
            # parameter_345
            paddle.static.InputSpec(shape=[26, 1, 512], dtype='float16'),
            # parameter_348
            paddle.static.InputSpec(shape=[26, 1, 512], dtype='float16'),
            # parameter_349
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_350
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_351
            paddle.static.InputSpec(shape=[512, 1024], dtype='float16'),
            # parameter_352
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_353
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_354
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_356
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_358
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_359
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_360
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_362
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_365
            paddle.static.InputSpec(shape=[512, 1024], dtype='float16'),
            # parameter_366
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_367
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_368
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_370
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_372
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_373
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_374
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_376
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_378
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_379
            paddle.static.InputSpec(shape=[512, 1024], dtype='float16'),
            # parameter_380
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_381
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_382
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_384
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_386
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_387
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_388
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_390
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_392
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_393
            paddle.static.InputSpec(shape=[512, 1024], dtype='float16'),
            # parameter_394
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_395
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_396
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_398
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_400
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_401
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_402
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_404
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[1024, 512], dtype='float16'),
            # parameter_406
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_407
            paddle.static.InputSpec(shape=[512, 37], dtype='float16'),
            # parameter_408
            paddle.static.InputSpec(shape=[37], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 32, 128], dtype='float32'),
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