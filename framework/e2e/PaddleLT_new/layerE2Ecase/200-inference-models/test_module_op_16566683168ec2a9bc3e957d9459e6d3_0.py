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
    return [634][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2432_0_0(self, parameter_345, parameter_331, constant_14, constant_13, constant_12, constant_11, parameter_317, parameter_311, parameter_297, parameter_283, parameter_269, parameter_255, parameter_241, parameter_227, parameter_213, parameter_199, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_185, parameter_179, parameter_177, parameter_175, parameter_169, parameter_167, parameter_165, parameter_159, parameter_157, parameter_155, parameter_149, parameter_147, parameter_145, parameter_139, parameter_137, parameter_135, parameter_129, parameter_127, parameter_125, parameter_119, parameter_117, parameter_115, parameter_109, parameter_107, parameter_105, parameter_99, parameter_97, parameter_95, parameter_89, parameter_87, parameter_85, parameter_79, parameter_73, parameter_71, parameter_69, parameter_63, parameter_61, parameter_59, parameter_53, parameter_51, parameter_49, parameter_43, parameter_41, parameter_39, parameter_33, parameter_31, parameter_29, parameter_23, parameter_21, parameter_19, parameter_13, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_17, parameter_14, parameter_16, parameter_15, parameter_18, parameter_20, parameter_22, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_30, parameter_32, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_40, parameter_42, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_50, parameter_52, parameter_57, parameter_54, parameter_56, parameter_55, parameter_58, parameter_60, parameter_62, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_70, parameter_72, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_86, parameter_88, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_96, parameter_98, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_106, parameter_108, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_116, parameter_118, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_126, parameter_128, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_136, parameter_138, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_146, parameter_148, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_156, parameter_158, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_166, parameter_168, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_176, parameter_178, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_187, parameter_186, parameter_188, parameter_189, parameter_190, parameter_191, parameter_193, parameter_192, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_201, parameter_200, parameter_202, parameter_203, parameter_204, parameter_205, parameter_207, parameter_206, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_215, parameter_214, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_229, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_235, parameter_234, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_243, parameter_242, parameter_244, parameter_245, parameter_246, parameter_247, parameter_249, parameter_248, parameter_250, parameter_251, parameter_252, parameter_253, parameter_254, parameter_257, parameter_256, parameter_258, parameter_259, parameter_260, parameter_261, parameter_263, parameter_262, parameter_264, parameter_265, parameter_266, parameter_267, parameter_268, parameter_271, parameter_270, parameter_272, parameter_273, parameter_274, parameter_275, parameter_277, parameter_276, parameter_278, parameter_279, parameter_280, parameter_281, parameter_282, parameter_285, parameter_284, parameter_286, parameter_287, parameter_288, parameter_289, parameter_291, parameter_290, parameter_292, parameter_293, parameter_294, parameter_295, parameter_296, parameter_299, parameter_298, parameter_300, parameter_301, parameter_302, parameter_303, parameter_305, parameter_304, parameter_306, parameter_307, parameter_308, parameter_309, parameter_310, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_330, parameter_333, parameter_332, parameter_334, parameter_335, parameter_336, parameter_337, parameter_339, parameter_338, parameter_340, parameter_341, parameter_342, parameter_343, parameter_344, parameter_347, parameter_346, parameter_348, parameter_349, parameter_350, parameter_351, parameter_353, parameter_352, parameter_354, parameter_355, parameter_356, parameter_357, parameter_361, parameter_358, parameter_360, parameter_359, parameter_362, parameter_363, feed_0):

        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 1x32x1x1xf32)
        add__0 = paddle._C_ops.add(conv2d_0, parameter_1)

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.gelu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        gelu_0 = paddle._C_ops.gelu(batch_norm__0, False)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x32x112x112xf32, 64x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(gelu_0, parameter_6, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__1 = paddle._C_ops.add(conv2d_1, parameter_7)

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__1, parameter_8, parameter_9, parameter_10, parameter_11, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(batch_norm__6, parameter_12, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__2 = paddle._C_ops.add(depthwise_conv2d_0, parameter_13)

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add__3 = paddle._C_ops.add(batch_norm__6, add__2)

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__3, parameter_14, parameter_15, parameter_16, parameter_17, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(batch_norm__12, parameter_18, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__4 = paddle._C_ops.add(conv2d_2, parameter_19)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x1x5x5xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(add__4, parameter_20, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__5 = paddle._C_ops.add(depthwise_conv2d_1, parameter_21)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(add__5, parameter_22, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__6 = paddle._C_ops.add(conv2d_3, parameter_23)

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add__7 = paddle._C_ops.add(add__3, add__6)

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__7, parameter_24, parameter_25, parameter_26, parameter_27, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(batch_norm__18, parameter_28, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__8 = paddle._C_ops.add(conv2d_4, parameter_29)

        # pd_op.gelu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        gelu_1 = paddle._C_ops.gelu(add__8, False)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(gelu_1, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__9 = paddle._C_ops.add(conv2d_5, parameter_31)

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add__10 = paddle._C_ops.add(add__7, add__9)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(add__10, parameter_32, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__11 = paddle._C_ops.add(depthwise_conv2d_2, parameter_33)

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add__12 = paddle._C_ops.add(add__10, add__11)

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__12, parameter_34, parameter_35, parameter_36, parameter_37, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(batch_norm__24, parameter_38, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__13 = paddle._C_ops.add(conv2d_6, parameter_39)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(add__13, parameter_40, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__14 = paddle._C_ops.add(depthwise_conv2d_3, parameter_41)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(add__14, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__15 = paddle._C_ops.add(conv2d_7, parameter_43)

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add__16 = paddle._C_ops.add(add__12, add__15)

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__16, parameter_44, parameter_45, parameter_46, parameter_47, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(batch_norm__30, parameter_48, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__17 = paddle._C_ops.add(conv2d_8, parameter_49)

        # pd_op.gelu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        gelu_2 = paddle._C_ops.gelu(add__17, False)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(gelu_2, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__18 = paddle._C_ops.add(conv2d_9, parameter_51)

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add__19 = paddle._C_ops.add(add__16, add__18)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(add__19, parameter_52, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__20 = paddle._C_ops.add(depthwise_conv2d_4, parameter_53)

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add__21 = paddle._C_ops.add(add__19, add__20)

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__21, parameter_54, parameter_55, parameter_56, parameter_57, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(batch_norm__36, parameter_58, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__22 = paddle._C_ops.add(conv2d_10, parameter_59)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(add__22, parameter_60, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__23 = paddle._C_ops.add(depthwise_conv2d_5, parameter_61)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(add__23, parameter_62, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__24 = paddle._C_ops.add(conv2d_11, parameter_63)

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add__25 = paddle._C_ops.add(add__21, add__24)

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__25, parameter_64, parameter_65, parameter_66, parameter_67, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(batch_norm__42, parameter_68, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__26 = paddle._C_ops.add(conv2d_12, parameter_69)

        # pd_op.gelu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        gelu_3 = paddle._C_ops.gelu(add__26, False)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(gelu_3, parameter_70, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 1x64x1x1xf32)
        add__27 = paddle._C_ops.add(conv2d_13, parameter_71)

        # pd_op.add_: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add__28 = paddle._C_ops.add(add__25, add__27)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x56x56xf32, 128x64x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(add__28, parameter_72, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__29 = paddle._C_ops.add(conv2d_14, parameter_73)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__29, parameter_74, parameter_75, parameter_76, parameter_77, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(batch_norm__48, parameter_78, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__30 = paddle._C_ops.add(depthwise_conv2d_6, parameter_79)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__31 = paddle._C_ops.add(batch_norm__48, add__30)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__31, parameter_80, parameter_81, parameter_82, parameter_83, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(batch_norm__54, parameter_84, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__32 = paddle._C_ops.add(conv2d_15, parameter_85)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(add__32, parameter_86, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__33 = paddle._C_ops.add(depthwise_conv2d_7, parameter_87)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(add__33, parameter_88, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__34 = paddle._C_ops.add(conv2d_16, parameter_89)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__35 = paddle._C_ops.add(add__31, add__34)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__35, parameter_90, parameter_91, parameter_92, parameter_93, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(batch_norm__60, parameter_94, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__36 = paddle._C_ops.add(conv2d_17, parameter_95)

        # pd_op.gelu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        gelu_4 = paddle._C_ops.gelu(add__36, False)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(gelu_4, parameter_96, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__37 = paddle._C_ops.add(conv2d_18, parameter_97)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__38 = paddle._C_ops.add(add__35, add__37)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(add__38, parameter_98, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__39 = paddle._C_ops.add(depthwise_conv2d_8, parameter_99)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__40 = paddle._C_ops.add(add__38, add__39)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__40, parameter_100, parameter_101, parameter_102, parameter_103, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(batch_norm__66, parameter_104, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__41 = paddle._C_ops.add(conv2d_19, parameter_105)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(add__41, parameter_106, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__42 = paddle._C_ops.add(depthwise_conv2d_9, parameter_107)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(add__42, parameter_108, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__43 = paddle._C_ops.add(conv2d_20, parameter_109)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__44 = paddle._C_ops.add(add__40, add__43)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__44, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(batch_norm__72, parameter_114, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__45 = paddle._C_ops.add(conv2d_21, parameter_115)

        # pd_op.gelu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        gelu_5 = paddle._C_ops.gelu(add__45, False)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(gelu_5, parameter_116, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__46 = paddle._C_ops.add(conv2d_22, parameter_117)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__47 = paddle._C_ops.add(add__44, add__46)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x3x3xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(add__47, parameter_118, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__48 = paddle._C_ops.add(depthwise_conv2d_10, parameter_119)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__49 = paddle._C_ops.add(add__47, add__48)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__49, parameter_120, parameter_121, parameter_122, parameter_123, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(batch_norm__78, parameter_124, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__50 = paddle._C_ops.add(conv2d_23, parameter_125)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x5x5xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(add__50, parameter_126, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__51 = paddle._C_ops.add(depthwise_conv2d_11, parameter_127)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(add__51, parameter_128, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__52 = paddle._C_ops.add(conv2d_24, parameter_129)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__53 = paddle._C_ops.add(add__49, add__52)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__53, parameter_130, parameter_131, parameter_132, parameter_133, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(batch_norm__84, parameter_134, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__54 = paddle._C_ops.add(conv2d_25, parameter_135)

        # pd_op.gelu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        gelu_6 = paddle._C_ops.gelu(add__54, False)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(gelu_6, parameter_136, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__55 = paddle._C_ops.add(conv2d_26, parameter_137)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__56 = paddle._C_ops.add(add__53, add__55)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x3x3xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(add__56, parameter_138, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__57 = paddle._C_ops.add(depthwise_conv2d_12, parameter_139)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__58 = paddle._C_ops.add(add__56, add__57)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__58, parameter_140, parameter_141, parameter_142, parameter_143, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(batch_norm__90, parameter_144, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__59 = paddle._C_ops.add(conv2d_27, parameter_145)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x5x5xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(add__59, parameter_146, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__60 = paddle._C_ops.add(depthwise_conv2d_13, parameter_147)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(add__60, parameter_148, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__61 = paddle._C_ops.add(conv2d_28, parameter_149)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__62 = paddle._C_ops.add(add__58, add__61)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__62, parameter_150, parameter_151, parameter_152, parameter_153, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(batch_norm__96, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__63 = paddle._C_ops.add(conv2d_29, parameter_155)

        # pd_op.gelu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        gelu_7 = paddle._C_ops.gelu(add__63, False)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(gelu_7, parameter_156, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__64 = paddle._C_ops.add(conv2d_30, parameter_157)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__65 = paddle._C_ops.add(add__62, add__64)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(add__65, parameter_158, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__66 = paddle._C_ops.add(depthwise_conv2d_14, parameter_159)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__67 = paddle._C_ops.add(add__65, add__66)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__67, parameter_160, parameter_161, parameter_162, parameter_163, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(batch_norm__102, parameter_164, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__68 = paddle._C_ops.add(conv2d_31, parameter_165)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x1x5x5xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(add__68, parameter_166, [1, 1], [2, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__69 = paddle._C_ops.add(depthwise_conv2d_15, parameter_167)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(add__69, parameter_168, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__70 = paddle._C_ops.add(conv2d_32, parameter_169)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__71 = paddle._C_ops.add(add__67, add__70)

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__71, parameter_170, parameter_171, parameter_172, parameter_173, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(batch_norm__108, parameter_174, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__72 = paddle._C_ops.add(conv2d_33, parameter_175)

        # pd_op.gelu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        gelu_8 = paddle._C_ops.gelu(add__72, False)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(gelu_8, parameter_176, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 1x128x1x1xf32)
        add__73 = paddle._C_ops.add(conv2d_34, parameter_177)

        # pd_op.add_: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add__74 = paddle._C_ops.add(add__71, add__73)

        # pd_op.conv2d: (-1x320x14x14xf32) <- (-1x128x28x28xf32, 320x128x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(add__74, parameter_178, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__75 = paddle._C_ops.add(conv2d_35, parameter_179)

        # pd_op.batch_norm_: (-1x320x14x14xf32, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__75, parameter_180, parameter_181, parameter_182, parameter_183, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 320x1x3x3xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(batch_norm__114, parameter_184, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__76 = paddle._C_ops.add(depthwise_conv2d_16, parameter_185)

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, -1x320x14x14xf32)
        add__77 = paddle._C_ops.add(batch_norm__114, add__76)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf32)
        shape_0 = paddle._C_ops.shape(add__77)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf32, None) <- (-1x320x14x14xf32)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__77, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf32) <- (-1x320x196xf32)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_0, parameter_186, parameter_187, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf32)
        shape_1 = paddle._C_ops.shape(layer_norm_0)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf32) <- (-1x196x320xf32, 320x960xf32)
        matmul_0 = paddle.matmul(layer_norm_0, parameter_188, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf32) <- (-1x196x960xf32, 960xf32)
        add__78 = paddle._C_ops.add(matmul_0, parameter_189)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_1, constant_2, constant_3, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x3x5x64xf32, 0x-1x196x960xf32) <- (-1x196x960xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__78, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf32) <- (-1x196x3x5x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_1, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_1, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(transpose_1, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf32) <- (-1x5x196x64xf32)
        transpose_2 = paddle._C_ops.transpose(slice_3, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf32) <- (-1x5x196x64xf32, -1x5x64x196xf32)
        matmul_1 = paddle.matmul(slice_2, transpose_2, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf32) <- (-1x5x196x196xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(matmul_1, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf32) <- (-1x5x196x196xf32)
        softmax__0 = paddle._C_ops.softmax(scale__0, -1)

        # pd_op.matmul: (-1x5x196x64xf32) <- (-1x5x196x196xf32, -1x5x196x64xf32)
        matmul_2 = paddle.matmul(softmax__0, slice_4, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf32) <- (-1x5x196x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x320xf32, 0x-1x196x5x64xf32) <- (-1x196x5x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_3, combine_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x320xf32, 320x320xf32)
        matmul_3 = paddle.matmul(reshape__2, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__79 = paddle._C_ops.add(matmul_3, parameter_191)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__80 = paddle._C_ops.add(transpose_0, add__79)

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__80, parameter_192, parameter_193, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf32) <- (-1x196x320xf32, 320x1280xf32)
        matmul_4 = paddle.matmul(layer_norm_3, parameter_194, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf32) <- (-1x196x1280xf32, 1280xf32)
        add__81 = paddle._C_ops.add(matmul_4, parameter_195)

        # pd_op.gelu: (-1x196x1280xf32) <- (-1x196x1280xf32)
        gelu_9 = paddle._C_ops.gelu(add__81, False)

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x1280xf32, 1280x320xf32)
        matmul_5 = paddle.matmul(gelu_9, parameter_196, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__82 = paddle._C_ops.add(matmul_5, parameter_197)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__83 = paddle._C_ops.add(add__80, add__82)

        # pd_op.transpose: (-1x320x196xf32) <- (-1x196x320xf32)
        transpose_4 = paddle._C_ops.transpose(add__83, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_0, constant_9, constant_10, constant_10]

        # pd_op.reshape_: (-1x320x14x14xf32, 0x-1x320x196xf32) <- (-1x320x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_4, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 320x1x3x3xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(reshape__4, parameter_198, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__84 = paddle._C_ops.add(depthwise_conv2d_17, parameter_199)

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, -1x320x14x14xf32)
        add__85 = paddle._C_ops.add(reshape__4, add__84)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf32)
        shape_2 = paddle._C_ops.shape(add__85)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_2, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf32, None) <- (-1x320x14x14xf32)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__85, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf32) <- (-1x320x196xf32)
        transpose_5 = paddle._C_ops.transpose(flatten__2, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_5, parameter_200, parameter_201, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf32)
        shape_3 = paddle._C_ops.shape(layer_norm_6)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_3, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf32) <- (-1x196x320xf32, 320x960xf32)
        matmul_6 = paddle.matmul(layer_norm_6, parameter_202, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf32) <- (-1x196x960xf32, 960xf32)
        add__86 = paddle._C_ops.add(matmul_6, parameter_203)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_6, constant_2, constant_3, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x3x5x64xf32, 0x-1x196x960xf32) <- (-1x196x960xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__86, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf32) <- (-1x196x3x5x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape__6, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_6, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(transpose_6, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_6, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf32) <- (-1x5x196x64xf32)
        transpose_7 = paddle._C_ops.transpose(slice_8, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf32) <- (-1x5x196x64xf32, -1x5x64x196xf32)
        matmul_7 = paddle.matmul(slice_7, transpose_7, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf32) <- (-1x5x196x196xf32, 1xf32)
        scale__1 = paddle._C_ops.scale(matmul_7, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf32) <- (-1x5x196x196xf32)
        softmax__1 = paddle._C_ops.softmax(scale__1, -1)

        # pd_op.matmul: (-1x5x196x64xf32) <- (-1x5x196x196xf32, -1x5x196x64xf32)
        matmul_8 = paddle.matmul(softmax__1, slice_9, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf32) <- (-1x5x196x64xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_6, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x320xf32, 0x-1x196x5x64xf32) <- (-1x196x5x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_8, combine_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x320xf32, 320x320xf32)
        matmul_9 = paddle.matmul(reshape__8, parameter_204, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__87 = paddle._C_ops.add(matmul_9, parameter_205)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__88 = paddle._C_ops.add(transpose_5, add__87)

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__88, parameter_206, parameter_207, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf32) <- (-1x196x320xf32, 320x1280xf32)
        matmul_10 = paddle.matmul(layer_norm_9, parameter_208, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf32) <- (-1x196x1280xf32, 1280xf32)
        add__89 = paddle._C_ops.add(matmul_10, parameter_209)

        # pd_op.gelu: (-1x196x1280xf32) <- (-1x196x1280xf32)
        gelu_10 = paddle._C_ops.gelu(add__89, False)

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x1280xf32, 1280x320xf32)
        matmul_11 = paddle.matmul(gelu_10, parameter_210, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__90 = paddle._C_ops.add(matmul_11, parameter_211)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__91 = paddle._C_ops.add(add__88, add__90)

        # pd_op.transpose: (-1x320x196xf32) <- (-1x196x320xf32)
        transpose_9 = paddle._C_ops.transpose(add__91, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_5, constant_9, constant_10, constant_10]

        # pd_op.reshape_: (-1x320x14x14xf32, 0x-1x320x196xf32) <- (-1x320x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_9, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 320x1x3x3xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(reshape__10, parameter_212, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__92 = paddle._C_ops.add(depthwise_conv2d_18, parameter_213)

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, -1x320x14x14xf32)
        add__93 = paddle._C_ops.add(reshape__10, add__92)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf32)
        shape_4 = paddle._C_ops.shape(add__93)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(shape_4, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf32, None) <- (-1x320x14x14xf32)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__93, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf32) <- (-1x320x196xf32)
        transpose_10 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_10, parameter_214, parameter_215, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf32)
        shape_5 = paddle._C_ops.shape(layer_norm_12)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_5, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf32) <- (-1x196x320xf32, 320x960xf32)
        matmul_12 = paddle.matmul(layer_norm_12, parameter_216, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf32) <- (-1x196x960xf32, 960xf32)
        add__94 = paddle._C_ops.add(matmul_12, parameter_217)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_11, constant_2, constant_3, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x3x5x64xf32, 0x-1x196x960xf32) <- (-1x196x960xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__94, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf32) <- (-1x196x3x5x64xf32)
        transpose_11 = paddle._C_ops.transpose(reshape__12, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(transpose_11, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(transpose_11, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(transpose_11, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf32) <- (-1x5x196x64xf32)
        transpose_12 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf32) <- (-1x5x196x64xf32, -1x5x64x196xf32)
        matmul_13 = paddle.matmul(slice_12, transpose_12, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf32) <- (-1x5x196x196xf32, 1xf32)
        scale__2 = paddle._C_ops.scale(matmul_13, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf32) <- (-1x5x196x196xf32)
        softmax__2 = paddle._C_ops.softmax(scale__2, -1)

        # pd_op.matmul: (-1x5x196x64xf32) <- (-1x5x196x196xf32, -1x5x196x64xf32)
        matmul_14 = paddle.matmul(softmax__2, slice_14, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf32) <- (-1x5x196x64xf32)
        transpose_13 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_7 = [slice_11, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x320xf32, 0x-1x196x5x64xf32) <- (-1x196x5x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_13, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x320xf32, 320x320xf32)
        matmul_15 = paddle.matmul(reshape__14, parameter_218, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__95 = paddle._C_ops.add(matmul_15, parameter_219)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__96 = paddle._C_ops.add(transpose_10, add__95)

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__96, parameter_220, parameter_221, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf32) <- (-1x196x320xf32, 320x1280xf32)
        matmul_16 = paddle.matmul(layer_norm_15, parameter_222, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf32) <- (-1x196x1280xf32, 1280xf32)
        add__97 = paddle._C_ops.add(matmul_16, parameter_223)

        # pd_op.gelu: (-1x196x1280xf32) <- (-1x196x1280xf32)
        gelu_11 = paddle._C_ops.gelu(add__97, False)

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x1280xf32, 1280x320xf32)
        matmul_17 = paddle.matmul(gelu_11, parameter_224, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__98 = paddle._C_ops.add(matmul_17, parameter_225)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__99 = paddle._C_ops.add(add__96, add__98)

        # pd_op.transpose: (-1x320x196xf32) <- (-1x196x320xf32)
        transpose_14 = paddle._C_ops.transpose(add__99, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_10, constant_9, constant_10, constant_10]

        # pd_op.reshape_: (-1x320x14x14xf32, 0x-1x320x196xf32) <- (-1x320x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_14, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 320x1x3x3xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(reshape__16, parameter_226, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__100 = paddle._C_ops.add(depthwise_conv2d_19, parameter_227)

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, -1x320x14x14xf32)
        add__101 = paddle._C_ops.add(reshape__16, add__100)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf32)
        shape_6 = paddle._C_ops.shape(add__101)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_6, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf32, None) <- (-1x320x14x14xf32)
        flatten__6, flatten__7 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__101, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf32) <- (-1x320x196xf32)
        transpose_15 = paddle._C_ops.transpose(flatten__6, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_15, parameter_228, parameter_229, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf32)
        shape_7 = paddle._C_ops.shape(layer_norm_18)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_7, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf32) <- (-1x196x320xf32, 320x960xf32)
        matmul_18 = paddle.matmul(layer_norm_18, parameter_230, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf32) <- (-1x196x960xf32, 960xf32)
        add__102 = paddle._C_ops.add(matmul_18, parameter_231)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_16, constant_2, constant_3, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x3x5x64xf32, 0x-1x196x960xf32) <- (-1x196x960xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__102, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf32) <- (-1x196x3x5x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape__18, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_16, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_16, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_16, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf32) <- (-1x5x196x64xf32)
        transpose_17 = paddle._C_ops.transpose(slice_18, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf32) <- (-1x5x196x64xf32, -1x5x64x196xf32)
        matmul_19 = paddle.matmul(slice_17, transpose_17, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf32) <- (-1x5x196x196xf32, 1xf32)
        scale__3 = paddle._C_ops.scale(matmul_19, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf32) <- (-1x5x196x196xf32)
        softmax__3 = paddle._C_ops.softmax(scale__3, -1)

        # pd_op.matmul: (-1x5x196x64xf32) <- (-1x5x196x196xf32, -1x5x196x64xf32)
        matmul_20 = paddle.matmul(softmax__3, slice_19, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf32) <- (-1x5x196x64xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_16, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x320xf32, 0x-1x196x5x64xf32) <- (-1x196x5x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_18, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x320xf32, 320x320xf32)
        matmul_21 = paddle.matmul(reshape__20, parameter_232, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__103 = paddle._C_ops.add(matmul_21, parameter_233)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__104 = paddle._C_ops.add(transpose_15, add__103)

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__104, parameter_234, parameter_235, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf32) <- (-1x196x320xf32, 320x1280xf32)
        matmul_22 = paddle.matmul(layer_norm_21, parameter_236, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf32) <- (-1x196x1280xf32, 1280xf32)
        add__105 = paddle._C_ops.add(matmul_22, parameter_237)

        # pd_op.gelu: (-1x196x1280xf32) <- (-1x196x1280xf32)
        gelu_12 = paddle._C_ops.gelu(add__105, False)

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x1280xf32, 1280x320xf32)
        matmul_23 = paddle.matmul(gelu_12, parameter_238, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__106 = paddle._C_ops.add(matmul_23, parameter_239)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__107 = paddle._C_ops.add(add__104, add__106)

        # pd_op.transpose: (-1x320x196xf32) <- (-1x196x320xf32)
        transpose_19 = paddle._C_ops.transpose(add__107, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_15, constant_9, constant_10, constant_10]

        # pd_op.reshape_: (-1x320x14x14xf32, 0x-1x320x196xf32) <- (-1x320x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_19, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 320x1x3x3xf32)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(reshape__22, parameter_240, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__108 = paddle._C_ops.add(depthwise_conv2d_20, parameter_241)

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, -1x320x14x14xf32)
        add__109 = paddle._C_ops.add(reshape__22, add__108)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf32)
        shape_8 = paddle._C_ops.shape(add__109)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_8, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf32, None) <- (-1x320x14x14xf32)
        flatten__8, flatten__9 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__109, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf32) <- (-1x320x196xf32)
        transpose_20 = paddle._C_ops.transpose(flatten__8, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_20, parameter_242, parameter_243, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf32)
        shape_9 = paddle._C_ops.shape(layer_norm_24)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(shape_9, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf32) <- (-1x196x320xf32, 320x960xf32)
        matmul_24 = paddle.matmul(layer_norm_24, parameter_244, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf32) <- (-1x196x960xf32, 960xf32)
        add__110 = paddle._C_ops.add(matmul_24, parameter_245)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_21, constant_2, constant_3, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x3x5x64xf32, 0x-1x196x960xf32) <- (-1x196x960xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__110, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf32) <- (-1x196x3x5x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape__24, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_21, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_21, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_21, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf32) <- (-1x5x196x64xf32)
        transpose_22 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf32) <- (-1x5x196x64xf32, -1x5x64x196xf32)
        matmul_25 = paddle.matmul(slice_22, transpose_22, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf32) <- (-1x5x196x196xf32, 1xf32)
        scale__4 = paddle._C_ops.scale(matmul_25, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf32) <- (-1x5x196x196xf32)
        softmax__4 = paddle._C_ops.softmax(scale__4, -1)

        # pd_op.matmul: (-1x5x196x64xf32) <- (-1x5x196x196xf32, -1x5x196x64xf32)
        matmul_26 = paddle.matmul(softmax__4, slice_24, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf32) <- (-1x5x196x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_13 = [slice_21, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x320xf32, 0x-1x196x5x64xf32) <- (-1x196x5x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_23, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x320xf32, 320x320xf32)
        matmul_27 = paddle.matmul(reshape__26, parameter_246, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__111 = paddle._C_ops.add(matmul_27, parameter_247)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__112 = paddle._C_ops.add(transpose_20, add__111)

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__112, parameter_248, parameter_249, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf32) <- (-1x196x320xf32, 320x1280xf32)
        matmul_28 = paddle.matmul(layer_norm_27, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf32) <- (-1x196x1280xf32, 1280xf32)
        add__113 = paddle._C_ops.add(matmul_28, parameter_251)

        # pd_op.gelu: (-1x196x1280xf32) <- (-1x196x1280xf32)
        gelu_13 = paddle._C_ops.gelu(add__113, False)

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x1280xf32, 1280x320xf32)
        matmul_29 = paddle.matmul(gelu_13, parameter_252, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__114 = paddle._C_ops.add(matmul_29, parameter_253)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__115 = paddle._C_ops.add(add__112, add__114)

        # pd_op.transpose: (-1x320x196xf32) <- (-1x196x320xf32)
        transpose_24 = paddle._C_ops.transpose(add__115, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_20, constant_9, constant_10, constant_10]

        # pd_op.reshape_: (-1x320x14x14xf32, 0x-1x320x196xf32) <- (-1x320x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_24, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 320x1x3x3xf32)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(reshape__28, parameter_254, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__116 = paddle._C_ops.add(depthwise_conv2d_21, parameter_255)

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, -1x320x14x14xf32)
        add__117 = paddle._C_ops.add(reshape__28, add__116)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf32)
        shape_10 = paddle._C_ops.shape(add__117)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_10, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf32, None) <- (-1x320x14x14xf32)
        flatten__10, flatten__11 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__117, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf32) <- (-1x320x196xf32)
        transpose_25 = paddle._C_ops.transpose(flatten__10, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_25, parameter_256, parameter_257, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf32)
        shape_11 = paddle._C_ops.shape(layer_norm_30)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(shape_11, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf32) <- (-1x196x320xf32, 320x960xf32)
        matmul_30 = paddle.matmul(layer_norm_30, parameter_258, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf32) <- (-1x196x960xf32, 960xf32)
        add__118 = paddle._C_ops.add(matmul_30, parameter_259)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_15 = [slice_26, constant_2, constant_3, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x3x5x64xf32, 0x-1x196x960xf32) <- (-1x196x960xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__118, combine_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf32) <- (-1x196x3x5x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape__30, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_26, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(transpose_26, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_26, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf32) <- (-1x5x196x64xf32)
        transpose_27 = paddle._C_ops.transpose(slice_28, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf32) <- (-1x5x196x64xf32, -1x5x64x196xf32)
        matmul_31 = paddle.matmul(slice_27, transpose_27, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf32) <- (-1x5x196x196xf32, 1xf32)
        scale__5 = paddle._C_ops.scale(matmul_31, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf32) <- (-1x5x196x196xf32)
        softmax__5 = paddle._C_ops.softmax(scale__5, -1)

        # pd_op.matmul: (-1x5x196x64xf32) <- (-1x5x196x196xf32, -1x5x196x64xf32)
        matmul_32 = paddle.matmul(softmax__5, slice_29, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf32) <- (-1x5x196x64xf32)
        transpose_28 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_16 = [slice_26, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x320xf32, 0x-1x196x5x64xf32) <- (-1x196x5x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_28, combine_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x320xf32, 320x320xf32)
        matmul_33 = paddle.matmul(reshape__32, parameter_260, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__119 = paddle._C_ops.add(matmul_33, parameter_261)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__120 = paddle._C_ops.add(transpose_25, add__119)

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__120, parameter_262, parameter_263, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf32) <- (-1x196x320xf32, 320x1280xf32)
        matmul_34 = paddle.matmul(layer_norm_33, parameter_264, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf32) <- (-1x196x1280xf32, 1280xf32)
        add__121 = paddle._C_ops.add(matmul_34, parameter_265)

        # pd_op.gelu: (-1x196x1280xf32) <- (-1x196x1280xf32)
        gelu_14 = paddle._C_ops.gelu(add__121, False)

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x1280xf32, 1280x320xf32)
        matmul_35 = paddle.matmul(gelu_14, parameter_266, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__122 = paddle._C_ops.add(matmul_35, parameter_267)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__123 = paddle._C_ops.add(add__120, add__122)

        # pd_op.transpose: (-1x320x196xf32) <- (-1x196x320xf32)
        transpose_29 = paddle._C_ops.transpose(add__123, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_25, constant_9, constant_10, constant_10]

        # pd_op.reshape_: (-1x320x14x14xf32, 0x-1x320x196xf32) <- (-1x320x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_29, combine_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 320x1x3x3xf32)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(reshape__34, parameter_268, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__124 = paddle._C_ops.add(depthwise_conv2d_22, parameter_269)

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, -1x320x14x14xf32)
        add__125 = paddle._C_ops.add(reshape__34, add__124)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf32)
        shape_12 = paddle._C_ops.shape(add__125)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(shape_12, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf32, None) <- (-1x320x14x14xf32)
        flatten__12, flatten__13 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__125, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf32) <- (-1x320x196xf32)
        transpose_30 = paddle._C_ops.transpose(flatten__12, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_30, parameter_270, parameter_271, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf32)
        shape_13 = paddle._C_ops.shape(layer_norm_36)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(shape_13, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf32) <- (-1x196x320xf32, 320x960xf32)
        matmul_36 = paddle.matmul(layer_norm_36, parameter_272, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf32) <- (-1x196x960xf32, 960xf32)
        add__126 = paddle._C_ops.add(matmul_36, parameter_273)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_31, constant_2, constant_3, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x3x5x64xf32, 0x-1x196x960xf32) <- (-1x196x960xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__126, combine_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf32) <- (-1x196x3x5x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape__36, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(transpose_31, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_31, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_31, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf32) <- (-1x5x196x64xf32)
        transpose_32 = paddle._C_ops.transpose(slice_33, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf32) <- (-1x5x196x64xf32, -1x5x64x196xf32)
        matmul_37 = paddle.matmul(slice_32, transpose_32, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf32) <- (-1x5x196x196xf32, 1xf32)
        scale__6 = paddle._C_ops.scale(matmul_37, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf32) <- (-1x5x196x196xf32)
        softmax__6 = paddle._C_ops.softmax(scale__6, -1)

        # pd_op.matmul: (-1x5x196x64xf32) <- (-1x5x196x196xf32, -1x5x196x64xf32)
        matmul_38 = paddle.matmul(softmax__6, slice_34, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf32) <- (-1x5x196x64xf32)
        transpose_33 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_19 = [slice_31, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x320xf32, 0x-1x196x5x64xf32) <- (-1x196x5x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_33, combine_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x320xf32, 320x320xf32)
        matmul_39 = paddle.matmul(reshape__38, parameter_274, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__127 = paddle._C_ops.add(matmul_39, parameter_275)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__128 = paddle._C_ops.add(transpose_30, add__127)

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__128, parameter_276, parameter_277, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf32) <- (-1x196x320xf32, 320x1280xf32)
        matmul_40 = paddle.matmul(layer_norm_39, parameter_278, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf32) <- (-1x196x1280xf32, 1280xf32)
        add__129 = paddle._C_ops.add(matmul_40, parameter_279)

        # pd_op.gelu: (-1x196x1280xf32) <- (-1x196x1280xf32)
        gelu_15 = paddle._C_ops.gelu(add__129, False)

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x1280xf32, 1280x320xf32)
        matmul_41 = paddle.matmul(gelu_15, parameter_280, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__130 = paddle._C_ops.add(matmul_41, parameter_281)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__131 = paddle._C_ops.add(add__128, add__130)

        # pd_op.transpose: (-1x320x196xf32) <- (-1x196x320xf32)
        transpose_34 = paddle._C_ops.transpose(add__131, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_30, constant_9, constant_10, constant_10]

        # pd_op.reshape_: (-1x320x14x14xf32, 0x-1x320x196xf32) <- (-1x320x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_34, combine_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 320x1x3x3xf32)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(reshape__40, parameter_282, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__132 = paddle._C_ops.add(depthwise_conv2d_23, parameter_283)

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, -1x320x14x14xf32)
        add__133 = paddle._C_ops.add(reshape__40, add__132)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf32)
        shape_14 = paddle._C_ops.shape(add__133)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(shape_14, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf32, None) <- (-1x320x14x14xf32)
        flatten__14, flatten__15 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__133, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf32) <- (-1x320x196xf32)
        transpose_35 = paddle._C_ops.transpose(flatten__14, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_35, parameter_284, parameter_285, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf32)
        shape_15 = paddle._C_ops.shape(layer_norm_42)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(shape_15, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf32) <- (-1x196x320xf32, 320x960xf32)
        matmul_42 = paddle.matmul(layer_norm_42, parameter_286, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf32) <- (-1x196x960xf32, 960xf32)
        add__134 = paddle._C_ops.add(matmul_42, parameter_287)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_36, constant_2, constant_3, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x3x5x64xf32, 0x-1x196x960xf32) <- (-1x196x960xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__134, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf32) <- (-1x196x3x5x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape__42, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(transpose_36, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(transpose_36, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(transpose_36, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf32) <- (-1x5x196x64xf32)
        transpose_37 = paddle._C_ops.transpose(slice_38, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf32) <- (-1x5x196x64xf32, -1x5x64x196xf32)
        matmul_43 = paddle.matmul(slice_37, transpose_37, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf32) <- (-1x5x196x196xf32, 1xf32)
        scale__7 = paddle._C_ops.scale(matmul_43, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf32) <- (-1x5x196x196xf32)
        softmax__7 = paddle._C_ops.softmax(scale__7, -1)

        # pd_op.matmul: (-1x5x196x64xf32) <- (-1x5x196x196xf32, -1x5x196x64xf32)
        matmul_44 = paddle.matmul(softmax__7, slice_39, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf32) <- (-1x5x196x64xf32)
        transpose_38 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_22 = [slice_36, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x320xf32, 0x-1x196x5x64xf32) <- (-1x196x5x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_38, combine_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x320xf32, 320x320xf32)
        matmul_45 = paddle.matmul(reshape__44, parameter_288, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__135 = paddle._C_ops.add(matmul_45, parameter_289)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__136 = paddle._C_ops.add(transpose_35, add__135)

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__136, parameter_290, parameter_291, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf32) <- (-1x196x320xf32, 320x1280xf32)
        matmul_46 = paddle.matmul(layer_norm_45, parameter_292, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf32) <- (-1x196x1280xf32, 1280xf32)
        add__137 = paddle._C_ops.add(matmul_46, parameter_293)

        # pd_op.gelu: (-1x196x1280xf32) <- (-1x196x1280xf32)
        gelu_16 = paddle._C_ops.gelu(add__137, False)

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x1280xf32, 1280x320xf32)
        matmul_47 = paddle.matmul(gelu_16, parameter_294, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__138 = paddle._C_ops.add(matmul_47, parameter_295)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__139 = paddle._C_ops.add(add__136, add__138)

        # pd_op.transpose: (-1x320x196xf32) <- (-1x196x320xf32)
        transpose_39 = paddle._C_ops.transpose(add__139, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_35, constant_9, constant_10, constant_10]

        # pd_op.reshape_: (-1x320x14x14xf32, 0x-1x320x196xf32) <- (-1x320x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_39, combine_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 320x1x3x3xf32)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(reshape__46, parameter_296, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, 1x320x1x1xf32)
        add__140 = paddle._C_ops.add(depthwise_conv2d_24, parameter_297)

        # pd_op.add_: (-1x320x14x14xf32) <- (-1x320x14x14xf32, -1x320x14x14xf32)
        add__141 = paddle._C_ops.add(reshape__46, add__140)

        # pd_op.shape: (4xi32) <- (-1x320x14x14xf32)
        shape_16 = paddle._C_ops.shape(add__141)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(shape_16, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x320x196xf32, None) <- (-1x320x14x14xf32)
        flatten__16, flatten__17 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__141, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf32) <- (-1x320x196xf32)
        transpose_40 = paddle._C_ops.transpose(flatten__16, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_40, parameter_298, parameter_299, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf32)
        shape_17 = paddle._C_ops.shape(layer_norm_48)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(shape_17, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x960xf32) <- (-1x196x320xf32, 320x960xf32)
        matmul_48 = paddle.matmul(layer_norm_48, parameter_300, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x960xf32) <- (-1x196x960xf32, 960xf32)
        add__142 = paddle._C_ops.add(matmul_48, parameter_301)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_41, constant_2, constant_3, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x3x5x64xf32, 0x-1x196x960xf32) <- (-1x196x960xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__142, combine_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x5x196x64xf32) <- (-1x196x3x5x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape__48, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(transpose_41, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(transpose_41, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x5x196x64xf32) <- (3x-1x5x196x64xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(transpose_41, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x5x64x196xf32) <- (-1x5x196x64xf32)
        transpose_42 = paddle._C_ops.transpose(slice_43, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x196xf32) <- (-1x5x196x64xf32, -1x5x64x196xf32)
        matmul_49 = paddle.matmul(slice_42, transpose_42, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x196xf32) <- (-1x5x196x196xf32, 1xf32)
        scale__8 = paddle._C_ops.scale(matmul_49, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x5x196x196xf32) <- (-1x5x196x196xf32)
        softmax__8 = paddle._C_ops.softmax(scale__8, -1)

        # pd_op.matmul: (-1x5x196x64xf32) <- (-1x5x196x196xf32, -1x5x196x64xf32)
        matmul_50 = paddle.matmul(softmax__8, slice_44, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x64xf32) <- (-1x5x196x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_50, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_41, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x320xf32, 0x-1x196x5x64xf32) <- (-1x196x5x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_43, combine_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x320xf32, 320x320xf32)
        matmul_51 = paddle.matmul(reshape__50, parameter_302, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__143 = paddle._C_ops.add(matmul_51, parameter_303)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__144 = paddle._C_ops.add(transpose_40, add__143)

        # pd_op.layer_norm: (-1x196x320xf32, -196xf32, -196xf32) <- (-1x196x320xf32, 320xf32, 320xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__144, parameter_304, parameter_305, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf32) <- (-1x196x320xf32, 320x1280xf32)
        matmul_52 = paddle.matmul(layer_norm_51, parameter_306, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x1280xf32) <- (-1x196x1280xf32, 1280xf32)
        add__145 = paddle._C_ops.add(matmul_52, parameter_307)

        # pd_op.gelu: (-1x196x1280xf32) <- (-1x196x1280xf32)
        gelu_17 = paddle._C_ops.gelu(add__145, False)

        # pd_op.matmul: (-1x196x320xf32) <- (-1x196x1280xf32, 1280x320xf32)
        matmul_53 = paddle.matmul(gelu_17, parameter_308, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, 320xf32)
        add__146 = paddle._C_ops.add(matmul_53, parameter_309)

        # pd_op.add_: (-1x196x320xf32) <- (-1x196x320xf32, -1x196x320xf32)
        add__147 = paddle._C_ops.add(add__144, add__146)

        # pd_op.transpose: (-1x320x196xf32) <- (-1x196x320xf32)
        transpose_44 = paddle._C_ops.transpose(add__147, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_40, constant_9, constant_10, constant_10]

        # pd_op.reshape_: (-1x320x14x14xf32, 0x-1x320x196xf32) <- (-1x320x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_44, combine_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x320x14x14xf32, 512x320x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(reshape__52, parameter_310, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 1x512x1x1xf32)
        add__148 = paddle._C_ops.add(conv2d_36, parameter_311)

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__148, parameter_312, parameter_313, parameter_314, parameter_315, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x3x3xf32)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(batch_norm__120, parameter_316, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 1x512x1x1xf32)
        add__149 = paddle._C_ops.add(depthwise_conv2d_25, parameter_317)

        # pd_op.add_: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        add__150 = paddle._C_ops.add(batch_norm__120, add__149)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf32)
        shape_18 = paddle._C_ops.shape(add__150)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(shape_18, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf32, None) <- (-1x512x7x7xf32)
        flatten__18, flatten__19 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__150, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf32) <- (-1x512x49xf32)
        transpose_45 = paddle._C_ops.transpose(flatten__18, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf32, -49xf32, -49xf32) <- (-1x49x512xf32, 512xf32, 512xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_45, parameter_318, parameter_319, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf32)
        shape_19 = paddle._C_ops.shape(layer_norm_54)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(shape_19, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf32) <- (-1x49x512xf32, 512x1536xf32)
        matmul_54 = paddle.matmul(layer_norm_54, parameter_320, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf32) <- (-1x49x1536xf32, 1536xf32)
        add__151 = paddle._C_ops.add(matmul_54, parameter_321)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_27 = [slice_46, constant_11, constant_3, constant_12, constant_5]

        # pd_op.reshape_: (-1x49x3x8x64xf32, 0x-1x49x1536xf32) <- (-1x49x1536xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__151, combine_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf32) <- (-1x49x3x8x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape__54, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf32) <- (3x-1x8x49x64xf32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(transpose_46, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf32) <- (3x-1x8x49x64xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(transpose_46, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x8x49x64xf32) <- (3x-1x8x49x64xf32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(transpose_46, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf32) <- (-1x8x49x64xf32)
        transpose_47 = paddle._C_ops.transpose(slice_48, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf32) <- (-1x8x49x64xf32, -1x8x64x49xf32)
        matmul_55 = paddle.matmul(slice_47, transpose_47, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf32) <- (-1x8x49x49xf32, 1xf32)
        scale__9 = paddle._C_ops.scale(matmul_55, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf32) <- (-1x8x49x49xf32)
        softmax__9 = paddle._C_ops.softmax(scale__9, -1)

        # pd_op.matmul: (-1x8x49x64xf32) <- (-1x8x49x49xf32, -1x8x49x64xf32)
        matmul_56 = paddle.matmul(softmax__9, slice_49, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf32) <- (-1x8x49x64xf32)
        transpose_48 = paddle._C_ops.transpose(matmul_56, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_28 = [slice_46, constant_11, constant_13]

        # pd_op.reshape_: (-1x49x512xf32, 0x-1x49x8x64xf32) <- (-1x49x8x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_48, combine_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf32) <- (-1x49x512xf32, 512x512xf32)
        matmul_57 = paddle.matmul(reshape__56, parameter_322, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, 512xf32)
        add__152 = paddle._C_ops.add(matmul_57, parameter_323)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, -1x49x512xf32)
        add__153 = paddle._C_ops.add(transpose_45, add__152)

        # pd_op.layer_norm: (-1x49x512xf32, -49xf32, -49xf32) <- (-1x49x512xf32, 512xf32, 512xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__153, parameter_324, parameter_325, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf32) <- (-1x49x512xf32, 512x2048xf32)
        matmul_58 = paddle.matmul(layer_norm_57, parameter_326, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf32) <- (-1x49x2048xf32, 2048xf32)
        add__154 = paddle._C_ops.add(matmul_58, parameter_327)

        # pd_op.gelu: (-1x49x2048xf32) <- (-1x49x2048xf32)
        gelu_18 = paddle._C_ops.gelu(add__154, False)

        # pd_op.matmul: (-1x49x512xf32) <- (-1x49x2048xf32, 2048x512xf32)
        matmul_59 = paddle.matmul(gelu_18, parameter_328, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, 512xf32)
        add__155 = paddle._C_ops.add(matmul_59, parameter_329)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, -1x49x512xf32)
        add__156 = paddle._C_ops.add(add__153, add__155)

        # pd_op.transpose: (-1x512x49xf32) <- (-1x49x512xf32)
        transpose_49 = paddle._C_ops.transpose(add__156, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_45, constant_13, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf32, 0x-1x512x49xf32) <- (-1x512x49xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_49, combine_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x3x3xf32)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(reshape__58, parameter_330, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 1x512x1x1xf32)
        add__157 = paddle._C_ops.add(depthwise_conv2d_26, parameter_331)

        # pd_op.add_: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        add__158 = paddle._C_ops.add(reshape__58, add__157)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf32)
        shape_20 = paddle._C_ops.shape(add__158)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(shape_20, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf32, None) <- (-1x512x7x7xf32)
        flatten__20, flatten__21 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__158, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf32) <- (-1x512x49xf32)
        transpose_50 = paddle._C_ops.transpose(flatten__20, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf32, -49xf32, -49xf32) <- (-1x49x512xf32, 512xf32, 512xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_50, parameter_332, parameter_333, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf32)
        shape_21 = paddle._C_ops.shape(layer_norm_60)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(shape_21, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf32) <- (-1x49x512xf32, 512x1536xf32)
        matmul_60 = paddle.matmul(layer_norm_60, parameter_334, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf32) <- (-1x49x1536xf32, 1536xf32)
        add__159 = paddle._C_ops.add(matmul_60, parameter_335)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_30 = [slice_51, constant_11, constant_3, constant_12, constant_5]

        # pd_op.reshape_: (-1x49x3x8x64xf32, 0x-1x49x1536xf32) <- (-1x49x1536xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__159, combine_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf32) <- (-1x49x3x8x64xf32)
        transpose_51 = paddle._C_ops.transpose(reshape__60, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf32) <- (3x-1x8x49x64xf32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(transpose_51, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf32) <- (3x-1x8x49x64xf32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(transpose_51, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x8x49x64xf32) <- (3x-1x8x49x64xf32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(transpose_51, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf32) <- (-1x8x49x64xf32)
        transpose_52 = paddle._C_ops.transpose(slice_53, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf32) <- (-1x8x49x64xf32, -1x8x64x49xf32)
        matmul_61 = paddle.matmul(slice_52, transpose_52, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf32) <- (-1x8x49x49xf32, 1xf32)
        scale__10 = paddle._C_ops.scale(matmul_61, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf32) <- (-1x8x49x49xf32)
        softmax__10 = paddle._C_ops.softmax(scale__10, -1)

        # pd_op.matmul: (-1x8x49x64xf32) <- (-1x8x49x49xf32, -1x8x49x64xf32)
        matmul_62 = paddle.matmul(softmax__10, slice_54, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf32) <- (-1x8x49x64xf32)
        transpose_53 = paddle._C_ops.transpose(matmul_62, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_31 = [slice_51, constant_11, constant_13]

        # pd_op.reshape_: (-1x49x512xf32, 0x-1x49x8x64xf32) <- (-1x49x8x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_53, combine_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf32) <- (-1x49x512xf32, 512x512xf32)
        matmul_63 = paddle.matmul(reshape__62, parameter_336, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, 512xf32)
        add__160 = paddle._C_ops.add(matmul_63, parameter_337)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, -1x49x512xf32)
        add__161 = paddle._C_ops.add(transpose_50, add__160)

        # pd_op.layer_norm: (-1x49x512xf32, -49xf32, -49xf32) <- (-1x49x512xf32, 512xf32, 512xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__161, parameter_338, parameter_339, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf32) <- (-1x49x512xf32, 512x2048xf32)
        matmul_64 = paddle.matmul(layer_norm_63, parameter_340, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf32) <- (-1x49x2048xf32, 2048xf32)
        add__162 = paddle._C_ops.add(matmul_64, parameter_341)

        # pd_op.gelu: (-1x49x2048xf32) <- (-1x49x2048xf32)
        gelu_19 = paddle._C_ops.gelu(add__162, False)

        # pd_op.matmul: (-1x49x512xf32) <- (-1x49x2048xf32, 2048x512xf32)
        matmul_65 = paddle.matmul(gelu_19, parameter_342, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, 512xf32)
        add__163 = paddle._C_ops.add(matmul_65, parameter_343)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, -1x49x512xf32)
        add__164 = paddle._C_ops.add(add__161, add__163)

        # pd_op.transpose: (-1x512x49xf32) <- (-1x49x512xf32)
        transpose_54 = paddle._C_ops.transpose(add__164, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_50, constant_13, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf32, 0x-1x512x49xf32) <- (-1x512x49xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_54, combine_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x1x3x3xf32)
        depthwise_conv2d_27 = paddle._C_ops.depthwise_conv2d(reshape__64, parameter_344, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 1x512x1x1xf32)
        add__165 = paddle._C_ops.add(depthwise_conv2d_27, parameter_345)

        # pd_op.add_: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        add__166 = paddle._C_ops.add(reshape__64, add__165)

        # pd_op.shape: (4xi32) <- (-1x512x7x7xf32)
        shape_22 = paddle._C_ops.shape(add__166)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(shape_22, [0], constant_0, constant_1, [1], [0])

        # pd_op.flatten_: (-1x512x49xf32, None) <- (-1x512x7x7xf32)
        flatten__22, flatten__23 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__166, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf32) <- (-1x512x49xf32)
        transpose_55 = paddle._C_ops.transpose(flatten__22, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf32, -49xf32, -49xf32) <- (-1x49x512xf32, 512xf32, 512xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_55, parameter_346, parameter_347, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf32)
        shape_23 = paddle._C_ops.shape(layer_norm_66)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(shape_23, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x1536xf32) <- (-1x49x512xf32, 512x1536xf32)
        matmul_66 = paddle.matmul(layer_norm_66, parameter_348, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1536xf32) <- (-1x49x1536xf32, 1536xf32)
        add__167 = paddle._C_ops.add(matmul_66, parameter_349)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_56, constant_11, constant_3, constant_12, constant_5]

        # pd_op.reshape_: (-1x49x3x8x64xf32, 0x-1x49x1536xf32) <- (-1x49x1536xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__167, combine_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x49x64xf32) <- (-1x49x3x8x64xf32)
        transpose_56 = paddle._C_ops.transpose(reshape__66, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x64xf32) <- (3x-1x8x49x64xf32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(transpose_56, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x64xf32) <- (3x-1x8x49x64xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(transpose_56, [0], constant_1, constant_6, [1], [0])

        # pd_op.slice: (-1x8x49x64xf32) <- (3x-1x8x49x64xf32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(transpose_56, [0], constant_6, constant_7, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf32) <- (-1x8x49x64xf32)
        transpose_57 = paddle._C_ops.transpose(slice_58, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf32) <- (-1x8x49x64xf32, -1x8x64x49xf32)
        matmul_67 = paddle.matmul(slice_57, transpose_57, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf32) <- (-1x8x49x49xf32, 1xf32)
        scale__11 = paddle._C_ops.scale(matmul_67, constant_8, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf32) <- (-1x8x49x49xf32)
        softmax__11 = paddle._C_ops.softmax(scale__11, -1)

        # pd_op.matmul: (-1x8x49x64xf32) <- (-1x8x49x49xf32, -1x8x49x64xf32)
        matmul_68 = paddle.matmul(softmax__11, slice_59, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x64xf32) <- (-1x8x49x64xf32)
        transpose_58 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_34 = [slice_56, constant_11, constant_13]

        # pd_op.reshape_: (-1x49x512xf32, 0x-1x49x8x64xf32) <- (-1x49x8x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_58, combine_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf32) <- (-1x49x512xf32, 512x512xf32)
        matmul_69 = paddle.matmul(reshape__68, parameter_350, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, 512xf32)
        add__168 = paddle._C_ops.add(matmul_69, parameter_351)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, -1x49x512xf32)
        add__169 = paddle._C_ops.add(transpose_55, add__168)

        # pd_op.layer_norm: (-1x49x512xf32, -49xf32, -49xf32) <- (-1x49x512xf32, 512xf32, 512xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__169, parameter_352, parameter_353, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf32) <- (-1x49x512xf32, 512x2048xf32)
        matmul_70 = paddle.matmul(layer_norm_69, parameter_354, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x2048xf32) <- (-1x49x2048xf32, 2048xf32)
        add__170 = paddle._C_ops.add(matmul_70, parameter_355)

        # pd_op.gelu: (-1x49x2048xf32) <- (-1x49x2048xf32)
        gelu_20 = paddle._C_ops.gelu(add__170, False)

        # pd_op.matmul: (-1x49x512xf32) <- (-1x49x2048xf32, 2048x512xf32)
        matmul_71 = paddle.matmul(gelu_20, parameter_356, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, 512xf32)
        add__171 = paddle._C_ops.add(matmul_71, parameter_357)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, -1x49x512xf32)
        add__172 = paddle._C_ops.add(add__169, add__171)

        # pd_op.transpose: (-1x512x49xf32) <- (-1x49x512xf32)
        transpose_59 = paddle._C_ops.transpose(add__172, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_55, constant_13, constant_14, constant_14]

        # pd_op.reshape_: (-1x512x7x7xf32, 0x-1x512x49xf32) <- (-1x512x49xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_59, combine_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__70, parameter_358, parameter_359, parameter_360, parameter_361, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.flatten_: (-1x512x49xf32, None) <- (-1x512x7x7xf32)
        flatten__24, flatten__25 = (lambda x, f: f(x))(paddle._C_ops.flatten(batch_norm__126, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.mean: (-1x512xf32) <- (-1x512x49xf32)
        mean_0 = paddle._C_ops.mean(flatten__24, [-1], False)

        # pd_op.matmul: (-1x1000xf32) <- (-1x512xf32, 512x1000xf32)
        matmul_72 = paddle.matmul(mean_0, parameter_362, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__173 = paddle._C_ops.add(matmul_72, parameter_363)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__12 = paddle._C_ops.softmax(add__173, -1)
        return softmax__12



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

    def forward(self, parameter_345, parameter_331, constant_14, constant_13, constant_12, constant_11, parameter_317, parameter_311, parameter_297, parameter_283, parameter_269, parameter_255, parameter_241, parameter_227, parameter_213, parameter_199, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_185, parameter_179, parameter_177, parameter_175, parameter_169, parameter_167, parameter_165, parameter_159, parameter_157, parameter_155, parameter_149, parameter_147, parameter_145, parameter_139, parameter_137, parameter_135, parameter_129, parameter_127, parameter_125, parameter_119, parameter_117, parameter_115, parameter_109, parameter_107, parameter_105, parameter_99, parameter_97, parameter_95, parameter_89, parameter_87, parameter_85, parameter_79, parameter_73, parameter_71, parameter_69, parameter_63, parameter_61, parameter_59, parameter_53, parameter_51, parameter_49, parameter_43, parameter_41, parameter_39, parameter_33, parameter_31, parameter_29, parameter_23, parameter_21, parameter_19, parameter_13, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_17, parameter_14, parameter_16, parameter_15, parameter_18, parameter_20, parameter_22, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_30, parameter_32, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_40, parameter_42, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_50, parameter_52, parameter_57, parameter_54, parameter_56, parameter_55, parameter_58, parameter_60, parameter_62, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_70, parameter_72, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_86, parameter_88, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_96, parameter_98, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_106, parameter_108, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_116, parameter_118, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_126, parameter_128, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_136, parameter_138, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_146, parameter_148, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_156, parameter_158, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_166, parameter_168, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_176, parameter_178, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_187, parameter_186, parameter_188, parameter_189, parameter_190, parameter_191, parameter_193, parameter_192, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_201, parameter_200, parameter_202, parameter_203, parameter_204, parameter_205, parameter_207, parameter_206, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_215, parameter_214, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_229, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_235, parameter_234, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_243, parameter_242, parameter_244, parameter_245, parameter_246, parameter_247, parameter_249, parameter_248, parameter_250, parameter_251, parameter_252, parameter_253, parameter_254, parameter_257, parameter_256, parameter_258, parameter_259, parameter_260, parameter_261, parameter_263, parameter_262, parameter_264, parameter_265, parameter_266, parameter_267, parameter_268, parameter_271, parameter_270, parameter_272, parameter_273, parameter_274, parameter_275, parameter_277, parameter_276, parameter_278, parameter_279, parameter_280, parameter_281, parameter_282, parameter_285, parameter_284, parameter_286, parameter_287, parameter_288, parameter_289, parameter_291, parameter_290, parameter_292, parameter_293, parameter_294, parameter_295, parameter_296, parameter_299, parameter_298, parameter_300, parameter_301, parameter_302, parameter_303, parameter_305, parameter_304, parameter_306, parameter_307, parameter_308, parameter_309, parameter_310, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_330, parameter_333, parameter_332, parameter_334, parameter_335, parameter_336, parameter_337, parameter_339, parameter_338, parameter_340, parameter_341, parameter_342, parameter_343, parameter_344, parameter_347, parameter_346, parameter_348, parameter_349, parameter_350, parameter_351, parameter_353, parameter_352, parameter_354, parameter_355, parameter_356, parameter_357, parameter_361, parameter_358, parameter_360, parameter_359, parameter_362, parameter_363, feed_0):
        return self.builtin_module_2432_0_0(parameter_345, parameter_331, constant_14, constant_13, constant_12, constant_11, parameter_317, parameter_311, parameter_297, parameter_283, parameter_269, parameter_255, parameter_241, parameter_227, parameter_213, parameter_199, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_185, parameter_179, parameter_177, parameter_175, parameter_169, parameter_167, parameter_165, parameter_159, parameter_157, parameter_155, parameter_149, parameter_147, parameter_145, parameter_139, parameter_137, parameter_135, parameter_129, parameter_127, parameter_125, parameter_119, parameter_117, parameter_115, parameter_109, parameter_107, parameter_105, parameter_99, parameter_97, parameter_95, parameter_89, parameter_87, parameter_85, parameter_79, parameter_73, parameter_71, parameter_69, parameter_63, parameter_61, parameter_59, parameter_53, parameter_51, parameter_49, parameter_43, parameter_41, parameter_39, parameter_33, parameter_31, parameter_29, parameter_23, parameter_21, parameter_19, parameter_13, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_17, parameter_14, parameter_16, parameter_15, parameter_18, parameter_20, parameter_22, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_30, parameter_32, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_40, parameter_42, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_50, parameter_52, parameter_57, parameter_54, parameter_56, parameter_55, parameter_58, parameter_60, parameter_62, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_70, parameter_72, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_86, parameter_88, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_96, parameter_98, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_106, parameter_108, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_116, parameter_118, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_126, parameter_128, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_136, parameter_138, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_146, parameter_148, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_156, parameter_158, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_166, parameter_168, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_176, parameter_178, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_187, parameter_186, parameter_188, parameter_189, parameter_190, parameter_191, parameter_193, parameter_192, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_201, parameter_200, parameter_202, parameter_203, parameter_204, parameter_205, parameter_207, parameter_206, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_215, parameter_214, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_229, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_235, parameter_234, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_243, parameter_242, parameter_244, parameter_245, parameter_246, parameter_247, parameter_249, parameter_248, parameter_250, parameter_251, parameter_252, parameter_253, parameter_254, parameter_257, parameter_256, parameter_258, parameter_259, parameter_260, parameter_261, parameter_263, parameter_262, parameter_264, parameter_265, parameter_266, parameter_267, parameter_268, parameter_271, parameter_270, parameter_272, parameter_273, parameter_274, parameter_275, parameter_277, parameter_276, parameter_278, parameter_279, parameter_280, parameter_281, parameter_282, parameter_285, parameter_284, parameter_286, parameter_287, parameter_288, parameter_289, parameter_291, parameter_290, parameter_292, parameter_293, parameter_294, parameter_295, parameter_296, parameter_299, parameter_298, parameter_300, parameter_301, parameter_302, parameter_303, parameter_305, parameter_304, parameter_306, parameter_307, parameter_308, parameter_309, parameter_310, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_330, parameter_333, parameter_332, parameter_334, parameter_335, parameter_336, parameter_337, parameter_339, parameter_338, parameter_340, parameter_341, parameter_342, parameter_343, parameter_344, parameter_347, parameter_346, parameter_348, parameter_349, parameter_350, parameter_351, parameter_353, parameter_352, parameter_354, parameter_355, parameter_356, parameter_357, parameter_361, parameter_358, parameter_360, parameter_359, parameter_362, parameter_363, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2432_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_345
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_14
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            # constant_13
            paddle.to_tensor([512], dtype='int32').reshape([1]),
            # constant_12
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            # constant_11
            paddle.to_tensor([49], dtype='int32').reshape([1]),
            # parameter_317
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_10
            paddle.to_tensor([14], dtype='int32').reshape([1]),
            # constant_9
            paddle.to_tensor([320], dtype='int32').reshape([1]),
            # constant_8
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_7
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            # constant_6
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_5
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([5], dtype='int32').reshape([1]),
            # constant_3
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            # constant_2
            paddle.to_tensor([196], dtype='int32').reshape([1]),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_185
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([32, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([64, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([64, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([128, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([128, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([128, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([128, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([320, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([320, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([320, 960], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([320, 1280], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([1280, 320], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([320, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([320, 960], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([320, 1280], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([1280, 320], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([320, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([320, 960], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([320, 1280], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1280, 320], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([320, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([320, 960], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([320, 1280], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([1280, 320], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([320, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([320, 960], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([320, 1280], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([1280, 320], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([320, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([320, 960], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([320, 1280], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([1280, 320], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([320, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([320, 960], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([320, 1280], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([1280, 320], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([320, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([320, 960], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([320, 1280], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([1280, 320], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([320, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([320, 960], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([320, 1280], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([1280, 320], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([512, 320, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([512, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([512, 2048], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([2048, 512], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([512, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([512, 2048], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([2048, 512], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([512, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([512, 2048], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([2048, 512], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([512, 1000], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_345
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_317
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_185
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[128, 1, 5, 5], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[320, 128, 3, 3], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[320, 960], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[960], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[320, 1280], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[1280, 320], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[320, 960], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[960], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[320, 1280], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[1280, 320], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[320, 960], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[960], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[320, 1280], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[1280, 320], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[320, 960], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[960], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[320, 1280], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[1280, 320], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[320, 960], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[960], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[320, 1280], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[1280, 320], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[320, 960], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[960], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[320, 1280], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[1280, 320], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[320, 960], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[960], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[320, 1280], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[1280, 320], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[320, 960], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[960], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[320, 1280], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[1280, 320], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[320, 960], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[960], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[320, 1280], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[1280, 320], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[512, 320, 3, 3], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[512, 2048], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[2048, 512], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[512, 2048], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[2048, 512], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[512, 2048], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[2048, 512], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[512, 1000], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
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