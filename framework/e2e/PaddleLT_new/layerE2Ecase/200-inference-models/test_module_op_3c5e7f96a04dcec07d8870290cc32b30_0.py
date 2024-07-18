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
    return [288][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1451_0_0(self, constant_15, parameter_350, constant_14, constant_13, parameter_338, parameter_332, constant_12, parameter_315, parameter_309, constant_11, parameter_292, parameter_286, constant_10, parameter_269, parameter_263, parameter_246, parameter_240, constant_9, parameter_223, parameter_217, constant_8, parameter_200, parameter_194, constant_7, parameter_177, parameter_171, constant_6, parameter_154, parameter_148, constant_5, parameter_131, parameter_125, parameter_108, parameter_102, constant_4, parameter_85, parameter_79, parameter_62, parameter_56, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_60, parameter_57, parameter_59, parameter_58, parameter_61, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_152, parameter_149, parameter_151, parameter_150, parameter_153, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_175, parameter_172, parameter_174, parameter_173, parameter_176, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_224, parameter_228, parameter_225, parameter_227, parameter_226, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_267, parameter_264, parameter_266, parameter_265, parameter_268, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_302, parameter_299, parameter_301, parameter_300, parameter_303, parameter_307, parameter_304, parameter_306, parameter_305, parameter_308, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_336, parameter_333, parameter_335, parameter_334, parameter_337, parameter_339, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_348, parameter_345, parameter_347, parameter_346, parameter_349, feed_0):

        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.depthwise_conv2d: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 32x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(swish_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        relu6_0 = paddle._C_ops.relu6(batch_norm__6)

        # pd_op.conv2d: (-1x16x112x112xf32) <- (-1x32x112x112xf32, 16x32x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu6_0, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x96x112x112xf32) <- (-1x16x112x112xf32, 96x16x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(batch_norm__12, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x112x112xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x112x112xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x112x112xf32) <- (-1x96x112x112xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__18)

        # pd_op.depthwise_conv2d: (-1x96x56x56xf32) <- (-1x96x112x112xf32, 96x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(swish_1, parameter_20, [2, 2], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu6_1 = paddle._C_ops.relu6(batch_norm__24)

        # pd_op.conv2d: (-1x27x56x56xf32) <- (-1x96x56x56xf32, 27x96x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu6_1, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x27x56x56xf32, 27xf32, 27xf32, xf32, xf32, None) <- (-1x27x56x56xf32, 27xf32, 27xf32, 27xf32, 27xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x162x56x56xf32) <- (-1x27x56x56xf32, 162x27x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(batch_norm__30, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x162x56x56xf32, 162xf32, 162xf32, xf32, xf32, None) <- (-1x162x56x56xf32, 162xf32, 162xf32, 162xf32, 162xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x162x56x56xf32) <- (-1x162x56x56xf32)
        swish_2 = paddle._C_ops.swish(batch_norm__36)

        # pd_op.depthwise_conv2d: (-1x162x56x56xf32) <- (-1x162x56x56xf32, 162x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(swish_2, parameter_35, [1, 1], [1, 1], 'EXPLICIT', 162, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x162x56x56xf32, 162xf32, 162xf32, xf32, xf32, None) <- (-1x162x56x56xf32, 162xf32, 162xf32, 162xf32, 162xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x162x56x56xf32) <- (-1x162x56x56xf32)
        relu6_2 = paddle._C_ops.relu6(batch_norm__42)

        # pd_op.conv2d: (-1x38x56x56xf32) <- (-1x162x56x56xf32, 38x162x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu6_2, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x38x56x56xf32, 38xf32, 38xf32, xf32, xf32, None) <- (-1x38x56x56xf32, 38xf32, 38xf32, 38xf32, 38xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x27x56x56xf32) <- (-1x38x56x56xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(batch_norm__48, [1], constant_0, constant_1, [1], [])

        # pd_op.add_: (-1x27x56x56xf32) <- (-1x27x56x56xf32, -1x27x56x56xf32)
        add__0 = paddle._C_ops.add(slice_0, batch_norm__30)

        # pd_op.set_value_with_tensor_: (-1x38x56x56xf32) <- (-1x38x56x56xf32, -1x27x56x56xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor(batch_norm__48, add__0, constant_0, constant_1, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x228x56x56xf32) <- (-1x38x56x56xf32, 228x38x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(set_value_with_tensor__0, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x228x56x56xf32, 228xf32, 228xf32, xf32, xf32, None) <- (-1x228x56x56xf32, 228xf32, 228xf32, 228xf32, 228xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x228x56x56xf32) <- (-1x228x56x56xf32)
        swish_3 = paddle._C_ops.swish(batch_norm__54)

        # pd_op.depthwise_conv2d: (-1x228x28x28xf32) <- (-1x228x56x56xf32, 228x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(swish_3, parameter_50, [2, 2], [1, 1], 'EXPLICIT', 228, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x228x28x28xf32, 228xf32, 228xf32, xf32, xf32, None) <- (-1x228x28x28xf32, 228xf32, 228xf32, 228xf32, 228xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x228x1x1xf32) <- (-1x228x28x28xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(batch_norm__60, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x19x1x1xf32) <- (-1x228x1x1xf32, 19x228x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(pool2d_0, parameter_55, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x19x1x1xf32) <- (-1x19x1x1xf32, 1x19x1x1xf32)
        add__1 = paddle._C_ops.add(conv2d_7, parameter_56)

        # pd_op.batch_norm_: (-1x19x1x1xf32, 19xf32, 19xf32, xf32, xf32, None) <- (-1x19x1x1xf32, 19xf32, 19xf32, 19xf32, 19xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__1, parameter_57, parameter_58, parameter_59, parameter_60, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x19x1x1xf32) <- (-1x19x1x1xf32)
        relu__0 = paddle._C_ops.relu(batch_norm__66)

        # pd_op.conv2d: (-1x228x1x1xf32) <- (-1x19x1x1xf32, 228x19x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu__0, parameter_61, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x228x1x1xf32) <- (-1x228x1x1xf32, 1x228x1x1xf32)
        add__2 = paddle._C_ops.add(conv2d_8, parameter_62)

        # pd_op.sigmoid_: (-1x228x1x1xf32) <- (-1x228x1x1xf32)
        sigmoid__0 = paddle._C_ops.sigmoid(add__2)

        # pd_op.multiply_: (-1x228x28x28xf32) <- (-1x228x28x28xf32, -1x228x1x1xf32)
        multiply__0 = paddle._C_ops.multiply(batch_norm__60, sigmoid__0)

        # pd_op.relu6: (-1x228x28x28xf32) <- (-1x228x28x28xf32)
        relu6_3 = paddle._C_ops.relu6(multiply__0)

        # pd_op.conv2d: (-1x50x28x28xf32) <- (-1x228x28x28xf32, 50x228x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu6_3, parameter_63, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x50x28x28xf32, 50xf32, 50xf32, xf32, xf32, None) <- (-1x50x28x28xf32, 50xf32, 50xf32, 50xf32, 50xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_64, parameter_65, parameter_66, parameter_67, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x300x28x28xf32) <- (-1x50x28x28xf32, 300x50x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(batch_norm__72, parameter_68, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x300x28x28xf32, 300xf32, 300xf32, xf32, xf32, None) <- (-1x300x28x28xf32, 300xf32, 300xf32, 300xf32, 300xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_69, parameter_70, parameter_71, parameter_72, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x300x28x28xf32) <- (-1x300x28x28xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__78)

        # pd_op.depthwise_conv2d: (-1x300x28x28xf32) <- (-1x300x28x28xf32, 300x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(swish_4, parameter_73, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x300x28x28xf32, 300xf32, 300xf32, xf32, xf32, None) <- (-1x300x28x28xf32, 300xf32, 300xf32, 300xf32, 300xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_74, parameter_75, parameter_76, parameter_77, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x300x1x1xf32) <- (-1x300x28x28xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(batch_norm__84, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x25x1x1xf32) <- (-1x300x1x1xf32, 25x300x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(pool2d_1, parameter_78, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x25x1x1xf32) <- (-1x25x1x1xf32, 1x25x1x1xf32)
        add__3 = paddle._C_ops.add(conv2d_11, parameter_79)

        # pd_op.batch_norm_: (-1x25x1x1xf32, 25xf32, 25xf32, xf32, xf32, None) <- (-1x25x1x1xf32, 25xf32, 25xf32, 25xf32, 25xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__3, parameter_80, parameter_81, parameter_82, parameter_83, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x25x1x1xf32) <- (-1x25x1x1xf32)
        relu__1 = paddle._C_ops.relu(batch_norm__90)

        # pd_op.conv2d: (-1x300x1x1xf32) <- (-1x25x1x1xf32, 300x25x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu__1, parameter_84, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x300x1x1xf32) <- (-1x300x1x1xf32, 1x300x1x1xf32)
        add__4 = paddle._C_ops.add(conv2d_12, parameter_85)

        # pd_op.sigmoid_: (-1x300x1x1xf32) <- (-1x300x1x1xf32)
        sigmoid__1 = paddle._C_ops.sigmoid(add__4)

        # pd_op.multiply_: (-1x300x28x28xf32) <- (-1x300x28x28xf32, -1x300x1x1xf32)
        multiply__1 = paddle._C_ops.multiply(batch_norm__84, sigmoid__1)

        # pd_op.relu6: (-1x300x28x28xf32) <- (-1x300x28x28xf32)
        relu6_4 = paddle._C_ops.relu6(multiply__1)

        # pd_op.conv2d: (-1x61x28x28xf32) <- (-1x300x28x28xf32, 61x300x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu6_4, parameter_86, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x61x28x28xf32, 61xf32, 61xf32, xf32, xf32, None) <- (-1x61x28x28xf32, 61xf32, 61xf32, 61xf32, 61xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_87, parameter_88, parameter_89, parameter_90, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x50x28x28xf32) <- (-1x61x28x28xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(batch_norm__96, [1], constant_0, constant_4, [1], [])

        # pd_op.add_: (-1x50x28x28xf32) <- (-1x50x28x28xf32, -1x50x28x28xf32)
        add__5 = paddle._C_ops.add(slice_1, batch_norm__72)

        # pd_op.set_value_with_tensor_: (-1x61x28x28xf32) <- (-1x61x28x28xf32, -1x50x28x28xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor(batch_norm__96, add__5, constant_0, constant_4, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x366x28x28xf32) <- (-1x61x28x28xf32, 366x61x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(set_value_with_tensor__1, parameter_91, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x366x28x28xf32, 366xf32, 366xf32, xf32, xf32, None) <- (-1x366x28x28xf32, 366xf32, 366xf32, 366xf32, 366xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x366x28x28xf32) <- (-1x366x28x28xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__102)

        # pd_op.depthwise_conv2d: (-1x366x14x14xf32) <- (-1x366x28x28xf32, 366x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(swish_5, parameter_96, [2, 2], [1, 1], 'EXPLICIT', 366, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x366x14x14xf32, 366xf32, 366xf32, xf32, xf32, None) <- (-1x366x14x14xf32, 366xf32, 366xf32, 366xf32, 366xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_97, parameter_98, parameter_99, parameter_100, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x366x1x1xf32) <- (-1x366x14x14xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(batch_norm__108, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x30x1x1xf32) <- (-1x366x1x1xf32, 30x366x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(pool2d_2, parameter_101, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x30x1x1xf32) <- (-1x30x1x1xf32, 1x30x1x1xf32)
        add__6 = paddle._C_ops.add(conv2d_15, parameter_102)

        # pd_op.batch_norm_: (-1x30x1x1xf32, 30xf32, 30xf32, xf32, xf32, None) <- (-1x30x1x1xf32, 30xf32, 30xf32, 30xf32, 30xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__6, parameter_103, parameter_104, parameter_105, parameter_106, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x30x1x1xf32) <- (-1x30x1x1xf32)
        relu__2 = paddle._C_ops.relu(batch_norm__114)

        # pd_op.conv2d: (-1x366x1x1xf32) <- (-1x30x1x1xf32, 366x30x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu__2, parameter_107, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x366x1x1xf32) <- (-1x366x1x1xf32, 1x366x1x1xf32)
        add__7 = paddle._C_ops.add(conv2d_16, parameter_108)

        # pd_op.sigmoid_: (-1x366x1x1xf32) <- (-1x366x1x1xf32)
        sigmoid__2 = paddle._C_ops.sigmoid(add__7)

        # pd_op.multiply_: (-1x366x14x14xf32) <- (-1x366x14x14xf32, -1x366x1x1xf32)
        multiply__2 = paddle._C_ops.multiply(batch_norm__108, sigmoid__2)

        # pd_op.relu6: (-1x366x14x14xf32) <- (-1x366x14x14xf32)
        relu6_5 = paddle._C_ops.relu6(multiply__2)

        # pd_op.conv2d: (-1x72x14x14xf32) <- (-1x366x14x14xf32, 72x366x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu6_5, parameter_109, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x14x14xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x14x14xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x432x14x14xf32) <- (-1x72x14x14xf32, 432x72x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(batch_norm__120, parameter_114, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x432x14x14xf32, 432xf32, 432xf32, xf32, xf32, None) <- (-1x432x14x14xf32, 432xf32, 432xf32, 432xf32, 432xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_115, parameter_116, parameter_117, parameter_118, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x432x14x14xf32) <- (-1x432x14x14xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__126)

        # pd_op.depthwise_conv2d: (-1x432x14x14xf32) <- (-1x432x14x14xf32, 432x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(swish_6, parameter_119, [1, 1], [1, 1], 'EXPLICIT', 432, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x432x14x14xf32, 432xf32, 432xf32, xf32, xf32, None) <- (-1x432x14x14xf32, 432xf32, 432xf32, 432xf32, 432xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_120, parameter_121, parameter_122, parameter_123, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x432x1x1xf32) <- (-1x432x14x14xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(batch_norm__132, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x36x1x1xf32) <- (-1x432x1x1xf32, 36x432x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(pool2d_3, parameter_124, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x36x1x1xf32) <- (-1x36x1x1xf32, 1x36x1x1xf32)
        add__8 = paddle._C_ops.add(conv2d_19, parameter_125)

        # pd_op.batch_norm_: (-1x36x1x1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x1x1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__8, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x36x1x1xf32) <- (-1x36x1x1xf32)
        relu__3 = paddle._C_ops.relu(batch_norm__138)

        # pd_op.conv2d: (-1x432x1x1xf32) <- (-1x36x1x1xf32, 432x36x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu__3, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x432x1x1xf32) <- (-1x432x1x1xf32, 1x432x1x1xf32)
        add__9 = paddle._C_ops.add(conv2d_20, parameter_131)

        # pd_op.sigmoid_: (-1x432x1x1xf32) <- (-1x432x1x1xf32)
        sigmoid__3 = paddle._C_ops.sigmoid(add__9)

        # pd_op.multiply_: (-1x432x14x14xf32) <- (-1x432x14x14xf32, -1x432x1x1xf32)
        multiply__3 = paddle._C_ops.multiply(batch_norm__132, sigmoid__3)

        # pd_op.relu6: (-1x432x14x14xf32) <- (-1x432x14x14xf32)
        relu6_6 = paddle._C_ops.relu6(multiply__3)

        # pd_op.conv2d: (-1x84x14x14xf32) <- (-1x432x14x14xf32, 84x432x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu6_6, parameter_132, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x84x14x14xf32, 84xf32, 84xf32, xf32, xf32, None) <- (-1x84x14x14xf32, 84xf32, 84xf32, 84xf32, 84xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_133, parameter_134, parameter_135, parameter_136, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x72x14x14xf32) <- (-1x84x14x14xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(batch_norm__144, [1], constant_0, constant_5, [1], [])

        # pd_op.add_: (-1x72x14x14xf32) <- (-1x72x14x14xf32, -1x72x14x14xf32)
        add__10 = paddle._C_ops.add(slice_2, batch_norm__120)

        # pd_op.set_value_with_tensor_: (-1x84x14x14xf32) <- (-1x84x14x14xf32, -1x72x14x14xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor(batch_norm__144, add__10, constant_0, constant_5, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x504x14x14xf32) <- (-1x84x14x14xf32, 504x84x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(set_value_with_tensor__2, parameter_137, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x504x14x14xf32, 504xf32, 504xf32, xf32, xf32, None) <- (-1x504x14x14xf32, 504xf32, 504xf32, 504xf32, 504xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_138, parameter_139, parameter_140, parameter_141, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x504x14x14xf32) <- (-1x504x14x14xf32)
        swish_7 = paddle._C_ops.swish(batch_norm__150)

        # pd_op.depthwise_conv2d: (-1x504x14x14xf32) <- (-1x504x14x14xf32, 504x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(swish_7, parameter_142, [1, 1], [1, 1], 'EXPLICIT', 504, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x504x14x14xf32, 504xf32, 504xf32, xf32, xf32, None) <- (-1x504x14x14xf32, 504xf32, 504xf32, 504xf32, 504xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_7, parameter_143, parameter_144, parameter_145, parameter_146, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x504x1x1xf32) <- (-1x504x14x14xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(batch_norm__156, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x42x1x1xf32) <- (-1x504x1x1xf32, 42x504x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(pool2d_4, parameter_147, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x42x1x1xf32) <- (-1x42x1x1xf32, 1x42x1x1xf32)
        add__11 = paddle._C_ops.add(conv2d_23, parameter_148)

        # pd_op.batch_norm_: (-1x42x1x1xf32, 42xf32, 42xf32, xf32, xf32, None) <- (-1x42x1x1xf32, 42xf32, 42xf32, 42xf32, 42xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__11, parameter_149, parameter_150, parameter_151, parameter_152, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x42x1x1xf32) <- (-1x42x1x1xf32)
        relu__4 = paddle._C_ops.relu(batch_norm__162)

        # pd_op.conv2d: (-1x504x1x1xf32) <- (-1x42x1x1xf32, 504x42x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__4, parameter_153, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x504x1x1xf32) <- (-1x504x1x1xf32, 1x504x1x1xf32)
        add__12 = paddle._C_ops.add(conv2d_24, parameter_154)

        # pd_op.sigmoid_: (-1x504x1x1xf32) <- (-1x504x1x1xf32)
        sigmoid__4 = paddle._C_ops.sigmoid(add__12)

        # pd_op.multiply_: (-1x504x14x14xf32) <- (-1x504x14x14xf32, -1x504x1x1xf32)
        multiply__4 = paddle._C_ops.multiply(batch_norm__156, sigmoid__4)

        # pd_op.relu6: (-1x504x14x14xf32) <- (-1x504x14x14xf32)
        relu6_7 = paddle._C_ops.relu6(multiply__4)

        # pd_op.conv2d: (-1x95x14x14xf32) <- (-1x504x14x14xf32, 95x504x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu6_7, parameter_155, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x95x14x14xf32, 95xf32, 95xf32, xf32, xf32, None) <- (-1x95x14x14xf32, 95xf32, 95xf32, 95xf32, 95xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x84x14x14xf32) <- (-1x95x14x14xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(batch_norm__168, [1], constant_0, constant_6, [1], [])

        # pd_op.add_: (-1x84x14x14xf32) <- (-1x84x14x14xf32, -1x84x14x14xf32)
        add__13 = paddle._C_ops.add(slice_3, set_value_with_tensor__2)

        # pd_op.set_value_with_tensor_: (-1x95x14x14xf32) <- (-1x95x14x14xf32, -1x84x14x14xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__3 = paddle._C_ops.set_value_with_tensor(batch_norm__168, add__13, constant_0, constant_6, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x570x14x14xf32) <- (-1x95x14x14xf32, 570x95x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(set_value_with_tensor__3, parameter_160, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x570x14x14xf32, 570xf32, 570xf32, xf32, xf32, None) <- (-1x570x14x14xf32, 570xf32, 570xf32, 570xf32, 570xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x570x14x14xf32) <- (-1x570x14x14xf32)
        swish_8 = paddle._C_ops.swish(batch_norm__174)

        # pd_op.depthwise_conv2d: (-1x570x14x14xf32) <- (-1x570x14x14xf32, 570x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(swish_8, parameter_165, [1, 1], [1, 1], 'EXPLICIT', 570, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x570x14x14xf32, 570xf32, 570xf32, xf32, xf32, None) <- (-1x570x14x14xf32, 570xf32, 570xf32, 570xf32, 570xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_8, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x570x1x1xf32) <- (-1x570x14x14xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(batch_norm__180, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x47x1x1xf32) <- (-1x570x1x1xf32, 47x570x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(pool2d_5, parameter_170, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x47x1x1xf32) <- (-1x47x1x1xf32, 1x47x1x1xf32)
        add__14 = paddle._C_ops.add(conv2d_27, parameter_171)

        # pd_op.batch_norm_: (-1x47x1x1xf32, 47xf32, 47xf32, xf32, xf32, None) <- (-1x47x1x1xf32, 47xf32, 47xf32, 47xf32, 47xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__14, parameter_172, parameter_173, parameter_174, parameter_175, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x47x1x1xf32) <- (-1x47x1x1xf32)
        relu__5 = paddle._C_ops.relu(batch_norm__186)

        # pd_op.conv2d: (-1x570x1x1xf32) <- (-1x47x1x1xf32, 570x47x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__5, parameter_176, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x570x1x1xf32) <- (-1x570x1x1xf32, 1x570x1x1xf32)
        add__15 = paddle._C_ops.add(conv2d_28, parameter_177)

        # pd_op.sigmoid_: (-1x570x1x1xf32) <- (-1x570x1x1xf32)
        sigmoid__5 = paddle._C_ops.sigmoid(add__15)

        # pd_op.multiply_: (-1x570x14x14xf32) <- (-1x570x14x14xf32, -1x570x1x1xf32)
        multiply__5 = paddle._C_ops.multiply(batch_norm__180, sigmoid__5)

        # pd_op.relu6: (-1x570x14x14xf32) <- (-1x570x14x14xf32)
        relu6_8 = paddle._C_ops.relu6(multiply__5)

        # pd_op.conv2d: (-1x106x14x14xf32) <- (-1x570x14x14xf32, 106x570x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu6_8, parameter_178, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x106x14x14xf32, 106xf32, 106xf32, xf32, xf32, None) <- (-1x106x14x14xf32, 106xf32, 106xf32, 106xf32, 106xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_179, parameter_180, parameter_181, parameter_182, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x95x14x14xf32) <- (-1x106x14x14xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(batch_norm__192, [1], constant_0, constant_7, [1], [])

        # pd_op.add_: (-1x95x14x14xf32) <- (-1x95x14x14xf32, -1x95x14x14xf32)
        add__16 = paddle._C_ops.add(slice_4, set_value_with_tensor__3)

        # pd_op.set_value_with_tensor_: (-1x106x14x14xf32) <- (-1x106x14x14xf32, -1x95x14x14xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__4 = paddle._C_ops.set_value_with_tensor(batch_norm__192, add__16, constant_0, constant_7, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x636x14x14xf32) <- (-1x106x14x14xf32, 636x106x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(set_value_with_tensor__4, parameter_183, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x636x14x14xf32, 636xf32, 636xf32, xf32, xf32, None) <- (-1x636x14x14xf32, 636xf32, 636xf32, 636xf32, 636xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_184, parameter_185, parameter_186, parameter_187, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x636x14x14xf32) <- (-1x636x14x14xf32)
        swish_9 = paddle._C_ops.swish(batch_norm__198)

        # pd_op.depthwise_conv2d: (-1x636x14x14xf32) <- (-1x636x14x14xf32, 636x1x3x3xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(swish_9, parameter_188, [1, 1], [1, 1], 'EXPLICIT', 636, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x636x14x14xf32, 636xf32, 636xf32, xf32, xf32, None) <- (-1x636x14x14xf32, 636xf32, 636xf32, 636xf32, 636xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_9, parameter_189, parameter_190, parameter_191, parameter_192, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x636x1x1xf32) <- (-1x636x14x14xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(batch_norm__204, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x53x1x1xf32) <- (-1x636x1x1xf32, 53x636x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(pool2d_6, parameter_193, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x53x1x1xf32) <- (-1x53x1x1xf32, 1x53x1x1xf32)
        add__17 = paddle._C_ops.add(conv2d_31, parameter_194)

        # pd_op.batch_norm_: (-1x53x1x1xf32, 53xf32, 53xf32, xf32, xf32, None) <- (-1x53x1x1xf32, 53xf32, 53xf32, 53xf32, 53xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__17, parameter_195, parameter_196, parameter_197, parameter_198, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x53x1x1xf32) <- (-1x53x1x1xf32)
        relu__6 = paddle._C_ops.relu(batch_norm__210)

        # pd_op.conv2d: (-1x636x1x1xf32) <- (-1x53x1x1xf32, 636x53x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu__6, parameter_199, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x636x1x1xf32) <- (-1x636x1x1xf32, 1x636x1x1xf32)
        add__18 = paddle._C_ops.add(conv2d_32, parameter_200)

        # pd_op.sigmoid_: (-1x636x1x1xf32) <- (-1x636x1x1xf32)
        sigmoid__6 = paddle._C_ops.sigmoid(add__18)

        # pd_op.multiply_: (-1x636x14x14xf32) <- (-1x636x14x14xf32, -1x636x1x1xf32)
        multiply__6 = paddle._C_ops.multiply(batch_norm__204, sigmoid__6)

        # pd_op.relu6: (-1x636x14x14xf32) <- (-1x636x14x14xf32)
        relu6_9 = paddle._C_ops.relu6(multiply__6)

        # pd_op.conv2d: (-1x117x14x14xf32) <- (-1x636x14x14xf32, 117x636x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(relu6_9, parameter_201, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x117x14x14xf32, 117xf32, 117xf32, xf32, xf32, None) <- (-1x117x14x14xf32, 117xf32, 117xf32, 117xf32, 117xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_202, parameter_203, parameter_204, parameter_205, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x106x14x14xf32) <- (-1x117x14x14xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(batch_norm__216, [1], constant_0, constant_8, [1], [])

        # pd_op.add_: (-1x106x14x14xf32) <- (-1x106x14x14xf32, -1x106x14x14xf32)
        add__19 = paddle._C_ops.add(slice_5, set_value_with_tensor__4)

        # pd_op.set_value_with_tensor_: (-1x117x14x14xf32) <- (-1x117x14x14xf32, -1x106x14x14xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__5 = paddle._C_ops.set_value_with_tensor(batch_norm__216, add__19, constant_0, constant_8, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x702x14x14xf32) <- (-1x117x14x14xf32, 702x117x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(set_value_with_tensor__5, parameter_206, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x702x14x14xf32, 702xf32, 702xf32, xf32, xf32, None) <- (-1x702x14x14xf32, 702xf32, 702xf32, 702xf32, 702xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_207, parameter_208, parameter_209, parameter_210, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x702x14x14xf32) <- (-1x702x14x14xf32)
        swish_10 = paddle._C_ops.swish(batch_norm__222)

        # pd_op.depthwise_conv2d: (-1x702x14x14xf32) <- (-1x702x14x14xf32, 702x1x3x3xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(swish_10, parameter_211, [1, 1], [1, 1], 'EXPLICIT', 702, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x702x14x14xf32, 702xf32, 702xf32, xf32, xf32, None) <- (-1x702x14x14xf32, 702xf32, 702xf32, 702xf32, 702xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_10, parameter_212, parameter_213, parameter_214, parameter_215, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x702x1x1xf32) <- (-1x702x14x14xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(batch_norm__228, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x58x1x1xf32) <- (-1x702x1x1xf32, 58x702x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(pool2d_7, parameter_216, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x58x1x1xf32) <- (-1x58x1x1xf32, 1x58x1x1xf32)
        add__20 = paddle._C_ops.add(conv2d_35, parameter_217)

        # pd_op.batch_norm_: (-1x58x1x1xf32, 58xf32, 58xf32, xf32, xf32, None) <- (-1x58x1x1xf32, 58xf32, 58xf32, 58xf32, 58xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__20, parameter_218, parameter_219, parameter_220, parameter_221, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x58x1x1xf32) <- (-1x58x1x1xf32)
        relu__7 = paddle._C_ops.relu(batch_norm__234)

        # pd_op.conv2d: (-1x702x1x1xf32) <- (-1x58x1x1xf32, 702x58x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(relu__7, parameter_222, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x702x1x1xf32) <- (-1x702x1x1xf32, 1x702x1x1xf32)
        add__21 = paddle._C_ops.add(conv2d_36, parameter_223)

        # pd_op.sigmoid_: (-1x702x1x1xf32) <- (-1x702x1x1xf32)
        sigmoid__7 = paddle._C_ops.sigmoid(add__21)

        # pd_op.multiply_: (-1x702x14x14xf32) <- (-1x702x14x14xf32, -1x702x1x1xf32)
        multiply__7 = paddle._C_ops.multiply(batch_norm__228, sigmoid__7)

        # pd_op.relu6: (-1x702x14x14xf32) <- (-1x702x14x14xf32)
        relu6_10 = paddle._C_ops.relu6(multiply__7)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x702x14x14xf32, 128x702x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(relu6_10, parameter_224, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_225, parameter_226, parameter_227, parameter_228, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x117x14x14xf32) <- (-1x128x14x14xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(batch_norm__240, [1], constant_0, constant_9, [1], [])

        # pd_op.add_: (-1x117x14x14xf32) <- (-1x117x14x14xf32, -1x117x14x14xf32)
        add__22 = paddle._C_ops.add(slice_6, set_value_with_tensor__5)

        # pd_op.set_value_with_tensor_: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x117x14x14xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__6 = paddle._C_ops.set_value_with_tensor(batch_norm__240, add__22, constant_0, constant_9, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x128x14x14xf32, 768x128x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(set_value_with_tensor__6, parameter_229, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x768x14x14xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_230, parameter_231, parameter_232, parameter_233, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x768x14x14xf32) <- (-1x768x14x14xf32)
        swish_11 = paddle._C_ops.swish(batch_norm__246)

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x14x14xf32, 768x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(swish_11, parameter_234, [2, 2], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_11, parameter_235, parameter_236, parameter_237, parameter_238, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x768x1x1xf32) <- (-1x768x7x7xf32, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(batch_norm__252, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf32) <- (-1x768x1x1xf32, 64x768x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(pool2d_8, parameter_239, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1x64x1x1xf32)
        add__23 = paddle._C_ops.add(conv2d_39, parameter_240)

        # pd_op.batch_norm_: (-1x64x1x1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x1x1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__23, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x1xf32) <- (-1x64x1x1xf32)
        relu__8 = paddle._C_ops.relu(batch_norm__258)

        # pd_op.conv2d: (-1x768x1x1xf32) <- (-1x64x1x1xf32, 768x64x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(relu__8, parameter_245, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x1x1xf32) <- (-1x768x1x1xf32, 1x768x1x1xf32)
        add__24 = paddle._C_ops.add(conv2d_40, parameter_246)

        # pd_op.sigmoid_: (-1x768x1x1xf32) <- (-1x768x1x1xf32)
        sigmoid__8 = paddle._C_ops.sigmoid(add__24)

        # pd_op.multiply_: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x1x1xf32)
        multiply__8 = paddle._C_ops.multiply(batch_norm__252, sigmoid__8)

        # pd_op.relu6: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        relu6_11 = paddle._C_ops.relu6(multiply__8)

        # pd_op.conv2d: (-1x140x7x7xf32) <- (-1x768x7x7xf32, 140x768x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(relu6_11, parameter_247, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x140x7x7xf32, 140xf32, 140xf32, xf32, xf32, None) <- (-1x140x7x7xf32, 140xf32, 140xf32, 140xf32, 140xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_248, parameter_249, parameter_250, parameter_251, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x840x7x7xf32) <- (-1x140x7x7xf32, 840x140x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(batch_norm__264, parameter_252, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x840x7x7xf32, 840xf32, 840xf32, xf32, xf32, None) <- (-1x840x7x7xf32, 840xf32, 840xf32, 840xf32, 840xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_253, parameter_254, parameter_255, parameter_256, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x840x7x7xf32) <- (-1x840x7x7xf32)
        swish_12 = paddle._C_ops.swish(batch_norm__270)

        # pd_op.depthwise_conv2d: (-1x840x7x7xf32) <- (-1x840x7x7xf32, 840x1x3x3xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(swish_12, parameter_257, [1, 1], [1, 1], 'EXPLICIT', 840, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x840x7x7xf32, 840xf32, 840xf32, xf32, xf32, None) <- (-1x840x7x7xf32, 840xf32, 840xf32, 840xf32, 840xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_12, parameter_258, parameter_259, parameter_260, parameter_261, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x840x1x1xf32) <- (-1x840x7x7xf32, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(batch_norm__276, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x70x1x1xf32) <- (-1x840x1x1xf32, 70x840x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(pool2d_9, parameter_262, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x70x1x1xf32) <- (-1x70x1x1xf32, 1x70x1x1xf32)
        add__25 = paddle._C_ops.add(conv2d_43, parameter_263)

        # pd_op.batch_norm_: (-1x70x1x1xf32, 70xf32, 70xf32, xf32, xf32, None) <- (-1x70x1x1xf32, 70xf32, 70xf32, 70xf32, 70xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__25, parameter_264, parameter_265, parameter_266, parameter_267, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x70x1x1xf32) <- (-1x70x1x1xf32)
        relu__9 = paddle._C_ops.relu(batch_norm__282)

        # pd_op.conv2d: (-1x840x1x1xf32) <- (-1x70x1x1xf32, 840x70x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(relu__9, parameter_268, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x840x1x1xf32) <- (-1x840x1x1xf32, 1x840x1x1xf32)
        add__26 = paddle._C_ops.add(conv2d_44, parameter_269)

        # pd_op.sigmoid_: (-1x840x1x1xf32) <- (-1x840x1x1xf32)
        sigmoid__9 = paddle._C_ops.sigmoid(add__26)

        # pd_op.multiply_: (-1x840x7x7xf32) <- (-1x840x7x7xf32, -1x840x1x1xf32)
        multiply__9 = paddle._C_ops.multiply(batch_norm__276, sigmoid__9)

        # pd_op.relu6: (-1x840x7x7xf32) <- (-1x840x7x7xf32)
        relu6_12 = paddle._C_ops.relu6(multiply__9)

        # pd_op.conv2d: (-1x151x7x7xf32) <- (-1x840x7x7xf32, 151x840x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(relu6_12, parameter_270, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x151x7x7xf32, 151xf32, 151xf32, xf32, xf32, None) <- (-1x151x7x7xf32, 151xf32, 151xf32, 151xf32, 151xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_271, parameter_272, parameter_273, parameter_274, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x140x7x7xf32) <- (-1x151x7x7xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(batch_norm__288, [1], constant_0, constant_10, [1], [])

        # pd_op.add_: (-1x140x7x7xf32) <- (-1x140x7x7xf32, -1x140x7x7xf32)
        add__27 = paddle._C_ops.add(slice_7, batch_norm__264)

        # pd_op.set_value_with_tensor_: (-1x151x7x7xf32) <- (-1x151x7x7xf32, -1x140x7x7xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__7 = paddle._C_ops.set_value_with_tensor(batch_norm__288, add__27, constant_0, constant_10, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x906x7x7xf32) <- (-1x151x7x7xf32, 906x151x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(set_value_with_tensor__7, parameter_275, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x906x7x7xf32, 906xf32, 906xf32, xf32, xf32, None) <- (-1x906x7x7xf32, 906xf32, 906xf32, 906xf32, 906xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_276, parameter_277, parameter_278, parameter_279, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x906x7x7xf32) <- (-1x906x7x7xf32)
        swish_13 = paddle._C_ops.swish(batch_norm__294)

        # pd_op.depthwise_conv2d: (-1x906x7x7xf32) <- (-1x906x7x7xf32, 906x1x3x3xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(swish_13, parameter_280, [1, 1], [1, 1], 'EXPLICIT', 906, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x906x7x7xf32, 906xf32, 906xf32, xf32, xf32, None) <- (-1x906x7x7xf32, 906xf32, 906xf32, 906xf32, 906xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_13, parameter_281, parameter_282, parameter_283, parameter_284, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x906x1x1xf32) <- (-1x906x7x7xf32, 2xi64)
        pool2d_10 = paddle._C_ops.pool2d(batch_norm__300, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x75x1x1xf32) <- (-1x906x1x1xf32, 75x906x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(pool2d_10, parameter_285, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x75x1x1xf32) <- (-1x75x1x1xf32, 1x75x1x1xf32)
        add__28 = paddle._C_ops.add(conv2d_47, parameter_286)

        # pd_op.batch_norm_: (-1x75x1x1xf32, 75xf32, 75xf32, xf32, xf32, None) <- (-1x75x1x1xf32, 75xf32, 75xf32, 75xf32, 75xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__28, parameter_287, parameter_288, parameter_289, parameter_290, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x75x1x1xf32) <- (-1x75x1x1xf32)
        relu__10 = paddle._C_ops.relu(batch_norm__306)

        # pd_op.conv2d: (-1x906x1x1xf32) <- (-1x75x1x1xf32, 906x75x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(relu__10, parameter_291, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x906x1x1xf32) <- (-1x906x1x1xf32, 1x906x1x1xf32)
        add__29 = paddle._C_ops.add(conv2d_48, parameter_292)

        # pd_op.sigmoid_: (-1x906x1x1xf32) <- (-1x906x1x1xf32)
        sigmoid__10 = paddle._C_ops.sigmoid(add__29)

        # pd_op.multiply_: (-1x906x7x7xf32) <- (-1x906x7x7xf32, -1x906x1x1xf32)
        multiply__10 = paddle._C_ops.multiply(batch_norm__300, sigmoid__10)

        # pd_op.relu6: (-1x906x7x7xf32) <- (-1x906x7x7xf32)
        relu6_13 = paddle._C_ops.relu6(multiply__10)

        # pd_op.conv2d: (-1x162x7x7xf32) <- (-1x906x7x7xf32, 162x906x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(relu6_13, parameter_293, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x162x7x7xf32, 162xf32, 162xf32, xf32, xf32, None) <- (-1x162x7x7xf32, 162xf32, 162xf32, 162xf32, 162xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_294, parameter_295, parameter_296, parameter_297, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x151x7x7xf32) <- (-1x162x7x7xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(batch_norm__312, [1], constant_0, constant_11, [1], [])

        # pd_op.add_: (-1x151x7x7xf32) <- (-1x151x7x7xf32, -1x151x7x7xf32)
        add__30 = paddle._C_ops.add(slice_8, set_value_with_tensor__7)

        # pd_op.set_value_with_tensor_: (-1x162x7x7xf32) <- (-1x162x7x7xf32, -1x151x7x7xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__8 = paddle._C_ops.set_value_with_tensor(batch_norm__312, add__30, constant_0, constant_11, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x972x7x7xf32) <- (-1x162x7x7xf32, 972x162x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(set_value_with_tensor__8, parameter_298, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x972x7x7xf32, 972xf32, 972xf32, xf32, xf32, None) <- (-1x972x7x7xf32, 972xf32, 972xf32, 972xf32, 972xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_299, parameter_300, parameter_301, parameter_302, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x972x7x7xf32) <- (-1x972x7x7xf32)
        swish_14 = paddle._C_ops.swish(batch_norm__318)

        # pd_op.depthwise_conv2d: (-1x972x7x7xf32) <- (-1x972x7x7xf32, 972x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(swish_14, parameter_303, [1, 1], [1, 1], 'EXPLICIT', 972, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x972x7x7xf32, 972xf32, 972xf32, xf32, xf32, None) <- (-1x972x7x7xf32, 972xf32, 972xf32, 972xf32, 972xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_14, parameter_304, parameter_305, parameter_306, parameter_307, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x972x1x1xf32) <- (-1x972x7x7xf32, 2xi64)
        pool2d_11 = paddle._C_ops.pool2d(batch_norm__324, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x81x1x1xf32) <- (-1x972x1x1xf32, 81x972x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(pool2d_11, parameter_308, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x81x1x1xf32) <- (-1x81x1x1xf32, 1x81x1x1xf32)
        add__31 = paddle._C_ops.add(conv2d_51, parameter_309)

        # pd_op.batch_norm_: (-1x81x1x1xf32, 81xf32, 81xf32, xf32, xf32, None) <- (-1x81x1x1xf32, 81xf32, 81xf32, 81xf32, 81xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__31, parameter_310, parameter_311, parameter_312, parameter_313, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x81x1x1xf32) <- (-1x81x1x1xf32)
        relu__11 = paddle._C_ops.relu(batch_norm__330)

        # pd_op.conv2d: (-1x972x1x1xf32) <- (-1x81x1x1xf32, 972x81x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(relu__11, parameter_314, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x972x1x1xf32) <- (-1x972x1x1xf32, 1x972x1x1xf32)
        add__32 = paddle._C_ops.add(conv2d_52, parameter_315)

        # pd_op.sigmoid_: (-1x972x1x1xf32) <- (-1x972x1x1xf32)
        sigmoid__11 = paddle._C_ops.sigmoid(add__32)

        # pd_op.multiply_: (-1x972x7x7xf32) <- (-1x972x7x7xf32, -1x972x1x1xf32)
        multiply__11 = paddle._C_ops.multiply(batch_norm__324, sigmoid__11)

        # pd_op.relu6: (-1x972x7x7xf32) <- (-1x972x7x7xf32)
        relu6_14 = paddle._C_ops.relu6(multiply__11)

        # pd_op.conv2d: (-1x174x7x7xf32) <- (-1x972x7x7xf32, 174x972x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(relu6_14, parameter_316, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x174x7x7xf32, 174xf32, 174xf32, xf32, xf32, None) <- (-1x174x7x7xf32, 174xf32, 174xf32, 174xf32, 174xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_317, parameter_318, parameter_319, parameter_320, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x162x7x7xf32) <- (-1x174x7x7xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(batch_norm__336, [1], constant_0, constant_12, [1], [])

        # pd_op.add_: (-1x162x7x7xf32) <- (-1x162x7x7xf32, -1x162x7x7xf32)
        add__33 = paddle._C_ops.add(slice_9, set_value_with_tensor__8)

        # pd_op.set_value_with_tensor_: (-1x174x7x7xf32) <- (-1x174x7x7xf32, -1x162x7x7xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__9 = paddle._C_ops.set_value_with_tensor(batch_norm__336, add__33, constant_0, constant_12, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x1044x7x7xf32) <- (-1x174x7x7xf32, 1044x174x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(set_value_with_tensor__9, parameter_321, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1044x7x7xf32, 1044xf32, 1044xf32, xf32, xf32, None) <- (-1x1044x7x7xf32, 1044xf32, 1044xf32, 1044xf32, 1044xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_322, parameter_323, parameter_324, parameter_325, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1044x7x7xf32) <- (-1x1044x7x7xf32)
        swish_15 = paddle._C_ops.swish(batch_norm__342)

        # pd_op.depthwise_conv2d: (-1x1044x7x7xf32) <- (-1x1044x7x7xf32, 1044x1x3x3xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(swish_15, parameter_326, [1, 1], [1, 1], 'EXPLICIT', 1044, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x1044x7x7xf32, 1044xf32, 1044xf32, xf32, xf32, None) <- (-1x1044x7x7xf32, 1044xf32, 1044xf32, 1044xf32, 1044xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_15, parameter_327, parameter_328, parameter_329, parameter_330, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x1044x1x1xf32) <- (-1x1044x7x7xf32, 2xi64)
        pool2d_12 = paddle._C_ops.pool2d(batch_norm__348, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x87x1x1xf32) <- (-1x1044x1x1xf32, 87x1044x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(pool2d_12, parameter_331, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x87x1x1xf32) <- (-1x87x1x1xf32, 1x87x1x1xf32)
        add__34 = paddle._C_ops.add(conv2d_55, parameter_332)

        # pd_op.batch_norm_: (-1x87x1x1xf32, 87xf32, 87xf32, xf32, xf32, None) <- (-1x87x1x1xf32, 87xf32, 87xf32, 87xf32, 87xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__34, parameter_333, parameter_334, parameter_335, parameter_336, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x87x1x1xf32) <- (-1x87x1x1xf32)
        relu__12 = paddle._C_ops.relu(batch_norm__354)

        # pd_op.conv2d: (-1x1044x1x1xf32) <- (-1x87x1x1xf32, 1044x87x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(relu__12, parameter_337, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1044x1x1xf32) <- (-1x1044x1x1xf32, 1x1044x1x1xf32)
        add__35 = paddle._C_ops.add(conv2d_56, parameter_338)

        # pd_op.sigmoid_: (-1x1044x1x1xf32) <- (-1x1044x1x1xf32)
        sigmoid__12 = paddle._C_ops.sigmoid(add__35)

        # pd_op.multiply_: (-1x1044x7x7xf32) <- (-1x1044x7x7xf32, -1x1044x1x1xf32)
        multiply__12 = paddle._C_ops.multiply(batch_norm__348, sigmoid__12)

        # pd_op.relu6: (-1x1044x7x7xf32) <- (-1x1044x7x7xf32)
        relu6_15 = paddle._C_ops.relu6(multiply__12)

        # pd_op.conv2d: (-1x185x7x7xf32) <- (-1x1044x7x7xf32, 185x1044x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(relu6_15, parameter_339, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x185x7x7xf32, 185xf32, 185xf32, xf32, xf32, None) <- (-1x185x7x7xf32, 185xf32, 185xf32, 185xf32, 185xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_340, parameter_341, parameter_342, parameter_343, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.slice: (-1x174x7x7xf32) <- (-1x185x7x7xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(batch_norm__360, [1], constant_0, constant_13, [1], [])

        # pd_op.add_: (-1x174x7x7xf32) <- (-1x174x7x7xf32, -1x174x7x7xf32)
        add__36 = paddle._C_ops.add(slice_10, set_value_with_tensor__9)

        # pd_op.set_value_with_tensor_: (-1x185x7x7xf32) <- (-1x185x7x7xf32, -1x174x7x7xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__10 = paddle._C_ops.set_value_with_tensor(batch_norm__360, add__36, constant_0, constant_13, constant_2, [1], [], [])

        # pd_op.conv2d: (-1x1280x7x7xf32) <- (-1x185x7x7xf32, 1280x185x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(set_value_with_tensor__10, parameter_344, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1280x7x7xf32, 1280xf32, 1280xf32, xf32, xf32, None) <- (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_345, parameter_346, parameter_347, parameter_348, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32)
        swish_16 = paddle._C_ops.swish(batch_norm__366)

        # pd_op.pool2d: (-1x1280x1x1xf32) <- (-1x1280x7x7xf32, 2xi64)
        pool2d_13 = paddle._C_ops.pool2d(swish_16, constant_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.dropout: (-1x1280x1x1xf32, None) <- (-1x1280x1x1xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(pool2d_13, None, constant_14, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x1000x1x1xf32) <- (-1x1280x1x1xf32, 1000x1280x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(dropout_0, parameter_349, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1000x1x1xf32) <- (-1x1000x1x1xf32, 1x1000x1x1xf32)
        add__37 = paddle._C_ops.add(conv2d_59, parameter_350)

        # pd_op.squeeze_: (-1x1000x1xf32, None) <- (-1x1000x1x1xf32, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze(add__37, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.squeeze_: (-1x1000xf32, None) <- (-1x1000x1xf32, 1xi64)
        squeeze__2, squeeze__3 = (lambda x, f: f(x))(paddle._C_ops.squeeze(squeeze__0, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__0 = paddle._C_ops.softmax(squeeze__2, -1)
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

    def forward(self, constant_15, parameter_350, constant_14, constant_13, parameter_338, parameter_332, constant_12, parameter_315, parameter_309, constant_11, parameter_292, parameter_286, constant_10, parameter_269, parameter_263, parameter_246, parameter_240, constant_9, parameter_223, parameter_217, constant_8, parameter_200, parameter_194, constant_7, parameter_177, parameter_171, constant_6, parameter_154, parameter_148, constant_5, parameter_131, parameter_125, parameter_108, parameter_102, constant_4, parameter_85, parameter_79, parameter_62, parameter_56, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_60, parameter_57, parameter_59, parameter_58, parameter_61, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_152, parameter_149, parameter_151, parameter_150, parameter_153, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_175, parameter_172, parameter_174, parameter_173, parameter_176, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_224, parameter_228, parameter_225, parameter_227, parameter_226, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_267, parameter_264, parameter_266, parameter_265, parameter_268, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_302, parameter_299, parameter_301, parameter_300, parameter_303, parameter_307, parameter_304, parameter_306, parameter_305, parameter_308, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_336, parameter_333, parameter_335, parameter_334, parameter_337, parameter_339, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_348, parameter_345, parameter_347, parameter_346, parameter_349, feed_0):
        return self.builtin_module_1451_0_0(constant_15, parameter_350, constant_14, constant_13, parameter_338, parameter_332, constant_12, parameter_315, parameter_309, constant_11, parameter_292, parameter_286, constant_10, parameter_269, parameter_263, parameter_246, parameter_240, constant_9, parameter_223, parameter_217, constant_8, parameter_200, parameter_194, constant_7, parameter_177, parameter_171, constant_6, parameter_154, parameter_148, constant_5, parameter_131, parameter_125, parameter_108, parameter_102, constant_4, parameter_85, parameter_79, parameter_62, parameter_56, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_60, parameter_57, parameter_59, parameter_58, parameter_61, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_152, parameter_149, parameter_151, parameter_150, parameter_153, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_175, parameter_172, parameter_174, parameter_173, parameter_176, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_224, parameter_228, parameter_225, parameter_227, parameter_226, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_267, parameter_264, parameter_266, parameter_265, parameter_268, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_302, parameter_299, parameter_301, parameter_300, parameter_303, parameter_307, parameter_304, parameter_306, parameter_305, parameter_308, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_336, parameter_333, parameter_335, parameter_334, parameter_337, parameter_339, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_348, parameter_345, parameter_347, parameter_346, parameter_349, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1451_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_15
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            # parameter_350
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_14
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_13
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            # parameter_338
            paddle.uniform([1, 1044, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([1, 87, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_12
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            # parameter_315
            paddle.uniform([1, 972, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([1, 81, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_11
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            # parameter_292
            paddle.uniform([1, 906, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([1, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_10
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            # parameter_269
            paddle.uniform([1, 840, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([1, 70, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_9
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            # parameter_223
            paddle.uniform([1, 702, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([1, 58, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_8
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            # parameter_200
            paddle.uniform([1, 636, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([1, 53, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_7
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            # parameter_177
            paddle.uniform([1, 570, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([1, 47, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_6
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            # parameter_154
            paddle.uniform([1, 504, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([1, 42, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_5
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            # parameter_131
            paddle.uniform([1, 432, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([1, 366, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([1, 30, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_4
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            # parameter_85
            paddle.uniform([1, 300, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([1, 25, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([1, 228, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([1, 19, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_3
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_2
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_1
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
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
            paddle.uniform([16, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([96, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([27, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([27], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([27], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([27], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([27], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([162, 27, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([162, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([38, 162, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([228, 38, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([228], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([228], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([228], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([228], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([228, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([228], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([228], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([228], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([228], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([19, 228, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([19], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([19], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([19], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([19], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([228, 19, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([50, 228, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([300, 50, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([25, 300, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([300, 25, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([61, 300, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([61], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([61], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([61], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([61], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([366, 61, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([366], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([366], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([366], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([366], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([366, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([366], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([366], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([366], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([366], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([30, 366, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([30], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([30], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([30], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([30], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([366, 30, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([72, 366, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([432, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([432, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([36, 432, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([432, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([84, 432, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([504, 84, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([504], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([504], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([504], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([504], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([504, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([504], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([504], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([504], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([504], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([42, 504, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([42], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([42], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([42], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([42], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([504, 42, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([95, 504, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([95], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([95], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([95], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([95], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([570, 95, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([570], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([570], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([570], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([570], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([570, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([570], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([570], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([570], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([570], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([47, 570, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([47], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([47], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([47], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([47], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([570, 47, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([106, 570, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([106], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([106], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([106], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([106], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([636, 106, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([636], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([636], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([636], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([636], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([636, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([636], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([636], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([636], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([636], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([53, 636, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([53], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([53], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([53], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([53], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([636, 53, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([117, 636, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([117], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([117], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([117], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([117], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([702, 117, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([702], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([702], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([702], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([702], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([702, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([702], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([702], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([702], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([702], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([58, 702, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([58], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([58], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([58], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([58], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([702, 58, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([128, 702, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([768, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([64, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([768, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([140, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([140], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([140], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([140], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([140], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([840, 140, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([840], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([840], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([840], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([840], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([840, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([840], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([840], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([840], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([840], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([70, 840, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([840, 70, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([151, 840, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([151], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([151], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([151], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([151], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([906, 151, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([906], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([906], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([906], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([906], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([906, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([906], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([906], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([906], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([906], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([75, 906, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([75], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([75], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([75], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([75], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([906, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([162, 906, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([162], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([972, 162, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([972], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([972], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([972], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([972], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([972, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([972], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([972], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([972], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([972], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([81, 972, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([81], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([81], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([81], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([81], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([972, 81, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([174, 972, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([174], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([174], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([174], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([174], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1044, 174, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([1044], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([1044], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([1044], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1044], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([1044, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([1044], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([1044], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([1044], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([1044], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([87, 1044, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([87], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([87], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([87], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([87], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1044, 87, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([185, 1044, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([185], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([185], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([185], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([185], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([1280, 185, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1000, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_15
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_350
            paddle.static.InputSpec(shape=[1, 1000, 1, 1], dtype='float32'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_338
            paddle.static.InputSpec(shape=[1, 1044, 1, 1], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[1, 87, 1, 1], dtype='float32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_315
            paddle.static.InputSpec(shape=[1, 972, 1, 1], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[1, 81, 1, 1], dtype='float32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_292
            paddle.static.InputSpec(shape=[1, 906, 1, 1], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[1, 75, 1, 1], dtype='float32'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_269
            paddle.static.InputSpec(shape=[1, 840, 1, 1], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[1, 70, 1, 1], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_223
            paddle.static.InputSpec(shape=[1, 702, 1, 1], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[1, 58, 1, 1], dtype='float32'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_200
            paddle.static.InputSpec(shape=[1, 636, 1, 1], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[1, 53, 1, 1], dtype='float32'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_177
            paddle.static.InputSpec(shape=[1, 570, 1, 1], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[1, 47, 1, 1], dtype='float32'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_154
            paddle.static.InputSpec(shape=[1, 504, 1, 1], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[1, 42, 1, 1], dtype='float32'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_131
            paddle.static.InputSpec(shape=[1, 432, 1, 1], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[1, 36, 1, 1], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[1, 366, 1, 1], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[1, 30, 1, 1], dtype='float32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_85
            paddle.static.InputSpec(shape=[1, 300, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[1, 25, 1, 1], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[1, 228, 1, 1], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[1, 19, 1, 1], dtype='float32'),
            # constant_3
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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
            paddle.static.InputSpec(shape=[16, 32, 1, 1], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[96, 16, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[27, 96, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[27], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[27], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[27], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[27], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[162, 27, 1, 1], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[162, 1, 3, 3], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[38, 162, 1, 1], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[228, 38, 1, 1], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[228], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[228], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[228], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[228], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[228, 1, 3, 3], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[228], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[228], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[228], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[228], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[19, 228, 1, 1], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[228, 19, 1, 1], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[50, 228, 1, 1], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[300, 50, 1, 1], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[300, 1, 3, 3], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[25, 300, 1, 1], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[300, 25, 1, 1], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[61, 300, 1, 1], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[61], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[61], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[61], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[61], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[366, 61, 1, 1], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[366], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[366], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[366], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[366], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[366, 1, 3, 3], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[366], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[366], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[366], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[366], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[30, 366, 1, 1], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[30], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[30], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[30], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[30], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[366, 30, 1, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[72, 366, 1, 1], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[432, 72, 1, 1], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[432, 1, 3, 3], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[36, 432, 1, 1], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[432, 36, 1, 1], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[84, 432, 1, 1], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[84], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[84], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[84], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[84], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[504, 84, 1, 1], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[504], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[504], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[504], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[504], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[504, 1, 3, 3], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[504], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[504], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[504], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[504], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[42, 504, 1, 1], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[42], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[42], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[42], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[42], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[504, 42, 1, 1], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[95, 504, 1, 1], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[95], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[95], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[95], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[95], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[570, 95, 1, 1], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[570], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[570], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[570], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[570], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[570, 1, 3, 3], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[570], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[570], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[570], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[570], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[47, 570, 1, 1], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[47], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[47], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[47], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[47], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[570, 47, 1, 1], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[106, 570, 1, 1], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[106], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[106], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[106], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[106], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[636, 106, 1, 1], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[636], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[636], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[636], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[636], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[636, 1, 3, 3], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[636], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[636], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[636], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[636], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[53, 636, 1, 1], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[53], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[53], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[53], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[53], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[636, 53, 1, 1], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[117, 636, 1, 1], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[117], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[117], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[117], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[117], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[702, 117, 1, 1], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[702], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[702], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[702], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[702], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[702, 1, 3, 3], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[702], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[702], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[702], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[702], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[58, 702, 1, 1], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[58], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[58], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[58], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[58], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[702, 58, 1, 1], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[128, 702, 1, 1], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[768, 128, 1, 1], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[64, 768, 1, 1], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[768, 64, 1, 1], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[140, 768, 1, 1], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[140], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[140], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[140], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[140], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[840, 140, 1, 1], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[840], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[840], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[840], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[840], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[840, 1, 3, 3], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[840], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[840], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[840], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[840], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[70, 840, 1, 1], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[840, 70, 1, 1], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[151, 840, 1, 1], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[151], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[151], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[151], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[151], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[906, 151, 1, 1], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[906], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[906], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[906], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[906], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[906, 1, 3, 3], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[906], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[906], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[906], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[906], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[75, 906, 1, 1], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[75], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[75], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[75], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[75], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[906, 75, 1, 1], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[162, 906, 1, 1], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[162], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[972, 162, 1, 1], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[972], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[972], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[972], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[972], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[972, 1, 3, 3], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[972], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[972], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[972], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[972], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[81, 972, 1, 1], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[81], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[81], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[81], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[81], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[972, 81, 1, 1], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[174, 972, 1, 1], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[174], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[174], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[174], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[174], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[1044, 174, 1, 1], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[1044], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[1044], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[1044], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[1044], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[1044, 1, 3, 3], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[1044], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[1044], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[1044], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[1044], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[87, 1044, 1, 1], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[87], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[87], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[87], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[87], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[1044, 87, 1, 1], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[185, 1044, 1, 1], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[185], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[185], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[185], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[185], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[1280, 185, 1, 1], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[1000, 1280, 1, 1], dtype='float32'),
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