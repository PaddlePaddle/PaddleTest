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
    return [599][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1934_0_0(self, constant_23, constant_22, constant_21, constant_20, parameter_407, parameter_405, parameter_384, parameter_382, constant_19, parameter_361, parameter_359, constant_18, parameter_339, parameter_337, constant_17, parameter_316, parameter_314, parameter_292, parameter_290, constant_16, parameter_268, parameter_266, constant_15, constant_14, parameter_245, parameter_243, parameter_225, parameter_223, parameter_201, parameter_199, constant_13, parameter_177, parameter_175, constant_12, constant_11, parameter_154, parameter_152, constant_10, parameter_132, parameter_130, parameter_110, parameter_108, parameter_88, parameter_86, constant_9, constant_8, parameter_67, parameter_65, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_16, parameter_20, parameter_17, parameter_19, parameter_18, parameter_21, parameter_22, parameter_23, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_55, parameter_52, parameter_54, parameter_53, parameter_56, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_66, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_87, parameter_89, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_109, parameter_111, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_118, parameter_122, parameter_119, parameter_121, parameter_120, parameter_123, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_131, parameter_133, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_145, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_153, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_167, parameter_168, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_176, parameter_178, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_191, parameter_192, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_200, parameter_202, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_215, parameter_216, parameter_217, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_224, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_236, parameter_233, parameter_235, parameter_234, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_244, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_258, parameter_259, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_267, parameter_269, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_276, parameter_280, parameter_277, parameter_279, parameter_278, parameter_281, parameter_282, parameter_283, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_291, parameter_293, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_306, parameter_307, parameter_308, parameter_312, parameter_309, parameter_311, parameter_310, parameter_313, parameter_315, parameter_317, parameter_318, parameter_322, parameter_319, parameter_321, parameter_320, parameter_323, parameter_327, parameter_324, parameter_326, parameter_325, parameter_328, parameter_329, parameter_330, parameter_331, parameter_335, parameter_332, parameter_334, parameter_333, parameter_336, parameter_338, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_351, parameter_352, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_360, parameter_362, parameter_363, parameter_367, parameter_364, parameter_366, parameter_365, parameter_368, parameter_372, parameter_369, parameter_371, parameter_370, parameter_373, parameter_374, parameter_375, parameter_376, parameter_380, parameter_377, parameter_379, parameter_378, parameter_381, parameter_383, parameter_385, parameter_386, parameter_390, parameter_387, parameter_389, parameter_388, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_397, parameter_398, parameter_399, parameter_403, parameter_400, parameter_402, parameter_401, parameter_404, parameter_406, parameter_408, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_420, feed_0):

        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x3x224x224xf32, 24x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.depthwise_conv2d: (-1x24x112x112xf32) <- (-1x24x112x112xf32, 24x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(relu__0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', 24, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x24x112x112xf32) <- (-1x24x112x112xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x24x112x112xf32, 24x24x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu__1, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x24x112x112xf32) <- (-1x24x112x112xf32, -1x24x112x112xf32)
        add__0 = paddle._C_ops.add_(batch_norm__12, relu__0)

        # pd_op.split: ([-1x12x112x112xf32, -1x12x112x112xf32]) <- (-1x24x112x112xf32, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(add__0, constant_0, constant_1)

        # builtin.slice: (-1x12x112x112xf32) <- ([-1x12x112x112xf32, -1x12x112x112xf32])
        slice_0 = split_0[0]

        # pd_op.conv2d: (-1x72x112x112xf32) <- (-1x12x112x112xf32, 72x12x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(slice_0, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x12x112x112xf32) <- ([-1x12x112x112xf32, -1x12x112x112xf32])
        slice_1 = split_0[1]

        # pd_op.conv2d: (-1x72x112x112xf32) <- (-1x12x112x112xf32, 72x12x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(slice_1, parameter_16, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x72x112x112xf32, -1x72x112x112xf32]) <- (-1x72x112x112xf32, -1x72x112x112xf32)
        combine_0 = [conv2d_2, conv2d_3]

        # pd_op.concat: (-1x144x112x112xf32) <- ([-1x72x112x112xf32, -1x72x112x112xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, constant_1)

        # pd_op.batch_norm_: (-1x144x112x112xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x112x112xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_0, parameter_17, parameter_18, parameter_19, parameter_20, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x144x112x112xf32) <- (-1x144x112x112xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.split: ([-1x48x112x112xf32, -1x48x112x112xf32, -1x48x112x112xf32]) <- (-1x144x112x112xf32, 3xi64, 1xi32)
        split_1 = paddle._C_ops.split(relu__2, constant_2, constant_1)

        # builtin.slice: (-1x48x112x112xf32) <- ([-1x48x112x112xf32, -1x48x112x112xf32, -1x48x112x112xf32])
        slice_2 = split_1[0]

        # pd_op.depthwise_conv2d: (-1x48x56x56xf32) <- (-1x48x112x112xf32, 48x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(slice_2, parameter_21, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # builtin.slice: (-1x48x112x112xf32) <- ([-1x48x112x112xf32, -1x48x112x112xf32, -1x48x112x112xf32])
        slice_3 = split_1[1]

        # pd_op.depthwise_conv2d: (-1x48x56x56xf32) <- (-1x48x112x112xf32, 48x1x5x5xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(slice_3, parameter_22, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # builtin.slice: (-1x48x112x112xf32) <- ([-1x48x112x112xf32, -1x48x112x112xf32, -1x48x112x112xf32])
        slice_4 = split_1[2]

        # pd_op.depthwise_conv2d: (-1x48x56x56xf32) <- (-1x48x112x112xf32, 48x1x7x7xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(slice_4, parameter_23, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # builtin.combine: ([-1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32]) <- (-1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32)
        combine_1 = [depthwise_conv2d_1, depthwise_conv2d_2, depthwise_conv2d_3]

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, constant_1)

        # pd_op.batch_norm_: (-1x144x56x56xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_1, parameter_24, parameter_25, parameter_26, parameter_27, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x144x56x56xf32) <- (-1x144x56x56xf32)
        relu__3 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.split: ([-1x72x56x56xf32, -1x72x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        split_2 = paddle._C_ops.split(relu__3, constant_3, constant_1)

        # builtin.slice: (-1x72x56x56xf32) <- ([-1x72x56x56xf32, -1x72x56x56xf32])
        slice_5 = split_2[0]

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x72x56x56xf32, 16x72x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(slice_5, parameter_28, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x72x56x56xf32) <- ([-1x72x56x56xf32, -1x72x56x56xf32])
        slice_6 = split_2[1]

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x72x56x56xf32, 16x72x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(slice_6, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x16x56x56xf32, -1x16x56x56xf32]) <- (-1x16x56x56xf32, -1x16x56x56xf32)
        combine_2 = [conv2d_4, conv2d_5]

        # pd_op.concat: (-1x32x56x56xf32) <- ([-1x16x56x56xf32, -1x16x56x56xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, constant_1)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_2, parameter_30, parameter_31, parameter_32, parameter_33, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.split: ([-1x16x56x56xf32, -1x16x56x56xf32]) <- (-1x32x56x56xf32, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(batch_norm__30, constant_4, constant_1)

        # builtin.slice: (-1x16x56x56xf32) <- ([-1x16x56x56xf32, -1x16x56x56xf32])
        slice_7 = split_3[0]

        # pd_op.conv2d: (-1x48x56x56xf32) <- (-1x16x56x56xf32, 48x16x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(slice_7, parameter_34, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x16x56x56xf32) <- ([-1x16x56x56xf32, -1x16x56x56xf32])
        slice_8 = split_3[1]

        # pd_op.conv2d: (-1x48x56x56xf32) <- (-1x16x56x56xf32, 48x16x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(slice_8, parameter_35, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x48x56x56xf32, -1x48x56x56xf32]) <- (-1x48x56x56xf32, -1x48x56x56xf32)
        combine_3 = [conv2d_6, conv2d_7]

        # pd_op.concat: (-1x96x56x56xf32) <- ([-1x48x56x56xf32, -1x48x56x56xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, constant_1)

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_3, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.depthwise_conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(relu__4, parameter_40, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu__5 = paddle._C_ops.relu_(batch_norm__42)

        # pd_op.split: ([-1x48x56x56xf32, -1x48x56x56xf32]) <- (-1x96x56x56xf32, 2xi64, 1xi32)
        split_4 = paddle._C_ops.split(relu__5, constant_5, constant_1)

        # builtin.slice: (-1x48x56x56xf32) <- ([-1x48x56x56xf32, -1x48x56x56xf32])
        slice_9 = split_4[0]

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x48x56x56xf32, 16x48x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(slice_9, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x48x56x56xf32) <- ([-1x48x56x56xf32, -1x48x56x56xf32])
        slice_10 = split_4[1]

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x48x56x56xf32, 16x48x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(slice_10, parameter_46, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x16x56x56xf32, -1x16x56x56xf32]) <- (-1x16x56x56xf32, -1x16x56x56xf32)
        combine_4 = [conv2d_8, conv2d_9]

        # pd_op.concat: (-1x32x56x56xf32) <- ([-1x16x56x56xf32, -1x16x56x56xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, constant_1)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_4, parameter_47, parameter_48, parameter_49, parameter_50, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__1 = paddle._C_ops.add_(batch_norm__48, batch_norm__30)

        # pd_op.conv2d: (-1x192x56x56xf32) <- (-1x32x56x56xf32, 192x32x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(add__1, parameter_51, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x56x56xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_52, parameter_53, parameter_54, parameter_55, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x56x56xf32) <- (-1x192x56x56xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__54)

        # pd_op.split: ([-1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32]) <- (-1x192x56x56xf32, 4xi64, 1xi32)
        split_5 = paddle._C_ops.split(swish_0, constant_6, constant_1)

        # builtin.slice: (-1x48x56x56xf32) <- ([-1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32])
        slice_11 = split_5[0]

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x56x56xf32, 48x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(slice_11, parameter_56, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # builtin.slice: (-1x48x56x56xf32) <- ([-1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32])
        slice_12 = split_5[1]

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x56x56xf32, 48x1x5x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(slice_12, parameter_57, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # builtin.slice: (-1x48x56x56xf32) <- ([-1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32])
        slice_13 = split_5[2]

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x56x56xf32, 48x1x7x7xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(slice_13, parameter_58, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # builtin.slice: (-1x48x56x56xf32) <- ([-1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32, -1x48x56x56xf32])
        slice_14 = split_5[3]

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x56x56xf32, 48x1x9x9xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(slice_14, parameter_59, [2, 2], [4, 4], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # builtin.combine: ([-1x48x28x28xf32, -1x48x28x28xf32, -1x48x28x28xf32, -1x48x28x28xf32]) <- (-1x48x28x28xf32, -1x48x28x28xf32, -1x48x28x28xf32, -1x48x28x28xf32)
        combine_5 = [depthwise_conv2d_5, depthwise_conv2d_6, depthwise_conv2d_7, depthwise_conv2d_8]

        # pd_op.concat: (-1x192x28x28xf32) <- ([-1x48x28x28xf32, -1x48x28x28xf32, -1x48x28x28xf32, -1x48x28x28xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, constant_1)

        # pd_op.batch_norm_: (-1x192x28x28xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_5, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__60)

        # pd_op.pool2d: (-1x192x1x1xf32) <- (-1x192x28x28xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(swish_1, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x16x1x1xf32) <- (-1x192x1x1xf32, 16x192x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(pool2d_0, parameter_64, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x16x1x1xf32) <- (-1x16x1x1xf32, 1x16x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_11, parameter_65)

        # pd_op.swish: (-1x16x1x1xf32) <- (-1x16x1x1xf32)
        swish_2 = paddle._C_ops.swish(add__2)

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x16x1x1xf32, 192x16x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(swish_2, parameter_66, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_12, parameter_67)

        # pd_op.sigmoid_: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(add__3)

        # pd_op.multiply_: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x1x1xf32)
        multiply__0 = paddle._C_ops.multiply_(swish_1, sigmoid__0)

        # pd_op.conv2d: (-1x40x28x28xf32) <- (-1x192x28x28xf32, 40x192x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(multiply__0, parameter_68, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x28x28xf32, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_69, parameter_70, parameter_71, parameter_72, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.split: ([-1x20x28x28xf32, -1x20x28x28xf32]) <- (-1x40x28x28xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(batch_norm__66, constant_8, constant_1)

        # builtin.slice: (-1x20x28x28xf32) <- ([-1x20x28x28xf32, -1x20x28x28xf32])
        slice_15 = split_6[0]

        # pd_op.conv2d: (-1x120x28x28xf32) <- (-1x20x28x28xf32, 120x20x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(slice_15, parameter_73, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x20x28x28xf32) <- ([-1x20x28x28xf32, -1x20x28x28xf32])
        slice_16 = split_6[1]

        # pd_op.conv2d: (-1x120x28x28xf32) <- (-1x20x28x28xf32, 120x20x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(slice_16, parameter_74, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x120x28x28xf32, -1x120x28x28xf32)
        combine_6 = [conv2d_14, conv2d_15]

        # pd_op.concat: (-1x240x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, constant_1)

        # pd_op.batch_norm_: (-1x240x28x28xf32, 240xf32, 240xf32, xf32, xf32, None) <- (-1x240x28x28xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_6, parameter_75, parameter_76, parameter_77, parameter_78, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x28x28xf32) <- (-1x240x28x28xf32)
        swish_3 = paddle._C_ops.swish(batch_norm__72)

        # pd_op.split: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x240x28x28xf32, 2xi64, 1xi32)
        split_7 = paddle._C_ops.split(swish_3, constant_9, constant_1)

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_17 = split_7[0]

        # pd_op.depthwise_conv2d: (-1x120x28x28xf32) <- (-1x120x28x28xf32, 120x1x3x3xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(slice_17, parameter_79, [1, 1], [1, 1], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_18 = split_7[1]

        # pd_op.depthwise_conv2d: (-1x120x28x28xf32) <- (-1x120x28x28xf32, 120x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(slice_18, parameter_80, [1, 1], [2, 2], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.combine: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x120x28x28xf32, -1x120x28x28xf32)
        combine_7 = [depthwise_conv2d_9, depthwise_conv2d_10]

        # pd_op.concat: (-1x240x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, constant_1)

        # pd_op.batch_norm_: (-1x240x28x28xf32, 240xf32, 240xf32, xf32, xf32, None) <- (-1x240x28x28xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_7, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x28x28xf32) <- (-1x240x28x28xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__78)

        # pd_op.pool2d: (-1x240x1x1xf32) <- (-1x240x28x28xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(swish_4, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x240x1x1xf32, 20x240x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(pool2d_1, parameter_85, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_16, parameter_86)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_5 = paddle._C_ops.swish(add__4)

        # pd_op.conv2d: (-1x240x1x1xf32) <- (-1x20x1x1xf32, 240x20x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(swish_5, parameter_87, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x240x1x1xf32) <- (-1x240x1x1xf32, 1x240x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_17, parameter_88)

        # pd_op.sigmoid_: (-1x240x1x1xf32) <- (-1x240x1x1xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(add__5)

        # pd_op.multiply_: (-1x240x28x28xf32) <- (-1x240x28x28xf32, -1x240x1x1xf32)
        multiply__1 = paddle._C_ops.multiply_(swish_4, sigmoid__1)

        # pd_op.split: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x240x28x28xf32, 2xi64, 1xi32)
        split_8 = paddle._C_ops.split(multiply__1, constant_9, constant_1)

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_19 = split_8[0]

        # pd_op.conv2d: (-1x20x28x28xf32) <- (-1x120x28x28xf32, 20x120x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(slice_19, parameter_89, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_20 = split_8[1]

        # pd_op.conv2d: (-1x20x28x28xf32) <- (-1x120x28x28xf32, 20x120x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(slice_20, parameter_90, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x20x28x28xf32, -1x20x28x28xf32]) <- (-1x20x28x28xf32, -1x20x28x28xf32)
        combine_8 = [conv2d_18, conv2d_19]

        # pd_op.concat: (-1x40x28x28xf32) <- ([-1x20x28x28xf32, -1x20x28x28xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, constant_1)

        # pd_op.batch_norm_: (-1x40x28x28xf32, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_8, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x40x28x28xf32) <- (-1x40x28x28xf32, -1x40x28x28xf32)
        add__6 = paddle._C_ops.add_(batch_norm__84, batch_norm__66)

        # pd_op.split: ([-1x20x28x28xf32, -1x20x28x28xf32]) <- (-1x40x28x28xf32, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(add__6, constant_8, constant_1)

        # builtin.slice: (-1x20x28x28xf32) <- ([-1x20x28x28xf32, -1x20x28x28xf32])
        slice_21 = split_9[0]

        # pd_op.conv2d: (-1x120x28x28xf32) <- (-1x20x28x28xf32, 120x20x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(slice_21, parameter_95, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x20x28x28xf32) <- ([-1x20x28x28xf32, -1x20x28x28xf32])
        slice_22 = split_9[1]

        # pd_op.conv2d: (-1x120x28x28xf32) <- (-1x20x28x28xf32, 120x20x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(slice_22, parameter_96, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x120x28x28xf32, -1x120x28x28xf32)
        combine_9 = [conv2d_20, conv2d_21]

        # pd_op.concat: (-1x240x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, constant_1)

        # pd_op.batch_norm_: (-1x240x28x28xf32, 240xf32, 240xf32, xf32, xf32, None) <- (-1x240x28x28xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_9, parameter_97, parameter_98, parameter_99, parameter_100, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x28x28xf32) <- (-1x240x28x28xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__90)

        # pd_op.split: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x240x28x28xf32, 2xi64, 1xi32)
        split_10 = paddle._C_ops.split(swish_6, constant_9, constant_1)

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_23 = split_10[0]

        # pd_op.depthwise_conv2d: (-1x120x28x28xf32) <- (-1x120x28x28xf32, 120x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(slice_23, parameter_101, [1, 1], [1, 1], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_24 = split_10[1]

        # pd_op.depthwise_conv2d: (-1x120x28x28xf32) <- (-1x120x28x28xf32, 120x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(slice_24, parameter_102, [1, 1], [2, 2], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.combine: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x120x28x28xf32, -1x120x28x28xf32)
        combine_10 = [depthwise_conv2d_11, depthwise_conv2d_12]

        # pd_op.concat: (-1x240x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, constant_1)

        # pd_op.batch_norm_: (-1x240x28x28xf32, 240xf32, 240xf32, xf32, xf32, None) <- (-1x240x28x28xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_10, parameter_103, parameter_104, parameter_105, parameter_106, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x28x28xf32) <- (-1x240x28x28xf32)
        swish_7 = paddle._C_ops.swish(batch_norm__96)

        # pd_op.pool2d: (-1x240x1x1xf32) <- (-1x240x28x28xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(swish_7, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x240x1x1xf32, 20x240x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(pool2d_2, parameter_107, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__7 = paddle._C_ops.add_(conv2d_22, parameter_108)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_8 = paddle._C_ops.swish(add__7)

        # pd_op.conv2d: (-1x240x1x1xf32) <- (-1x20x1x1xf32, 240x20x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(swish_8, parameter_109, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x240x1x1xf32) <- (-1x240x1x1xf32, 1x240x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_23, parameter_110)

        # pd_op.sigmoid_: (-1x240x1x1xf32) <- (-1x240x1x1xf32)
        sigmoid__2 = paddle._C_ops.sigmoid_(add__8)

        # pd_op.multiply_: (-1x240x28x28xf32) <- (-1x240x28x28xf32, -1x240x1x1xf32)
        multiply__2 = paddle._C_ops.multiply_(swish_7, sigmoid__2)

        # pd_op.split: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x240x28x28xf32, 2xi64, 1xi32)
        split_11 = paddle._C_ops.split(multiply__2, constant_9, constant_1)

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_25 = split_11[0]

        # pd_op.conv2d: (-1x20x28x28xf32) <- (-1x120x28x28xf32, 20x120x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(slice_25, parameter_111, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_26 = split_11[1]

        # pd_op.conv2d: (-1x20x28x28xf32) <- (-1x120x28x28xf32, 20x120x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(slice_26, parameter_112, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x20x28x28xf32, -1x20x28x28xf32]) <- (-1x20x28x28xf32, -1x20x28x28xf32)
        combine_11 = [conv2d_24, conv2d_25]

        # pd_op.concat: (-1x40x28x28xf32) <- ([-1x20x28x28xf32, -1x20x28x28xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, constant_1)

        # pd_op.batch_norm_: (-1x40x28x28xf32, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_11, parameter_113, parameter_114, parameter_115, parameter_116, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x40x28x28xf32) <- (-1x40x28x28xf32, -1x40x28x28xf32)
        add__9 = paddle._C_ops.add_(batch_norm__102, add__6)

        # pd_op.split: ([-1x20x28x28xf32, -1x20x28x28xf32]) <- (-1x40x28x28xf32, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(add__9, constant_8, constant_1)

        # builtin.slice: (-1x20x28x28xf32) <- ([-1x20x28x28xf32, -1x20x28x28xf32])
        slice_27 = split_12[0]

        # pd_op.conv2d: (-1x120x28x28xf32) <- (-1x20x28x28xf32, 120x20x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(slice_27, parameter_117, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x20x28x28xf32) <- ([-1x20x28x28xf32, -1x20x28x28xf32])
        slice_28 = split_12[1]

        # pd_op.conv2d: (-1x120x28x28xf32) <- (-1x20x28x28xf32, 120x20x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(slice_28, parameter_118, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x120x28x28xf32, -1x120x28x28xf32)
        combine_12 = [conv2d_26, conv2d_27]

        # pd_op.concat: (-1x240x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, constant_1)

        # pd_op.batch_norm_: (-1x240x28x28xf32, 240xf32, 240xf32, xf32, xf32, None) <- (-1x240x28x28xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_12, parameter_119, parameter_120, parameter_121, parameter_122, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x28x28xf32) <- (-1x240x28x28xf32)
        swish_9 = paddle._C_ops.swish(batch_norm__108)

        # pd_op.split: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x240x28x28xf32, 2xi64, 1xi32)
        split_13 = paddle._C_ops.split(swish_9, constant_9, constant_1)

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_29 = split_13[0]

        # pd_op.depthwise_conv2d: (-1x120x28x28xf32) <- (-1x120x28x28xf32, 120x1x3x3xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(slice_29, parameter_123, [1, 1], [1, 1], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_30 = split_13[1]

        # pd_op.depthwise_conv2d: (-1x120x28x28xf32) <- (-1x120x28x28xf32, 120x1x5x5xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(slice_30, parameter_124, [1, 1], [2, 2], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.combine: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x120x28x28xf32, -1x120x28x28xf32)
        combine_13 = [depthwise_conv2d_13, depthwise_conv2d_14]

        # pd_op.concat: (-1x240x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, constant_1)

        # pd_op.batch_norm_: (-1x240x28x28xf32, 240xf32, 240xf32, xf32, xf32, None) <- (-1x240x28x28xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_13, parameter_125, parameter_126, parameter_127, parameter_128, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x28x28xf32) <- (-1x240x28x28xf32)
        swish_10 = paddle._C_ops.swish(batch_norm__114)

        # pd_op.pool2d: (-1x240x1x1xf32) <- (-1x240x28x28xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(swish_10, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x240x1x1xf32, 20x240x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(pool2d_3, parameter_129, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_28, parameter_130)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_11 = paddle._C_ops.swish(add__10)

        # pd_op.conv2d: (-1x240x1x1xf32) <- (-1x20x1x1xf32, 240x20x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(swish_11, parameter_131, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x240x1x1xf32) <- (-1x240x1x1xf32, 1x240x1x1xf32)
        add__11 = paddle._C_ops.add_(conv2d_29, parameter_132)

        # pd_op.sigmoid_: (-1x240x1x1xf32) <- (-1x240x1x1xf32)
        sigmoid__3 = paddle._C_ops.sigmoid_(add__11)

        # pd_op.multiply_: (-1x240x28x28xf32) <- (-1x240x28x28xf32, -1x240x1x1xf32)
        multiply__3 = paddle._C_ops.multiply_(swish_10, sigmoid__3)

        # pd_op.split: ([-1x120x28x28xf32, -1x120x28x28xf32]) <- (-1x240x28x28xf32, 2xi64, 1xi32)
        split_14 = paddle._C_ops.split(multiply__3, constant_9, constant_1)

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_31 = split_14[0]

        # pd_op.conv2d: (-1x20x28x28xf32) <- (-1x120x28x28xf32, 20x120x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(slice_31, parameter_133, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x120x28x28xf32) <- ([-1x120x28x28xf32, -1x120x28x28xf32])
        slice_32 = split_14[1]

        # pd_op.conv2d: (-1x20x28x28xf32) <- (-1x120x28x28xf32, 20x120x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(slice_32, parameter_134, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x20x28x28xf32, -1x20x28x28xf32]) <- (-1x20x28x28xf32, -1x20x28x28xf32)
        combine_14 = [conv2d_30, conv2d_31]

        # pd_op.concat: (-1x40x28x28xf32) <- ([-1x20x28x28xf32, -1x20x28x28xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, constant_1)

        # pd_op.batch_norm_: (-1x40x28x28xf32, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_14, parameter_135, parameter_136, parameter_137, parameter_138, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x40x28x28xf32) <- (-1x40x28x28xf32, -1x40x28x28xf32)
        add__12 = paddle._C_ops.add_(batch_norm__120, add__9)

        # pd_op.conv2d: (-1x240x28x28xf32) <- (-1x40x28x28xf32, 240x40x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(add__12, parameter_139, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x240x28x28xf32, 240xf32, 240xf32, xf32, xf32, None) <- (-1x240x28x28xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_140, parameter_141, parameter_142, parameter_143, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x28x28xf32) <- (-1x240x28x28xf32)
        swish_12 = paddle._C_ops.swish(batch_norm__126)

        # pd_op.split: ([-1x80x28x28xf32, -1x80x28x28xf32, -1x80x28x28xf32]) <- (-1x240x28x28xf32, 3xi64, 1xi32)
        split_15 = paddle._C_ops.split(swish_12, constant_10, constant_1)

        # builtin.slice: (-1x80x28x28xf32) <- ([-1x80x28x28xf32, -1x80x28x28xf32, -1x80x28x28xf32])
        slice_33 = split_15[0]

        # pd_op.depthwise_conv2d: (-1x80x14x14xf32) <- (-1x80x28x28xf32, 80x1x3x3xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(slice_33, parameter_144, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # builtin.slice: (-1x80x28x28xf32) <- ([-1x80x28x28xf32, -1x80x28x28xf32, -1x80x28x28xf32])
        slice_34 = split_15[1]

        # pd_op.depthwise_conv2d: (-1x80x14x14xf32) <- (-1x80x28x28xf32, 80x1x5x5xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(slice_34, parameter_145, [2, 2], [2, 2], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # builtin.slice: (-1x80x28x28xf32) <- ([-1x80x28x28xf32, -1x80x28x28xf32, -1x80x28x28xf32])
        slice_35 = split_15[2]

        # pd_op.depthwise_conv2d: (-1x80x14x14xf32) <- (-1x80x28x28xf32, 80x1x7x7xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(slice_35, parameter_146, [2, 2], [3, 3], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # builtin.combine: ([-1x80x14x14xf32, -1x80x14x14xf32, -1x80x14x14xf32]) <- (-1x80x14x14xf32, -1x80x14x14xf32, -1x80x14x14xf32)
        combine_15 = [depthwise_conv2d_15, depthwise_conv2d_16, depthwise_conv2d_17]

        # pd_op.concat: (-1x240x14x14xf32) <- ([-1x80x14x14xf32, -1x80x14x14xf32, -1x80x14x14xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, constant_1)

        # pd_op.batch_norm_: (-1x240x14x14xf32, 240xf32, 240xf32, xf32, xf32, None) <- (-1x240x14x14xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_15, parameter_147, parameter_148, parameter_149, parameter_150, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x240x14x14xf32) <- (-1x240x14x14xf32)
        swish_13 = paddle._C_ops.swish(batch_norm__132)

        # pd_op.pool2d: (-1x240x1x1xf32) <- (-1x240x14x14xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(swish_13, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x10x1x1xf32) <- (-1x240x1x1xf32, 10x240x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(pool2d_4, parameter_151, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x10x1x1xf32) <- (-1x10x1x1xf32, 1x10x1x1xf32)
        add__13 = paddle._C_ops.add_(conv2d_33, parameter_152)

        # pd_op.swish: (-1x10x1x1xf32) <- (-1x10x1x1xf32)
        swish_14 = paddle._C_ops.swish(add__13)

        # pd_op.conv2d: (-1x240x1x1xf32) <- (-1x10x1x1xf32, 240x10x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(swish_14, parameter_153, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x240x1x1xf32) <- (-1x240x1x1xf32, 1x240x1x1xf32)
        add__14 = paddle._C_ops.add_(conv2d_34, parameter_154)

        # pd_op.sigmoid_: (-1x240x1x1xf32) <- (-1x240x1x1xf32)
        sigmoid__4 = paddle._C_ops.sigmoid_(add__14)

        # pd_op.multiply_: (-1x240x14x14xf32) <- (-1x240x14x14xf32, -1x240x1x1xf32)
        multiply__4 = paddle._C_ops.multiply_(swish_13, sigmoid__4)

        # pd_op.conv2d: (-1x80x14x14xf32) <- (-1x240x14x14xf32, 80x240x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(multiply__4, parameter_155, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x14x14xf32, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.split: ([-1x40x14x14xf32, -1x40x14x14xf32]) <- (-1x80x14x14xf32, 2xi64, 1xi32)
        split_16 = paddle._C_ops.split(batch_norm__138, constant_11, constant_1)

        # builtin.slice: (-1x40x14x14xf32) <- ([-1x40x14x14xf32, -1x40x14x14xf32])
        slice_36 = split_16[0]

        # pd_op.conv2d: (-1x240x14x14xf32) <- (-1x40x14x14xf32, 240x40x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(slice_36, parameter_160, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x40x14x14xf32) <- ([-1x40x14x14xf32, -1x40x14x14xf32])
        slice_37 = split_16[1]

        # pd_op.conv2d: (-1x240x14x14xf32) <- (-1x40x14x14xf32, 240x40x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(slice_37, parameter_161, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x240x14x14xf32, -1x240x14x14xf32]) <- (-1x240x14x14xf32, -1x240x14x14xf32)
        combine_16 = [conv2d_36, conv2d_37]

        # pd_op.concat: (-1x480x14x14xf32) <- ([-1x240x14x14xf32, -1x240x14x14xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, constant_1)

        # pd_op.batch_norm_: (-1x480x14x14xf32, 480xf32, 480xf32, xf32, xf32, None) <- (-1x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_16, parameter_162, parameter_163, parameter_164, parameter_165, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x14x14xf32) <- (-1x480x14x14xf32)
        swish_15 = paddle._C_ops.swish(batch_norm__144)

        # pd_op.split: ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32]) <- (-1x480x14x14xf32, 4xi64, 1xi32)
        split_17 = paddle._C_ops.split(swish_15, constant_12, constant_1)

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_38 = split_17[0]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x3x3xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(slice_38, parameter_166, [1, 1], [1, 1], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_39 = split_17[1]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x5x5xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(slice_39, parameter_167, [1, 1], [2, 2], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_40 = split_17[2]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x7x7xf32)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(slice_40, parameter_168, [1, 1], [3, 3], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_41 = split_17[3]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x9x9xf32)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(slice_41, parameter_169, [1, 1], [4, 4], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.combine: ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32]) <- (-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32)
        combine_17 = [depthwise_conv2d_18, depthwise_conv2d_19, depthwise_conv2d_20, depthwise_conv2d_21]

        # pd_op.concat: (-1x480x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, constant_1)

        # pd_op.batch_norm_: (-1x480x14x14xf32, 480xf32, 480xf32, xf32, xf32, None) <- (-1x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_17, parameter_170, parameter_171, parameter_172, parameter_173, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x14x14xf32) <- (-1x480x14x14xf32)
        swish_16 = paddle._C_ops.swish(batch_norm__150)

        # pd_op.pool2d: (-1x480x1x1xf32) <- (-1x480x14x14xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(swish_16, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x480x1x1xf32, 20x480x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(pool2d_5, parameter_174, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__15 = paddle._C_ops.add_(conv2d_38, parameter_175)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_17 = paddle._C_ops.swish(add__15)

        # pd_op.conv2d: (-1x480x1x1xf32) <- (-1x20x1x1xf32, 480x20x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(swish_17, parameter_176, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x480x1x1xf32) <- (-1x480x1x1xf32, 1x480x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_39, parameter_177)

        # pd_op.sigmoid_: (-1x480x1x1xf32) <- (-1x480x1x1xf32)
        sigmoid__5 = paddle._C_ops.sigmoid_(add__16)

        # pd_op.multiply_: (-1x480x14x14xf32) <- (-1x480x14x14xf32, -1x480x1x1xf32)
        multiply__5 = paddle._C_ops.multiply_(swish_16, sigmoid__5)

        # pd_op.split: ([-1x240x14x14xf32, -1x240x14x14xf32]) <- (-1x480x14x14xf32, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(multiply__5, constant_13, constant_1)

        # builtin.slice: (-1x240x14x14xf32) <- ([-1x240x14x14xf32, -1x240x14x14xf32])
        slice_42 = split_18[0]

        # pd_op.conv2d: (-1x40x14x14xf32) <- (-1x240x14x14xf32, 40x240x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(slice_42, parameter_178, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x240x14x14xf32) <- ([-1x240x14x14xf32, -1x240x14x14xf32])
        slice_43 = split_18[1]

        # pd_op.conv2d: (-1x40x14x14xf32) <- (-1x240x14x14xf32, 40x240x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(slice_43, parameter_179, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x40x14x14xf32, -1x40x14x14xf32]) <- (-1x40x14x14xf32, -1x40x14x14xf32)
        combine_18 = [conv2d_40, conv2d_41]

        # pd_op.concat: (-1x80x14x14xf32) <- ([-1x40x14x14xf32, -1x40x14x14xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_18, constant_1)

        # pd_op.batch_norm_: (-1x80x14x14xf32, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_18, parameter_180, parameter_181, parameter_182, parameter_183, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x14x14xf32) <- (-1x80x14x14xf32, -1x80x14x14xf32)
        add__17 = paddle._C_ops.add_(batch_norm__156, batch_norm__138)

        # pd_op.split: ([-1x40x14x14xf32, -1x40x14x14xf32]) <- (-1x80x14x14xf32, 2xi64, 1xi32)
        split_19 = paddle._C_ops.split(add__17, constant_11, constant_1)

        # builtin.slice: (-1x40x14x14xf32) <- ([-1x40x14x14xf32, -1x40x14x14xf32])
        slice_44 = split_19[0]

        # pd_op.conv2d: (-1x240x14x14xf32) <- (-1x40x14x14xf32, 240x40x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(slice_44, parameter_184, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x40x14x14xf32) <- ([-1x40x14x14xf32, -1x40x14x14xf32])
        slice_45 = split_19[1]

        # pd_op.conv2d: (-1x240x14x14xf32) <- (-1x40x14x14xf32, 240x40x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(slice_45, parameter_185, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x240x14x14xf32, -1x240x14x14xf32]) <- (-1x240x14x14xf32, -1x240x14x14xf32)
        combine_19 = [conv2d_42, conv2d_43]

        # pd_op.concat: (-1x480x14x14xf32) <- ([-1x240x14x14xf32, -1x240x14x14xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_19, constant_1)

        # pd_op.batch_norm_: (-1x480x14x14xf32, 480xf32, 480xf32, xf32, xf32, None) <- (-1x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_19, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x14x14xf32) <- (-1x480x14x14xf32)
        swish_18 = paddle._C_ops.swish(batch_norm__162)

        # pd_op.split: ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32]) <- (-1x480x14x14xf32, 4xi64, 1xi32)
        split_20 = paddle._C_ops.split(swish_18, constant_12, constant_1)

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_46 = split_20[0]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x3x3xf32)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(slice_46, parameter_190, [1, 1], [1, 1], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_47 = split_20[1]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x5x5xf32)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(slice_47, parameter_191, [1, 1], [2, 2], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_48 = split_20[2]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x7x7xf32)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(slice_48, parameter_192, [1, 1], [3, 3], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_49 = split_20[3]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x9x9xf32)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(slice_49, parameter_193, [1, 1], [4, 4], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.combine: ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32]) <- (-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32)
        combine_20 = [depthwise_conv2d_22, depthwise_conv2d_23, depthwise_conv2d_24, depthwise_conv2d_25]

        # pd_op.concat: (-1x480x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_20, constant_1)

        # pd_op.batch_norm_: (-1x480x14x14xf32, 480xf32, 480xf32, xf32, xf32, None) <- (-1x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_20, parameter_194, parameter_195, parameter_196, parameter_197, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x14x14xf32) <- (-1x480x14x14xf32)
        swish_19 = paddle._C_ops.swish(batch_norm__168)

        # pd_op.pool2d: (-1x480x1x1xf32) <- (-1x480x14x14xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(swish_19, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x480x1x1xf32, 20x480x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(pool2d_6, parameter_198, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_44, parameter_199)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_20 = paddle._C_ops.swish(add__18)

        # pd_op.conv2d: (-1x480x1x1xf32) <- (-1x20x1x1xf32, 480x20x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(swish_20, parameter_200, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x480x1x1xf32) <- (-1x480x1x1xf32, 1x480x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_45, parameter_201)

        # pd_op.sigmoid_: (-1x480x1x1xf32) <- (-1x480x1x1xf32)
        sigmoid__6 = paddle._C_ops.sigmoid_(add__19)

        # pd_op.multiply_: (-1x480x14x14xf32) <- (-1x480x14x14xf32, -1x480x1x1xf32)
        multiply__6 = paddle._C_ops.multiply_(swish_19, sigmoid__6)

        # pd_op.split: ([-1x240x14x14xf32, -1x240x14x14xf32]) <- (-1x480x14x14xf32, 2xi64, 1xi32)
        split_21 = paddle._C_ops.split(multiply__6, constant_13, constant_1)

        # builtin.slice: (-1x240x14x14xf32) <- ([-1x240x14x14xf32, -1x240x14x14xf32])
        slice_50 = split_21[0]

        # pd_op.conv2d: (-1x40x14x14xf32) <- (-1x240x14x14xf32, 40x240x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(slice_50, parameter_202, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x240x14x14xf32) <- ([-1x240x14x14xf32, -1x240x14x14xf32])
        slice_51 = split_21[1]

        # pd_op.conv2d: (-1x40x14x14xf32) <- (-1x240x14x14xf32, 40x240x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(slice_51, parameter_203, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x40x14x14xf32, -1x40x14x14xf32]) <- (-1x40x14x14xf32, -1x40x14x14xf32)
        combine_21 = [conv2d_46, conv2d_47]

        # pd_op.concat: (-1x80x14x14xf32) <- ([-1x40x14x14xf32, -1x40x14x14xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_21, constant_1)

        # pd_op.batch_norm_: (-1x80x14x14xf32, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_21, parameter_204, parameter_205, parameter_206, parameter_207, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x14x14xf32) <- (-1x80x14x14xf32, -1x80x14x14xf32)
        add__20 = paddle._C_ops.add_(batch_norm__174, add__17)

        # pd_op.split: ([-1x40x14x14xf32, -1x40x14x14xf32]) <- (-1x80x14x14xf32, 2xi64, 1xi32)
        split_22 = paddle._C_ops.split(add__20, constant_11, constant_1)

        # builtin.slice: (-1x40x14x14xf32) <- ([-1x40x14x14xf32, -1x40x14x14xf32])
        slice_52 = split_22[0]

        # pd_op.conv2d: (-1x240x14x14xf32) <- (-1x40x14x14xf32, 240x40x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(slice_52, parameter_208, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x40x14x14xf32) <- ([-1x40x14x14xf32, -1x40x14x14xf32])
        slice_53 = split_22[1]

        # pd_op.conv2d: (-1x240x14x14xf32) <- (-1x40x14x14xf32, 240x40x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(slice_53, parameter_209, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x240x14x14xf32, -1x240x14x14xf32]) <- (-1x240x14x14xf32, -1x240x14x14xf32)
        combine_22 = [conv2d_48, conv2d_49]

        # pd_op.concat: (-1x480x14x14xf32) <- ([-1x240x14x14xf32, -1x240x14x14xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_22, constant_1)

        # pd_op.batch_norm_: (-1x480x14x14xf32, 480xf32, 480xf32, xf32, xf32, None) <- (-1x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_22, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x14x14xf32) <- (-1x480x14x14xf32)
        swish_21 = paddle._C_ops.swish(batch_norm__180)

        # pd_op.split: ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32]) <- (-1x480x14x14xf32, 4xi64, 1xi32)
        split_23 = paddle._C_ops.split(swish_21, constant_12, constant_1)

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_54 = split_23[0]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x3x3xf32)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(slice_54, parameter_214, [1, 1], [1, 1], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_55 = split_23[1]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x5x5xf32)
        depthwise_conv2d_27 = paddle._C_ops.depthwise_conv2d(slice_55, parameter_215, [1, 1], [2, 2], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_56 = split_23[2]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x7x7xf32)
        depthwise_conv2d_28 = paddle._C_ops.depthwise_conv2d(slice_56, parameter_216, [1, 1], [3, 3], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.slice: (-1x120x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32])
        slice_57 = split_23[3]

        # pd_op.depthwise_conv2d: (-1x120x14x14xf32) <- (-1x120x14x14xf32, 120x1x9x9xf32)
        depthwise_conv2d_29 = paddle._C_ops.depthwise_conv2d(slice_57, parameter_217, [1, 1], [4, 4], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # builtin.combine: ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32]) <- (-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32)
        combine_23 = [depthwise_conv2d_26, depthwise_conv2d_27, depthwise_conv2d_28, depthwise_conv2d_29]

        # pd_op.concat: (-1x480x14x14xf32) <- ([-1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32, -1x120x14x14xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_23, constant_1)

        # pd_op.batch_norm_: (-1x480x14x14xf32, 480xf32, 480xf32, xf32, xf32, None) <- (-1x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_23, parameter_218, parameter_219, parameter_220, parameter_221, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x14x14xf32) <- (-1x480x14x14xf32)
        swish_22 = paddle._C_ops.swish(batch_norm__186)

        # pd_op.pool2d: (-1x480x1x1xf32) <- (-1x480x14x14xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(swish_22, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x20x1x1xf32) <- (-1x480x1x1xf32, 20x480x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(pool2d_7, parameter_222, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x20x1x1xf32) <- (-1x20x1x1xf32, 1x20x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_50, parameter_223)

        # pd_op.swish: (-1x20x1x1xf32) <- (-1x20x1x1xf32)
        swish_23 = paddle._C_ops.swish(add__21)

        # pd_op.conv2d: (-1x480x1x1xf32) <- (-1x20x1x1xf32, 480x20x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(swish_23, parameter_224, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x480x1x1xf32) <- (-1x480x1x1xf32, 1x480x1x1xf32)
        add__22 = paddle._C_ops.add_(conv2d_51, parameter_225)

        # pd_op.sigmoid_: (-1x480x1x1xf32) <- (-1x480x1x1xf32)
        sigmoid__7 = paddle._C_ops.sigmoid_(add__22)

        # pd_op.multiply_: (-1x480x14x14xf32) <- (-1x480x14x14xf32, -1x480x1x1xf32)
        multiply__7 = paddle._C_ops.multiply_(swish_22, sigmoid__7)

        # pd_op.split: ([-1x240x14x14xf32, -1x240x14x14xf32]) <- (-1x480x14x14xf32, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(multiply__7, constant_13, constant_1)

        # builtin.slice: (-1x240x14x14xf32) <- ([-1x240x14x14xf32, -1x240x14x14xf32])
        slice_58 = split_24[0]

        # pd_op.conv2d: (-1x40x14x14xf32) <- (-1x240x14x14xf32, 40x240x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(slice_58, parameter_226, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x240x14x14xf32) <- ([-1x240x14x14xf32, -1x240x14x14xf32])
        slice_59 = split_24[1]

        # pd_op.conv2d: (-1x40x14x14xf32) <- (-1x240x14x14xf32, 40x240x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(slice_59, parameter_227, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x40x14x14xf32, -1x40x14x14xf32]) <- (-1x40x14x14xf32, -1x40x14x14xf32)
        combine_24 = [conv2d_52, conv2d_53]

        # pd_op.concat: (-1x80x14x14xf32) <- ([-1x40x14x14xf32, -1x40x14x14xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_24, constant_1)

        # pd_op.batch_norm_: (-1x80x14x14xf32, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_24, parameter_228, parameter_229, parameter_230, parameter_231, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x14x14xf32) <- (-1x80x14x14xf32, -1x80x14x14xf32)
        add__23 = paddle._C_ops.add_(batch_norm__192, add__20)

        # pd_op.conv2d: (-1x480x14x14xf32) <- (-1x80x14x14xf32, 480x80x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(add__23, parameter_232, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x480x14x14xf32, 480xf32, 480xf32, xf32, xf32, None) <- (-1x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_233, parameter_234, parameter_235, parameter_236, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x14x14xf32) <- (-1x480x14x14xf32)
        swish_24 = paddle._C_ops.swish(batch_norm__198)

        # pd_op.depthwise_conv2d: (-1x480x14x14xf32) <- (-1x480x14x14xf32, 480x1x3x3xf32)
        depthwise_conv2d_30 = paddle._C_ops.depthwise_conv2d(swish_24, parameter_237, [1, 1], [1, 1], 'EXPLICIT', 480, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x480x14x14xf32, 480xf32, 480xf32, xf32, xf32, None) <- (-1x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_30, parameter_238, parameter_239, parameter_240, parameter_241, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x480x14x14xf32) <- (-1x480x14x14xf32)
        swish_25 = paddle._C_ops.swish(batch_norm__204)

        # pd_op.pool2d: (-1x480x1x1xf32) <- (-1x480x14x14xf32, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(swish_25, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x40x1x1xf32) <- (-1x480x1x1xf32, 40x480x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(pool2d_8, parameter_242, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x40x1x1xf32) <- (-1x40x1x1xf32, 1x40x1x1xf32)
        add__24 = paddle._C_ops.add_(conv2d_55, parameter_243)

        # pd_op.swish: (-1x40x1x1xf32) <- (-1x40x1x1xf32)
        swish_26 = paddle._C_ops.swish(add__24)

        # pd_op.conv2d: (-1x480x1x1xf32) <- (-1x40x1x1xf32, 480x40x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(swish_26, parameter_244, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x480x1x1xf32) <- (-1x480x1x1xf32, 1x480x1x1xf32)
        add__25 = paddle._C_ops.add_(conv2d_56, parameter_245)

        # pd_op.sigmoid_: (-1x480x1x1xf32) <- (-1x480x1x1xf32)
        sigmoid__8 = paddle._C_ops.sigmoid_(add__25)

        # pd_op.multiply_: (-1x480x14x14xf32) <- (-1x480x14x14xf32, -1x480x1x1xf32)
        multiply__8 = paddle._C_ops.multiply_(swish_25, sigmoid__8)

        # pd_op.conv2d: (-1x120x14x14xf32) <- (-1x480x14x14xf32, 120x480x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(multiply__8, parameter_246, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x120x14x14xf32, 120xf32, 120xf32, xf32, xf32, None) <- (-1x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_247, parameter_248, parameter_249, parameter_250, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.split: ([-1x60x14x14xf32, -1x60x14x14xf32]) <- (-1x120x14x14xf32, 2xi64, 1xi32)
        split_25 = paddle._C_ops.split(batch_norm__210, constant_14, constant_1)

        # builtin.slice: (-1x60x14x14xf32) <- ([-1x60x14x14xf32, -1x60x14x14xf32])
        slice_60 = split_25[0]

        # pd_op.conv2d: (-1x180x14x14xf32) <- (-1x60x14x14xf32, 180x60x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(slice_60, parameter_251, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x60x14x14xf32) <- ([-1x60x14x14xf32, -1x60x14x14xf32])
        slice_61 = split_25[1]

        # pd_op.conv2d: (-1x180x14x14xf32) <- (-1x60x14x14xf32, 180x60x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(slice_61, parameter_252, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x180x14x14xf32, -1x180x14x14xf32]) <- (-1x180x14x14xf32, -1x180x14x14xf32)
        combine_25 = [conv2d_58, conv2d_59]

        # pd_op.concat: (-1x360x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_25, constant_1)

        # pd_op.batch_norm_: (-1x360x14x14xf32, 360xf32, 360xf32, xf32, xf32, None) <- (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_25, parameter_253, parameter_254, parameter_255, parameter_256, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x360x14x14xf32) <- (-1x360x14x14xf32)
        swish_27 = paddle._C_ops.swish(batch_norm__216)

        # pd_op.split: ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32]) <- (-1x360x14x14xf32, 4xi64, 1xi32)
        split_26 = paddle._C_ops.split(swish_27, constant_15, constant_1)

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_62 = split_26[0]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x3x3xf32)
        depthwise_conv2d_31 = paddle._C_ops.depthwise_conv2d(slice_62, parameter_257, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_63 = split_26[1]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x5x5xf32)
        depthwise_conv2d_32 = paddle._C_ops.depthwise_conv2d(slice_63, parameter_258, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_64 = split_26[2]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x7x7xf32)
        depthwise_conv2d_33 = paddle._C_ops.depthwise_conv2d(slice_64, parameter_259, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_65 = split_26[3]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x9x9xf32)
        depthwise_conv2d_34 = paddle._C_ops.depthwise_conv2d(slice_65, parameter_260, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.combine: ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32]) <- (-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32)
        combine_26 = [depthwise_conv2d_31, depthwise_conv2d_32, depthwise_conv2d_33, depthwise_conv2d_34]

        # pd_op.concat: (-1x360x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_26, constant_1)

        # pd_op.batch_norm_: (-1x360x14x14xf32, 360xf32, 360xf32, xf32, xf32, None) <- (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_26, parameter_261, parameter_262, parameter_263, parameter_264, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x360x14x14xf32) <- (-1x360x14x14xf32)
        swish_28 = paddle._C_ops.swish(batch_norm__222)

        # pd_op.pool2d: (-1x360x1x1xf32) <- (-1x360x14x14xf32, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(swish_28, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x60x1x1xf32) <- (-1x360x1x1xf32, 60x360x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(pool2d_9, parameter_265, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x60x1x1xf32) <- (-1x60x1x1xf32, 1x60x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_60, parameter_266)

        # pd_op.swish: (-1x60x1x1xf32) <- (-1x60x1x1xf32)
        swish_29 = paddle._C_ops.swish(add__26)

        # pd_op.conv2d: (-1x360x1x1xf32) <- (-1x60x1x1xf32, 360x60x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(swish_29, parameter_267, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x360x1x1xf32) <- (-1x360x1x1xf32, 1x360x1x1xf32)
        add__27 = paddle._C_ops.add_(conv2d_61, parameter_268)

        # pd_op.sigmoid_: (-1x360x1x1xf32) <- (-1x360x1x1xf32)
        sigmoid__9 = paddle._C_ops.sigmoid_(add__27)

        # pd_op.multiply_: (-1x360x14x14xf32) <- (-1x360x14x14xf32, -1x360x1x1xf32)
        multiply__9 = paddle._C_ops.multiply_(swish_28, sigmoid__9)

        # pd_op.split: ([-1x180x14x14xf32, -1x180x14x14xf32]) <- (-1x360x14x14xf32, 2xi64, 1xi32)
        split_27 = paddle._C_ops.split(multiply__9, constant_16, constant_1)

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32])
        slice_66 = split_27[0]

        # pd_op.conv2d: (-1x60x14x14xf32) <- (-1x180x14x14xf32, 60x180x1x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(slice_66, parameter_269, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32])
        slice_67 = split_27[1]

        # pd_op.conv2d: (-1x60x14x14xf32) <- (-1x180x14x14xf32, 60x180x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(slice_67, parameter_270, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x60x14x14xf32, -1x60x14x14xf32]) <- (-1x60x14x14xf32, -1x60x14x14xf32)
        combine_27 = [conv2d_62, conv2d_63]

        # pd_op.concat: (-1x120x14x14xf32) <- ([-1x60x14x14xf32, -1x60x14x14xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_27, constant_1)

        # pd_op.batch_norm_: (-1x120x14x14xf32, 120xf32, 120xf32, xf32, xf32, None) <- (-1x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_27, parameter_271, parameter_272, parameter_273, parameter_274, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x120x14x14xf32) <- (-1x120x14x14xf32, -1x120x14x14xf32)
        add__28 = paddle._C_ops.add_(batch_norm__228, batch_norm__210)

        # pd_op.split: ([-1x60x14x14xf32, -1x60x14x14xf32]) <- (-1x120x14x14xf32, 2xi64, 1xi32)
        split_28 = paddle._C_ops.split(add__28, constant_14, constant_1)

        # builtin.slice: (-1x60x14x14xf32) <- ([-1x60x14x14xf32, -1x60x14x14xf32])
        slice_68 = split_28[0]

        # pd_op.conv2d: (-1x180x14x14xf32) <- (-1x60x14x14xf32, 180x60x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(slice_68, parameter_275, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x60x14x14xf32) <- ([-1x60x14x14xf32, -1x60x14x14xf32])
        slice_69 = split_28[1]

        # pd_op.conv2d: (-1x180x14x14xf32) <- (-1x60x14x14xf32, 180x60x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(slice_69, parameter_276, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x180x14x14xf32, -1x180x14x14xf32]) <- (-1x180x14x14xf32, -1x180x14x14xf32)
        combine_28 = [conv2d_64, conv2d_65]

        # pd_op.concat: (-1x360x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_28, constant_1)

        # pd_op.batch_norm_: (-1x360x14x14xf32, 360xf32, 360xf32, xf32, xf32, None) <- (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_28, parameter_277, parameter_278, parameter_279, parameter_280, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x360x14x14xf32) <- (-1x360x14x14xf32)
        swish_30 = paddle._C_ops.swish(batch_norm__234)

        # pd_op.split: ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32]) <- (-1x360x14x14xf32, 4xi64, 1xi32)
        split_29 = paddle._C_ops.split(swish_30, constant_15, constant_1)

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_70 = split_29[0]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x3x3xf32)
        depthwise_conv2d_35 = paddle._C_ops.depthwise_conv2d(slice_70, parameter_281, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_71 = split_29[1]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x5x5xf32)
        depthwise_conv2d_36 = paddle._C_ops.depthwise_conv2d(slice_71, parameter_282, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_72 = split_29[2]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x7x7xf32)
        depthwise_conv2d_37 = paddle._C_ops.depthwise_conv2d(slice_72, parameter_283, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_73 = split_29[3]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x9x9xf32)
        depthwise_conv2d_38 = paddle._C_ops.depthwise_conv2d(slice_73, parameter_284, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.combine: ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32]) <- (-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32)
        combine_29 = [depthwise_conv2d_35, depthwise_conv2d_36, depthwise_conv2d_37, depthwise_conv2d_38]

        # pd_op.concat: (-1x360x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_29, constant_1)

        # pd_op.batch_norm_: (-1x360x14x14xf32, 360xf32, 360xf32, xf32, xf32, None) <- (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_29, parameter_285, parameter_286, parameter_287, parameter_288, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x360x14x14xf32) <- (-1x360x14x14xf32)
        swish_31 = paddle._C_ops.swish(batch_norm__240)

        # pd_op.pool2d: (-1x360x1x1xf32) <- (-1x360x14x14xf32, 2xi64)
        pool2d_10 = paddle._C_ops.pool2d(swish_31, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x60x1x1xf32) <- (-1x360x1x1xf32, 60x360x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(pool2d_10, parameter_289, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x60x1x1xf32) <- (-1x60x1x1xf32, 1x60x1x1xf32)
        add__29 = paddle._C_ops.add_(conv2d_66, parameter_290)

        # pd_op.swish: (-1x60x1x1xf32) <- (-1x60x1x1xf32)
        swish_32 = paddle._C_ops.swish(add__29)

        # pd_op.conv2d: (-1x360x1x1xf32) <- (-1x60x1x1xf32, 360x60x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(swish_32, parameter_291, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x360x1x1xf32) <- (-1x360x1x1xf32, 1x360x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_67, parameter_292)

        # pd_op.sigmoid_: (-1x360x1x1xf32) <- (-1x360x1x1xf32)
        sigmoid__10 = paddle._C_ops.sigmoid_(add__30)

        # pd_op.multiply_: (-1x360x14x14xf32) <- (-1x360x14x14xf32, -1x360x1x1xf32)
        multiply__10 = paddle._C_ops.multiply_(swish_31, sigmoid__10)

        # pd_op.split: ([-1x180x14x14xf32, -1x180x14x14xf32]) <- (-1x360x14x14xf32, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(multiply__10, constant_16, constant_1)

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32])
        slice_74 = split_30[0]

        # pd_op.conv2d: (-1x60x14x14xf32) <- (-1x180x14x14xf32, 60x180x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(slice_74, parameter_293, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32])
        slice_75 = split_30[1]

        # pd_op.conv2d: (-1x60x14x14xf32) <- (-1x180x14x14xf32, 60x180x1x1xf32)
        conv2d_69 = paddle._C_ops.conv2d(slice_75, parameter_294, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x60x14x14xf32, -1x60x14x14xf32]) <- (-1x60x14x14xf32, -1x60x14x14xf32)
        combine_30 = [conv2d_68, conv2d_69]

        # pd_op.concat: (-1x120x14x14xf32) <- ([-1x60x14x14xf32, -1x60x14x14xf32], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_30, constant_1)

        # pd_op.batch_norm_: (-1x120x14x14xf32, 120xf32, 120xf32, xf32, xf32, None) <- (-1x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_30, parameter_295, parameter_296, parameter_297, parameter_298, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x120x14x14xf32) <- (-1x120x14x14xf32, -1x120x14x14xf32)
        add__31 = paddle._C_ops.add_(batch_norm__246, add__28)

        # pd_op.split: ([-1x60x14x14xf32, -1x60x14x14xf32]) <- (-1x120x14x14xf32, 2xi64, 1xi32)
        split_31 = paddle._C_ops.split(add__31, constant_14, constant_1)

        # builtin.slice: (-1x60x14x14xf32) <- ([-1x60x14x14xf32, -1x60x14x14xf32])
        slice_76 = split_31[0]

        # pd_op.conv2d: (-1x180x14x14xf32) <- (-1x60x14x14xf32, 180x60x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(slice_76, parameter_299, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x60x14x14xf32) <- ([-1x60x14x14xf32, -1x60x14x14xf32])
        slice_77 = split_31[1]

        # pd_op.conv2d: (-1x180x14x14xf32) <- (-1x60x14x14xf32, 180x60x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(slice_77, parameter_300, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x180x14x14xf32, -1x180x14x14xf32]) <- (-1x180x14x14xf32, -1x180x14x14xf32)
        combine_31 = [conv2d_70, conv2d_71]

        # pd_op.concat: (-1x360x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_31, constant_1)

        # pd_op.batch_norm_: (-1x360x14x14xf32, 360xf32, 360xf32, xf32, xf32, None) <- (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_31, parameter_301, parameter_302, parameter_303, parameter_304, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x360x14x14xf32) <- (-1x360x14x14xf32)
        swish_33 = paddle._C_ops.swish(batch_norm__252)

        # pd_op.split: ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32]) <- (-1x360x14x14xf32, 4xi64, 1xi32)
        split_32 = paddle._C_ops.split(swish_33, constant_15, constant_1)

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_78 = split_32[0]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x3x3xf32)
        depthwise_conv2d_39 = paddle._C_ops.depthwise_conv2d(slice_78, parameter_305, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_79 = split_32[1]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x5x5xf32)
        depthwise_conv2d_40 = paddle._C_ops.depthwise_conv2d(slice_79, parameter_306, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_80 = split_32[2]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x7x7xf32)
        depthwise_conv2d_41 = paddle._C_ops.depthwise_conv2d(slice_80, parameter_307, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.slice: (-1x90x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32])
        slice_81 = split_32[3]

        # pd_op.depthwise_conv2d: (-1x90x14x14xf32) <- (-1x90x14x14xf32, 90x1x9x9xf32)
        depthwise_conv2d_42 = paddle._C_ops.depthwise_conv2d(slice_81, parameter_308, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

        # builtin.combine: ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32]) <- (-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32)
        combine_32 = [depthwise_conv2d_39, depthwise_conv2d_40, depthwise_conv2d_41, depthwise_conv2d_42]

        # pd_op.concat: (-1x360x14x14xf32) <- ([-1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32, -1x90x14x14xf32], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_32, constant_1)

        # pd_op.batch_norm_: (-1x360x14x14xf32, 360xf32, 360xf32, xf32, xf32, None) <- (-1x360x14x14xf32, 360xf32, 360xf32, 360xf32, 360xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_32, parameter_309, parameter_310, parameter_311, parameter_312, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x360x14x14xf32) <- (-1x360x14x14xf32)
        swish_34 = paddle._C_ops.swish(batch_norm__258)

        # pd_op.pool2d: (-1x360x1x1xf32) <- (-1x360x14x14xf32, 2xi64)
        pool2d_11 = paddle._C_ops.pool2d(swish_34, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x60x1x1xf32) <- (-1x360x1x1xf32, 60x360x1x1xf32)
        conv2d_72 = paddle._C_ops.conv2d(pool2d_11, parameter_313, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x60x1x1xf32) <- (-1x60x1x1xf32, 1x60x1x1xf32)
        add__32 = paddle._C_ops.add_(conv2d_72, parameter_314)

        # pd_op.swish: (-1x60x1x1xf32) <- (-1x60x1x1xf32)
        swish_35 = paddle._C_ops.swish(add__32)

        # pd_op.conv2d: (-1x360x1x1xf32) <- (-1x60x1x1xf32, 360x60x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(swish_35, parameter_315, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x360x1x1xf32) <- (-1x360x1x1xf32, 1x360x1x1xf32)
        add__33 = paddle._C_ops.add_(conv2d_73, parameter_316)

        # pd_op.sigmoid_: (-1x360x1x1xf32) <- (-1x360x1x1xf32)
        sigmoid__11 = paddle._C_ops.sigmoid_(add__33)

        # pd_op.multiply_: (-1x360x14x14xf32) <- (-1x360x14x14xf32, -1x360x1x1xf32)
        multiply__11 = paddle._C_ops.multiply_(swish_34, sigmoid__11)

        # pd_op.split: ([-1x180x14x14xf32, -1x180x14x14xf32]) <- (-1x360x14x14xf32, 2xi64, 1xi32)
        split_33 = paddle._C_ops.split(multiply__11, constant_16, constant_1)

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32])
        slice_82 = split_33[0]

        # pd_op.conv2d: (-1x60x14x14xf32) <- (-1x180x14x14xf32, 60x180x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(slice_82, parameter_317, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32])
        slice_83 = split_33[1]

        # pd_op.conv2d: (-1x60x14x14xf32) <- (-1x180x14x14xf32, 60x180x1x1xf32)
        conv2d_75 = paddle._C_ops.conv2d(slice_83, parameter_318, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x60x14x14xf32, -1x60x14x14xf32]) <- (-1x60x14x14xf32, -1x60x14x14xf32)
        combine_33 = [conv2d_74, conv2d_75]

        # pd_op.concat: (-1x120x14x14xf32) <- ([-1x60x14x14xf32, -1x60x14x14xf32], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_33, constant_1)

        # pd_op.batch_norm_: (-1x120x14x14xf32, 120xf32, 120xf32, xf32, xf32, None) <- (-1x120x14x14xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_33, parameter_319, parameter_320, parameter_321, parameter_322, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x120x14x14xf32) <- (-1x120x14x14xf32, -1x120x14x14xf32)
        add__34 = paddle._C_ops.add_(batch_norm__264, add__31)

        # pd_op.conv2d: (-1x720x14x14xf32) <- (-1x120x14x14xf32, 720x120x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(add__34, parameter_323, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x720x14x14xf32, 720xf32, 720xf32, xf32, xf32, None) <- (-1x720x14x14xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_324, parameter_325, parameter_326, parameter_327, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x720x14x14xf32) <- (-1x720x14x14xf32)
        swish_36 = paddle._C_ops.swish(batch_norm__270)

        # pd_op.split: ([-1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32]) <- (-1x720x14x14xf32, 4xi64, 1xi32)
        split_34 = paddle._C_ops.split(swish_36, constant_17, constant_1)

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32])
        slice_84 = split_34[0]

        # pd_op.depthwise_conv2d: (-1x180x7x7xf32) <- (-1x180x14x14xf32, 180x1x3x3xf32)
        depthwise_conv2d_43 = paddle._C_ops.depthwise_conv2d(slice_84, parameter_328, [2, 2], [1, 1], 'EXPLICIT', 180, [1, 1], 'NCHW')

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32])
        slice_85 = split_34[1]

        # pd_op.depthwise_conv2d: (-1x180x7x7xf32) <- (-1x180x14x14xf32, 180x1x5x5xf32)
        depthwise_conv2d_44 = paddle._C_ops.depthwise_conv2d(slice_85, parameter_329, [2, 2], [2, 2], 'EXPLICIT', 180, [1, 1], 'NCHW')

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32])
        slice_86 = split_34[2]

        # pd_op.depthwise_conv2d: (-1x180x7x7xf32) <- (-1x180x14x14xf32, 180x1x7x7xf32)
        depthwise_conv2d_45 = paddle._C_ops.depthwise_conv2d(slice_86, parameter_330, [2, 2], [3, 3], 'EXPLICIT', 180, [1, 1], 'NCHW')

        # builtin.slice: (-1x180x14x14xf32) <- ([-1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32, -1x180x14x14xf32])
        slice_87 = split_34[3]

        # pd_op.depthwise_conv2d: (-1x180x7x7xf32) <- (-1x180x14x14xf32, 180x1x9x9xf32)
        depthwise_conv2d_46 = paddle._C_ops.depthwise_conv2d(slice_87, parameter_331, [2, 2], [4, 4], 'EXPLICIT', 180, [1, 1], 'NCHW')

        # builtin.combine: ([-1x180x7x7xf32, -1x180x7x7xf32, -1x180x7x7xf32, -1x180x7x7xf32]) <- (-1x180x7x7xf32, -1x180x7x7xf32, -1x180x7x7xf32, -1x180x7x7xf32)
        combine_34 = [depthwise_conv2d_43, depthwise_conv2d_44, depthwise_conv2d_45, depthwise_conv2d_46]

        # pd_op.concat: (-1x720x7x7xf32) <- ([-1x180x7x7xf32, -1x180x7x7xf32, -1x180x7x7xf32, -1x180x7x7xf32], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_34, constant_1)

        # pd_op.batch_norm_: (-1x720x7x7xf32, 720xf32, 720xf32, xf32, xf32, None) <- (-1x720x7x7xf32, 720xf32, 720xf32, 720xf32, 720xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_34, parameter_332, parameter_333, parameter_334, parameter_335, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x720x7x7xf32) <- (-1x720x7x7xf32)
        swish_37 = paddle._C_ops.swish(batch_norm__276)

        # pd_op.pool2d: (-1x720x1x1xf32) <- (-1x720x7x7xf32, 2xi64)
        pool2d_12 = paddle._C_ops.pool2d(swish_37, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x60x1x1xf32) <- (-1x720x1x1xf32, 60x720x1x1xf32)
        conv2d_77 = paddle._C_ops.conv2d(pool2d_12, parameter_336, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x60x1x1xf32) <- (-1x60x1x1xf32, 1x60x1x1xf32)
        add__35 = paddle._C_ops.add_(conv2d_77, parameter_337)

        # pd_op.swish: (-1x60x1x1xf32) <- (-1x60x1x1xf32)
        swish_38 = paddle._C_ops.swish(add__35)

        # pd_op.conv2d: (-1x720x1x1xf32) <- (-1x60x1x1xf32, 720x60x1x1xf32)
        conv2d_78 = paddle._C_ops.conv2d(swish_38, parameter_338, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x720x1x1xf32) <- (-1x720x1x1xf32, 1x720x1x1xf32)
        add__36 = paddle._C_ops.add_(conv2d_78, parameter_339)

        # pd_op.sigmoid_: (-1x720x1x1xf32) <- (-1x720x1x1xf32)
        sigmoid__12 = paddle._C_ops.sigmoid_(add__36)

        # pd_op.multiply_: (-1x720x7x7xf32) <- (-1x720x7x7xf32, -1x720x1x1xf32)
        multiply__12 = paddle._C_ops.multiply_(swish_37, sigmoid__12)

        # pd_op.conv2d: (-1x200x7x7xf32) <- (-1x720x7x7xf32, 200x720x1x1xf32)
        conv2d_79 = paddle._C_ops.conv2d(multiply__12, parameter_340, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x200x7x7xf32, 200xf32, 200xf32, xf32, xf32, None) <- (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_341, parameter_342, parameter_343, parameter_344, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1200x7x7xf32) <- (-1x200x7x7xf32, 1200x200x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(batch_norm__282, parameter_345, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, xf32, xf32, None) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_346, parameter_347, parameter_348, parameter_349, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        swish_39 = paddle._C_ops.swish(batch_norm__288)

        # pd_op.split: ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32]) <- (-1x1200x7x7xf32, 4xi64, 1xi32)
        split_35 = paddle._C_ops.split(swish_39, constant_18, constant_1)

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_88 = split_35[0]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x3x3xf32)
        depthwise_conv2d_47 = paddle._C_ops.depthwise_conv2d(slice_88, parameter_350, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_89 = split_35[1]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x5x5xf32)
        depthwise_conv2d_48 = paddle._C_ops.depthwise_conv2d(slice_89, parameter_351, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_90 = split_35[2]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x7x7xf32)
        depthwise_conv2d_49 = paddle._C_ops.depthwise_conv2d(slice_90, parameter_352, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_91 = split_35[3]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x9x9xf32)
        depthwise_conv2d_50 = paddle._C_ops.depthwise_conv2d(slice_91, parameter_353, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.combine: ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32]) <- (-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32)
        combine_35 = [depthwise_conv2d_47, depthwise_conv2d_48, depthwise_conv2d_49, depthwise_conv2d_50]

        # pd_op.concat: (-1x1200x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_35, constant_1)

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, xf32, xf32, None) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_35, parameter_354, parameter_355, parameter_356, parameter_357, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        swish_40 = paddle._C_ops.swish(batch_norm__294)

        # pd_op.pool2d: (-1x1200x1x1xf32) <- (-1x1200x7x7xf32, 2xi64)
        pool2d_13 = paddle._C_ops.pool2d(swish_40, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x100x1x1xf32) <- (-1x1200x1x1xf32, 100x1200x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(pool2d_13, parameter_358, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x100x1x1xf32) <- (-1x100x1x1xf32, 1x100x1x1xf32)
        add__37 = paddle._C_ops.add_(conv2d_81, parameter_359)

        # pd_op.swish: (-1x100x1x1xf32) <- (-1x100x1x1xf32)
        swish_41 = paddle._C_ops.swish(add__37)

        # pd_op.conv2d: (-1x1200x1x1xf32) <- (-1x100x1x1xf32, 1200x100x1x1xf32)
        conv2d_82 = paddle._C_ops.conv2d(swish_41, parameter_360, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32, 1x1200x1x1xf32)
        add__38 = paddle._C_ops.add_(conv2d_82, parameter_361)

        # pd_op.sigmoid_: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32)
        sigmoid__13 = paddle._C_ops.sigmoid_(add__38)

        # pd_op.multiply_: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32, -1x1200x1x1xf32)
        multiply__13 = paddle._C_ops.multiply_(swish_40, sigmoid__13)

        # pd_op.split: ([-1x600x7x7xf32, -1x600x7x7xf32]) <- (-1x1200x7x7xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(multiply__13, constant_19, constant_1)

        # builtin.slice: (-1x600x7x7xf32) <- ([-1x600x7x7xf32, -1x600x7x7xf32])
        slice_92 = split_36[0]

        # pd_op.conv2d: (-1x100x7x7xf32) <- (-1x600x7x7xf32, 100x600x1x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(slice_92, parameter_362, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x600x7x7xf32) <- ([-1x600x7x7xf32, -1x600x7x7xf32])
        slice_93 = split_36[1]

        # pd_op.conv2d: (-1x100x7x7xf32) <- (-1x600x7x7xf32, 100x600x1x1xf32)
        conv2d_84 = paddle._C_ops.conv2d(slice_93, parameter_363, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x100x7x7xf32, -1x100x7x7xf32]) <- (-1x100x7x7xf32, -1x100x7x7xf32)
        combine_36 = [conv2d_83, conv2d_84]

        # pd_op.concat: (-1x200x7x7xf32) <- ([-1x100x7x7xf32, -1x100x7x7xf32], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_36, constant_1)

        # pd_op.batch_norm_: (-1x200x7x7xf32, 200xf32, 200xf32, xf32, xf32, None) <- (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_36, parameter_364, parameter_365, parameter_366, parameter_367, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x200x7x7xf32) <- (-1x200x7x7xf32, -1x200x7x7xf32)
        add__39 = paddle._C_ops.add_(batch_norm__300, batch_norm__282)

        # pd_op.conv2d: (-1x1200x7x7xf32) <- (-1x200x7x7xf32, 1200x200x1x1xf32)
        conv2d_85 = paddle._C_ops.conv2d(add__39, parameter_368, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, xf32, xf32, None) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_369, parameter_370, parameter_371, parameter_372, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        swish_42 = paddle._C_ops.swish(batch_norm__306)

        # pd_op.split: ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32]) <- (-1x1200x7x7xf32, 4xi64, 1xi32)
        split_37 = paddle._C_ops.split(swish_42, constant_18, constant_1)

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_94 = split_37[0]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x3x3xf32)
        depthwise_conv2d_51 = paddle._C_ops.depthwise_conv2d(slice_94, parameter_373, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_95 = split_37[1]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x5x5xf32)
        depthwise_conv2d_52 = paddle._C_ops.depthwise_conv2d(slice_95, parameter_374, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_96 = split_37[2]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x7x7xf32)
        depthwise_conv2d_53 = paddle._C_ops.depthwise_conv2d(slice_96, parameter_375, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_97 = split_37[3]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x9x9xf32)
        depthwise_conv2d_54 = paddle._C_ops.depthwise_conv2d(slice_97, parameter_376, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.combine: ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32]) <- (-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32)
        combine_37 = [depthwise_conv2d_51, depthwise_conv2d_52, depthwise_conv2d_53, depthwise_conv2d_54]

        # pd_op.concat: (-1x1200x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_37, constant_1)

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, xf32, xf32, None) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_37, parameter_377, parameter_378, parameter_379, parameter_380, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        swish_43 = paddle._C_ops.swish(batch_norm__312)

        # pd_op.pool2d: (-1x1200x1x1xf32) <- (-1x1200x7x7xf32, 2xi64)
        pool2d_14 = paddle._C_ops.pool2d(swish_43, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x100x1x1xf32) <- (-1x1200x1x1xf32, 100x1200x1x1xf32)
        conv2d_86 = paddle._C_ops.conv2d(pool2d_14, parameter_381, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x100x1x1xf32) <- (-1x100x1x1xf32, 1x100x1x1xf32)
        add__40 = paddle._C_ops.add_(conv2d_86, parameter_382)

        # pd_op.swish: (-1x100x1x1xf32) <- (-1x100x1x1xf32)
        swish_44 = paddle._C_ops.swish(add__40)

        # pd_op.conv2d: (-1x1200x1x1xf32) <- (-1x100x1x1xf32, 1200x100x1x1xf32)
        conv2d_87 = paddle._C_ops.conv2d(swish_44, parameter_383, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32, 1x1200x1x1xf32)
        add__41 = paddle._C_ops.add_(conv2d_87, parameter_384)

        # pd_op.sigmoid_: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32)
        sigmoid__14 = paddle._C_ops.sigmoid_(add__41)

        # pd_op.multiply_: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32, -1x1200x1x1xf32)
        multiply__14 = paddle._C_ops.multiply_(swish_43, sigmoid__14)

        # pd_op.split: ([-1x600x7x7xf32, -1x600x7x7xf32]) <- (-1x1200x7x7xf32, 2xi64, 1xi32)
        split_38 = paddle._C_ops.split(multiply__14, constant_19, constant_1)

        # builtin.slice: (-1x600x7x7xf32) <- ([-1x600x7x7xf32, -1x600x7x7xf32])
        slice_98 = split_38[0]

        # pd_op.conv2d: (-1x100x7x7xf32) <- (-1x600x7x7xf32, 100x600x1x1xf32)
        conv2d_88 = paddle._C_ops.conv2d(slice_98, parameter_385, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x600x7x7xf32) <- ([-1x600x7x7xf32, -1x600x7x7xf32])
        slice_99 = split_38[1]

        # pd_op.conv2d: (-1x100x7x7xf32) <- (-1x600x7x7xf32, 100x600x1x1xf32)
        conv2d_89 = paddle._C_ops.conv2d(slice_99, parameter_386, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x100x7x7xf32, -1x100x7x7xf32]) <- (-1x100x7x7xf32, -1x100x7x7xf32)
        combine_38 = [conv2d_88, conv2d_89]

        # pd_op.concat: (-1x200x7x7xf32) <- ([-1x100x7x7xf32, -1x100x7x7xf32], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_38, constant_1)

        # pd_op.batch_norm_: (-1x200x7x7xf32, 200xf32, 200xf32, xf32, xf32, None) <- (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_38, parameter_387, parameter_388, parameter_389, parameter_390, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x200x7x7xf32) <- (-1x200x7x7xf32, -1x200x7x7xf32)
        add__42 = paddle._C_ops.add_(batch_norm__318, add__39)

        # pd_op.conv2d: (-1x1200x7x7xf32) <- (-1x200x7x7xf32, 1200x200x1x1xf32)
        conv2d_90 = paddle._C_ops.conv2d(add__42, parameter_391, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, xf32, xf32, None) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_90, parameter_392, parameter_393, parameter_394, parameter_395, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        swish_45 = paddle._C_ops.swish(batch_norm__324)

        # pd_op.split: ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32]) <- (-1x1200x7x7xf32, 4xi64, 1xi32)
        split_39 = paddle._C_ops.split(swish_45, constant_18, constant_1)

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_100 = split_39[0]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x3x3xf32)
        depthwise_conv2d_55 = paddle._C_ops.depthwise_conv2d(slice_100, parameter_396, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_101 = split_39[1]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x5x5xf32)
        depthwise_conv2d_56 = paddle._C_ops.depthwise_conv2d(slice_101, parameter_397, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_102 = split_39[2]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x7x7xf32)
        depthwise_conv2d_57 = paddle._C_ops.depthwise_conv2d(slice_102, parameter_398, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.slice: (-1x300x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32])
        slice_103 = split_39[3]

        # pd_op.depthwise_conv2d: (-1x300x7x7xf32) <- (-1x300x7x7xf32, 300x1x9x9xf32)
        depthwise_conv2d_58 = paddle._C_ops.depthwise_conv2d(slice_103, parameter_399, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

        # builtin.combine: ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32]) <- (-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32)
        combine_39 = [depthwise_conv2d_55, depthwise_conv2d_56, depthwise_conv2d_57, depthwise_conv2d_58]

        # pd_op.concat: (-1x1200x7x7xf32) <- ([-1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32, -1x300x7x7xf32], 1xi32)
        concat_39 = paddle._C_ops.concat(combine_39, constant_1)

        # pd_op.batch_norm_: (-1x1200x7x7xf32, 1200xf32, 1200xf32, xf32, xf32, None) <- (-1x1200x7x7xf32, 1200xf32, 1200xf32, 1200xf32, 1200xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_39, parameter_400, parameter_401, parameter_402, parameter_403, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32)
        swish_46 = paddle._C_ops.swish(batch_norm__330)

        # pd_op.pool2d: (-1x1200x1x1xf32) <- (-1x1200x7x7xf32, 2xi64)
        pool2d_15 = paddle._C_ops.pool2d(swish_46, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x100x1x1xf32) <- (-1x1200x1x1xf32, 100x1200x1x1xf32)
        conv2d_91 = paddle._C_ops.conv2d(pool2d_15, parameter_404, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x100x1x1xf32) <- (-1x100x1x1xf32, 1x100x1x1xf32)
        add__43 = paddle._C_ops.add_(conv2d_91, parameter_405)

        # pd_op.swish: (-1x100x1x1xf32) <- (-1x100x1x1xf32)
        swish_47 = paddle._C_ops.swish(add__43)

        # pd_op.conv2d: (-1x1200x1x1xf32) <- (-1x100x1x1xf32, 1200x100x1x1xf32)
        conv2d_92 = paddle._C_ops.conv2d(swish_47, parameter_406, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32, 1x1200x1x1xf32)
        add__44 = paddle._C_ops.add_(conv2d_92, parameter_407)

        # pd_op.sigmoid_: (-1x1200x1x1xf32) <- (-1x1200x1x1xf32)
        sigmoid__15 = paddle._C_ops.sigmoid_(add__44)

        # pd_op.multiply_: (-1x1200x7x7xf32) <- (-1x1200x7x7xf32, -1x1200x1x1xf32)
        multiply__15 = paddle._C_ops.multiply_(swish_46, sigmoid__15)

        # pd_op.split: ([-1x600x7x7xf32, -1x600x7x7xf32]) <- (-1x1200x7x7xf32, 2xi64, 1xi32)
        split_40 = paddle._C_ops.split(multiply__15, constant_19, constant_1)

        # builtin.slice: (-1x600x7x7xf32) <- ([-1x600x7x7xf32, -1x600x7x7xf32])
        slice_104 = split_40[0]

        # pd_op.conv2d: (-1x100x7x7xf32) <- (-1x600x7x7xf32, 100x600x1x1xf32)
        conv2d_93 = paddle._C_ops.conv2d(slice_104, parameter_408, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.slice: (-1x600x7x7xf32) <- ([-1x600x7x7xf32, -1x600x7x7xf32])
        slice_105 = split_40[1]

        # pd_op.conv2d: (-1x100x7x7xf32) <- (-1x600x7x7xf32, 100x600x1x1xf32)
        conv2d_94 = paddle._C_ops.conv2d(slice_105, parameter_409, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([-1x100x7x7xf32, -1x100x7x7xf32]) <- (-1x100x7x7xf32, -1x100x7x7xf32)
        combine_40 = [conv2d_93, conv2d_94]

        # pd_op.concat: (-1x200x7x7xf32) <- ([-1x100x7x7xf32, -1x100x7x7xf32], 1xi32)
        concat_40 = paddle._C_ops.concat(combine_40, constant_1)

        # pd_op.batch_norm_: (-1x200x7x7xf32, 200xf32, 200xf32, xf32, xf32, None) <- (-1x200x7x7xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_40, parameter_410, parameter_411, parameter_412, parameter_413, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x200x7x7xf32) <- (-1x200x7x7xf32, -1x200x7x7xf32)
        add__45 = paddle._C_ops.add_(batch_norm__336, add__42)

        # pd_op.conv2d: (-1x1536x7x7xf32) <- (-1x200x7x7xf32, 1536x200x1x1xf32)
        conv2d_95 = paddle._C_ops.conv2d(add__45, parameter_414, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1536x7x7xf32, 1536xf32, 1536xf32, xf32, xf32, None) <- (-1x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_415, parameter_416, parameter_417, parameter_418, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1536x7x7xf32) <- (-1x1536x7x7xf32)
        relu__6 = paddle._C_ops.relu_(batch_norm__342)

        # pd_op.pool2d: (-1x1536x1x1xf32) <- (-1x1536x7x7xf32, 2xi64)
        pool2d_16 = paddle._C_ops.pool2d(relu__6, constant_20, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.shape: (4xi32) <- (-1x1536x1x1xf32)
        shape_0 = paddle._C_ops.shape(pool2d_16)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(shape_0, [0], constant_21, constant_22, [1], [0])

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_41 = [slice_106, constant_23]

        # pd_op.reshape_: (-1x1536xf32, 0x-1x1536x1x1xf32) <- (-1x1536x1x1xf32, [1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(pool2d_16, combine_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf32) <- (-1x1536xf32, 1536x1000xf32)
        matmul_0 = paddle.matmul(reshape__0, parameter_419, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__46 = paddle._C_ops.add_(matmul_0, parameter_420)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__0 = paddle._C_ops.softmax_(add__46, -1)
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

    def forward(self, constant_23, constant_22, constant_21, constant_20, parameter_407, parameter_405, parameter_384, parameter_382, constant_19, parameter_361, parameter_359, constant_18, parameter_339, parameter_337, constant_17, parameter_316, parameter_314, parameter_292, parameter_290, constant_16, parameter_268, parameter_266, constant_15, constant_14, parameter_245, parameter_243, parameter_225, parameter_223, parameter_201, parameter_199, constant_13, parameter_177, parameter_175, constant_12, constant_11, parameter_154, parameter_152, constant_10, parameter_132, parameter_130, parameter_110, parameter_108, parameter_88, parameter_86, constant_9, constant_8, parameter_67, parameter_65, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_16, parameter_20, parameter_17, parameter_19, parameter_18, parameter_21, parameter_22, parameter_23, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_55, parameter_52, parameter_54, parameter_53, parameter_56, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_66, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_87, parameter_89, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_109, parameter_111, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_118, parameter_122, parameter_119, parameter_121, parameter_120, parameter_123, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_131, parameter_133, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_145, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_153, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_167, parameter_168, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_176, parameter_178, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_191, parameter_192, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_200, parameter_202, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_215, parameter_216, parameter_217, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_224, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_236, parameter_233, parameter_235, parameter_234, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_244, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_258, parameter_259, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_267, parameter_269, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_276, parameter_280, parameter_277, parameter_279, parameter_278, parameter_281, parameter_282, parameter_283, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_291, parameter_293, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_306, parameter_307, parameter_308, parameter_312, parameter_309, parameter_311, parameter_310, parameter_313, parameter_315, parameter_317, parameter_318, parameter_322, parameter_319, parameter_321, parameter_320, parameter_323, parameter_327, parameter_324, parameter_326, parameter_325, parameter_328, parameter_329, parameter_330, parameter_331, parameter_335, parameter_332, parameter_334, parameter_333, parameter_336, parameter_338, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_351, parameter_352, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_360, parameter_362, parameter_363, parameter_367, parameter_364, parameter_366, parameter_365, parameter_368, parameter_372, parameter_369, parameter_371, parameter_370, parameter_373, parameter_374, parameter_375, parameter_376, parameter_380, parameter_377, parameter_379, parameter_378, parameter_381, parameter_383, parameter_385, parameter_386, parameter_390, parameter_387, parameter_389, parameter_388, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_397, parameter_398, parameter_399, parameter_403, parameter_400, parameter_402, parameter_401, parameter_404, parameter_406, parameter_408, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_420, feed_0):
        return self.builtin_module_1934_0_0(constant_23, constant_22, constant_21, constant_20, parameter_407, parameter_405, parameter_384, parameter_382, constant_19, parameter_361, parameter_359, constant_18, parameter_339, parameter_337, constant_17, parameter_316, parameter_314, parameter_292, parameter_290, constant_16, parameter_268, parameter_266, constant_15, constant_14, parameter_245, parameter_243, parameter_225, parameter_223, parameter_201, parameter_199, constant_13, parameter_177, parameter_175, constant_12, constant_11, parameter_154, parameter_152, constant_10, parameter_132, parameter_130, parameter_110, parameter_108, parameter_88, parameter_86, constant_9, constant_8, parameter_67, parameter_65, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_16, parameter_20, parameter_17, parameter_19, parameter_18, parameter_21, parameter_22, parameter_23, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_55, parameter_52, parameter_54, parameter_53, parameter_56, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_66, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_87, parameter_89, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_109, parameter_111, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_118, parameter_122, parameter_119, parameter_121, parameter_120, parameter_123, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_131, parameter_133, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_145, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_153, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_167, parameter_168, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_176, parameter_178, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_191, parameter_192, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_200, parameter_202, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_215, parameter_216, parameter_217, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_224, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_236, parameter_233, parameter_235, parameter_234, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_244, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_258, parameter_259, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_267, parameter_269, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_276, parameter_280, parameter_277, parameter_279, parameter_278, parameter_281, parameter_282, parameter_283, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_291, parameter_293, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_306, parameter_307, parameter_308, parameter_312, parameter_309, parameter_311, parameter_310, parameter_313, parameter_315, parameter_317, parameter_318, parameter_322, parameter_319, parameter_321, parameter_320, parameter_323, parameter_327, parameter_324, parameter_326, parameter_325, parameter_328, parameter_329, parameter_330, parameter_331, parameter_335, parameter_332, parameter_334, parameter_333, parameter_336, parameter_338, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_351, parameter_352, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_360, parameter_362, parameter_363, parameter_367, parameter_364, parameter_366, parameter_365, parameter_368, parameter_372, parameter_369, parameter_371, parameter_370, parameter_373, parameter_374, parameter_375, parameter_376, parameter_380, parameter_377, parameter_379, parameter_378, parameter_381, parameter_383, parameter_385, parameter_386, parameter_390, parameter_387, parameter_389, parameter_388, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_397, parameter_398, parameter_399, parameter_403, parameter_400, parameter_402, parameter_401, parameter_404, parameter_406, parameter_408, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_420, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1934_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_23
            paddle.to_tensor([1536], dtype='int32').reshape([1]),
            # constant_22
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_21
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # constant_20
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            # parameter_407
            paddle.uniform([1, 1200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([1, 1200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_19
            paddle.to_tensor([600, 600], dtype='int64').reshape([2]),
            # parameter_361
            paddle.uniform([1, 1200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_18
            paddle.to_tensor([300, 300, 300, 300], dtype='int64').reshape([4]),
            # parameter_339
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_17
            paddle.to_tensor([180, 180, 180, 180], dtype='int64').reshape([4]),
            # parameter_316
            paddle.uniform([1, 360, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([1, 360, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_16
            paddle.to_tensor([180, 180], dtype='int64').reshape([2]),
            # parameter_268
            paddle.uniform([1, 360, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_15
            paddle.to_tensor([90, 90, 90, 90], dtype='int64').reshape([4]),
            # constant_14
            paddle.to_tensor([60, 60], dtype='int64').reshape([2]),
            # parameter_245
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_13
            paddle.to_tensor([240, 240], dtype='int64').reshape([2]),
            # parameter_177
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_12
            paddle.to_tensor([120, 120, 120, 120], dtype='int64').reshape([4]),
            # constant_11
            paddle.to_tensor([40, 40], dtype='int64').reshape([2]),
            # parameter_154
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([1, 10, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_10
            paddle.to_tensor([80, 80, 80], dtype='int64').reshape([3]),
            # parameter_132
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([1, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_9
            paddle.to_tensor([120, 120], dtype='int64').reshape([2]),
            # constant_8
            paddle.to_tensor([20, 20], dtype='int64').reshape([2]),
            # parameter_67
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([1, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_7
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_6
            paddle.to_tensor([48, 48, 48, 48], dtype='int64').reshape([4]),
            # constant_5
            paddle.to_tensor([48, 48], dtype='int64').reshape([2]),
            # constant_4
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            # constant_3
            paddle.to_tensor([72, 72], dtype='int64').reshape([2]),
            # constant_2
            paddle.to_tensor([48, 48, 48], dtype='int64').reshape([3]),
            # constant_1
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_0
            paddle.to_tensor([12, 12], dtype='int64').reshape([2]),
            # parameter_0
            paddle.uniform([24, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([24, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([72, 12, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([72, 12, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([48, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([16, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([16, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([48, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([48, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([16, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([16, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([192, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([48, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([48, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([16, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([192, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([40, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([120, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([120, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([20, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([240, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([120, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([120, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([20, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([240, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([120, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([120, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([20, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([240, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([80, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([80, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([80, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([80, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([120, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([120, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([120, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([120, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([40, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([40, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([120, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([120, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([120, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([120, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([40, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([40, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([120, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([120, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([120, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([120, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([40, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([40, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([480, 80, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([480, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([40, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([480, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([60, 360, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([360, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([60, 360, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([360, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([360], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([60, 360, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([360, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([720, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([180, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([180, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([180, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([180, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([60, 720, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([720, 60, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([200, 720, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([1200, 200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([100, 1200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([1200, 100, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([1200, 200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([100, 1200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([1200, 100, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([1200, 200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([1200], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([100, 1200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([1200, 100, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([1536, 200, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_23
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_22
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_21
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_20
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_407
            paddle.static.InputSpec(shape=[1, 1200, 1, 1], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[1, 1200, 1, 1], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
            # constant_19
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_361
            paddle.static.InputSpec(shape=[1, 1200, 1, 1], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
            # constant_18
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # parameter_339
            paddle.static.InputSpec(shape=[1, 720, 1, 1], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[1, 60, 1, 1], dtype='float32'),
            # constant_17
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # parameter_316
            paddle.static.InputSpec(shape=[1, 360, 1, 1], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[1, 60, 1, 1], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[1, 360, 1, 1], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[1, 60, 1, 1], dtype='float32'),
            # constant_16
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_268
            paddle.static.InputSpec(shape=[1, 360, 1, 1], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[1, 60, 1, 1], dtype='float32'),
            # constant_15
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_14
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_245
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # constant_13
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_177
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # constant_12
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_11
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_154
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
            # constant_10
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_132
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            # constant_9
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_8
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_67
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            # constant_7
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_6
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_5
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_4
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_3
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_2
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_0
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[24, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[24, 24, 1, 1], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[72, 12, 1, 1], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[72, 12, 1, 1], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[48, 1, 7, 7], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[16, 72, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[16, 72, 1, 1], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[48, 16, 1, 1], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[48, 16, 1, 1], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[16, 48, 1, 1], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[16, 48, 1, 1], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[192, 32, 1, 1], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[48, 1, 7, 7], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[48, 1, 9, 9], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[16, 192, 1, 1], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[192, 16, 1, 1], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[40, 192, 1, 1], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[120, 20, 1, 1], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[120, 20, 1, 1], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[120, 1, 3, 3], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[120, 1, 5, 5], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[20, 240, 1, 1], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[240, 20, 1, 1], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[20, 120, 1, 1], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[20, 120, 1, 1], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[120, 20, 1, 1], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[120, 20, 1, 1], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[120, 1, 3, 3], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[120, 1, 5, 5], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[20, 240, 1, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[240, 20, 1, 1], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[20, 120, 1, 1], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[20, 120, 1, 1], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[120, 20, 1, 1], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[120, 20, 1, 1], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[120, 1, 3, 3], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[120, 1, 5, 5], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[20, 240, 1, 1], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[240, 20, 1, 1], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[20, 120, 1, 1], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[20, 120, 1, 1], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[80, 1, 5, 5], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[80, 1, 7, 7], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[10, 240, 1, 1], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[240, 10, 1, 1], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[80, 240, 1, 1], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[120, 1, 3, 3], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[120, 1, 5, 5], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[120, 1, 7, 7], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[120, 1, 9, 9], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[20, 480, 1, 1], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[480, 20, 1, 1], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[40, 240, 1, 1], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[40, 240, 1, 1], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[120, 1, 3, 3], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[120, 1, 5, 5], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[120, 1, 7, 7], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[120, 1, 9, 9], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[20, 480, 1, 1], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[480, 20, 1, 1], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[40, 240, 1, 1], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[40, 240, 1, 1], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[120, 1, 3, 3], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[120, 1, 5, 5], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[120, 1, 7, 7], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[120, 1, 9, 9], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[20, 480, 1, 1], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[480, 20, 1, 1], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[40, 240, 1, 1], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[40, 240, 1, 1], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[480, 80, 1, 1], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[480, 1, 3, 3], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[40, 480, 1, 1], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[480, 40, 1, 1], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[120, 480, 1, 1], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[180, 60, 1, 1], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[180, 60, 1, 1], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[90, 1, 3, 3], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[90, 1, 5, 5], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[90, 1, 7, 7], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[90, 1, 9, 9], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[60, 360, 1, 1], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[360, 60, 1, 1], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[60, 180, 1, 1], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[60, 180, 1, 1], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[180, 60, 1, 1], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[180, 60, 1, 1], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[90, 1, 3, 3], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[90, 1, 5, 5], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[90, 1, 7, 7], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[90, 1, 9, 9], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[60, 360, 1, 1], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[360, 60, 1, 1], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[60, 180, 1, 1], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[60, 180, 1, 1], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[180, 60, 1, 1], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[180, 60, 1, 1], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[90, 1, 3, 3], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[90, 1, 5, 5], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[90, 1, 7, 7], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[90, 1, 9, 9], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[360], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[60, 360, 1, 1], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[360, 60, 1, 1], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[60, 180, 1, 1], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[60, 180, 1, 1], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[720, 120, 1, 1], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[180, 1, 3, 3], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[180, 1, 5, 5], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[180, 1, 7, 7], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[180, 1, 9, 9], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[60, 720, 1, 1], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[720, 60, 1, 1], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[200, 720, 1, 1], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[1200, 200, 1, 1], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[300, 1, 3, 3], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[300, 1, 5, 5], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[300, 1, 7, 7], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[300, 1, 9, 9], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[100, 1200, 1, 1], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[1200, 100, 1, 1], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[100, 600, 1, 1], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[100, 600, 1, 1], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[1200, 200, 1, 1], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[300, 1, 3, 3], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[300, 1, 5, 5], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[300, 1, 7, 7], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[300, 1, 9, 9], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[100, 1200, 1, 1], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[1200, 100, 1, 1], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[100, 600, 1, 1], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[100, 600, 1, 1], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[1200, 200, 1, 1], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[300, 1, 3, 3], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[300, 1, 5, 5], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[300, 1, 7, 7], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[300, 1, 9, 9], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[1200], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[100, 1200, 1, 1], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[1200, 100, 1, 1], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[100, 600, 1, 1], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[100, 600, 1, 1], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[1536, 200, 1, 1], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
            # parameter_420
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