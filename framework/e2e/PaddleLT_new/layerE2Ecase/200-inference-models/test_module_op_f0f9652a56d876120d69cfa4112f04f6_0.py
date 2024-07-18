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
    return [379][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2602_0_0(self, parameter_366, parameter_364, parameter_362, parameter_355, parameter_353, parameter_351, parameter_349, parameter_347, parameter_340, parameter_338, parameter_336, parameter_329, parameter_327, parameter_325, parameter_323, parameter_321, parameter_311, constant_7, constant_6, parameter_306, parameter_304, parameter_302, parameter_295, parameter_293, parameter_291, parameter_289, parameter_287, parameter_280, parameter_278, parameter_276, parameter_269, parameter_267, parameter_265, parameter_263, parameter_261, parameter_254, parameter_252, parameter_250, parameter_243, parameter_241, parameter_239, parameter_237, parameter_235, parameter_228, parameter_226, parameter_224, parameter_217, parameter_215, parameter_213, parameter_211, parameter_209, parameter_202, parameter_200, parameter_198, parameter_191, parameter_189, parameter_187, parameter_185, parameter_183, parameter_173, constant_5, constant_4, parameter_168, parameter_166, parameter_164, parameter_157, parameter_155, parameter_153, parameter_151, parameter_149, parameter_142, parameter_140, parameter_138, parameter_131, parameter_129, parameter_127, parameter_125, parameter_123, parameter_116, parameter_114, parameter_112, parameter_105, parameter_103, parameter_101, parameter_99, parameter_97, parameter_87, constant_3, constant_2, parameter_82, parameter_80, parameter_78, parameter_71, parameter_69, parameter_67, parameter_65, parameter_63, parameter_56, parameter_54, parameter_52, parameter_45, parameter_43, parameter_41, parameter_39, parameter_37, parameter_30, parameter_28, parameter_26, parameter_19, parameter_17, parameter_15, parameter_13, parameter_11, parameter_1, constant_1, constant_0, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_12, parameter_14, parameter_16, parameter_18, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_27, parameter_29, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_38, parameter_40, parameter_42, parameter_44, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_53, parameter_55, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_64, parameter_66, parameter_68, parameter_70, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_79, parameter_81, parameter_83, parameter_85, parameter_84, parameter_86, parameter_91, parameter_88, parameter_90, parameter_89, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_98, parameter_100, parameter_102, parameter_104, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_113, parameter_115, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_124, parameter_126, parameter_128, parameter_130, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_139, parameter_141, parameter_143, parameter_147, parameter_144, parameter_146, parameter_145, parameter_148, parameter_150, parameter_152, parameter_154, parameter_156, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_165, parameter_167, parameter_169, parameter_171, parameter_170, parameter_172, parameter_177, parameter_174, parameter_176, parameter_175, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_184, parameter_186, parameter_188, parameter_190, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_199, parameter_201, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_210, parameter_212, parameter_214, parameter_216, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_225, parameter_227, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_236, parameter_238, parameter_240, parameter_242, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_251, parameter_253, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_262, parameter_264, parameter_266, parameter_268, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_277, parameter_279, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_288, parameter_290, parameter_292, parameter_294, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_303, parameter_305, parameter_307, parameter_309, parameter_308, parameter_310, parameter_315, parameter_312, parameter_314, parameter_313, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_322, parameter_324, parameter_326, parameter_328, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_337, parameter_339, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_348, parameter_350, parameter_352, parameter_354, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_363, parameter_365, parameter_367, parameter_369, parameter_368, parameter_370, parameter_371, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.shape: (4xi32) <- (-1x3x224x224xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(cast_0, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x3x224x224xf16, 32x3x7x7xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [4, 4], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__0 = paddle._C_ops.add(conv2d_0, parameter_1)

        # pd_op.batch_norm_: (-1x32x56x56xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.batch_norm_: (-1x32x56x56xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(batch_norm__0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x32x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(batch_norm__6, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__1 = paddle._C_ops.add(conv2d_1, parameter_11)

        # pd_op.gelu: (-1x32x56x56xf16) <- (-1x32x56x56xf16)
        gelu_0 = paddle._C_ops.gelu(add__1, False)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x1x5x5xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(gelu_0, parameter_12, [1, 1], [2, 2], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__2 = paddle._C_ops.add(depthwise_conv2d_0, parameter_13)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x1x7x7xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(add__2, parameter_14, [1, 1], [9, 9], 'EXPLICIT', 32, [3, 3], 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__3 = paddle._C_ops.add(depthwise_conv2d_1, parameter_15)

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x32x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(add__3, parameter_16, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__4 = paddle._C_ops.add(conv2d_2, parameter_17)

        # pd_op.multiply_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        multiply__0 = paddle._C_ops.multiply(gelu_0, add__4)

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x32x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(multiply__0, parameter_18, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__5 = paddle._C_ops.add(conv2d_3, parameter_19)

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        add__6 = paddle._C_ops.add(add__5, batch_norm__6)

        # pd_op.multiply: (-1x32x56x56xf16) <- (32x1x1xf16, -1x32x56x56xf16)
        multiply_0 = parameter_20 * add__6

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        add__7 = paddle._C_ops.add(batch_norm__0, multiply_0)

        # pd_op.batch_norm_: (-1x32x56x56xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__7, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x32x56x56xf16, 256x32x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(batch_norm__12, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__8 = paddle._C_ops.add(conv2d_4, parameter_26)

        # pd_op.depthwise_conv2d: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 256x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(add__8, parameter_27, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__9 = paddle._C_ops.add(depthwise_conv2d_2, parameter_28)

        # pd_op.gelu: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        gelu_1 = paddle._C_ops.gelu(add__9, False)

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x256x56x56xf16, 32x256x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(gelu_1, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__10 = paddle._C_ops.add(conv2d_5, parameter_30)

        # pd_op.multiply: (-1x32x56x56xf16) <- (32x1x1xf16, -1x32x56x56xf16)
        multiply_1 = parameter_31 * add__10

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        add__11 = paddle._C_ops.add(add__7, multiply_1)

        # pd_op.batch_norm_: (-1x32x56x56xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__11, parameter_32, parameter_33, parameter_34, parameter_35, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x32x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(batch_norm__18, parameter_36, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__12 = paddle._C_ops.add(conv2d_6, parameter_37)

        # pd_op.gelu: (-1x32x56x56xf16) <- (-1x32x56x56xf16)
        gelu_2 = paddle._C_ops.gelu(add__12, False)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x1x5x5xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(gelu_2, parameter_38, [1, 1], [2, 2], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__13 = paddle._C_ops.add(depthwise_conv2d_3, parameter_39)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x1x7x7xf16)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(add__13, parameter_40, [1, 1], [9, 9], 'EXPLICIT', 32, [3, 3], 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__14 = paddle._C_ops.add(depthwise_conv2d_4, parameter_41)

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x32x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(add__14, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__15 = paddle._C_ops.add(conv2d_7, parameter_43)

        # pd_op.multiply_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        multiply__1 = paddle._C_ops.multiply(gelu_2, add__15)

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x32x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(multiply__1, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__16 = paddle._C_ops.add(conv2d_8, parameter_45)

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        add__17 = paddle._C_ops.add(add__16, batch_norm__18)

        # pd_op.multiply: (-1x32x56x56xf16) <- (32x1x1xf16, -1x32x56x56xf16)
        multiply_2 = parameter_46 * add__17

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        add__18 = paddle._C_ops.add(add__11, multiply_2)

        # pd_op.batch_norm_: (-1x32x56x56xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__18, parameter_47, parameter_48, parameter_49, parameter_50, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x32x56x56xf16, 256x32x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(batch_norm__24, parameter_51, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__19 = paddle._C_ops.add(conv2d_9, parameter_52)

        # pd_op.depthwise_conv2d: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 256x1x3x3xf16)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(add__19, parameter_53, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__20 = paddle._C_ops.add(depthwise_conv2d_5, parameter_54)

        # pd_op.gelu: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        gelu_3 = paddle._C_ops.gelu(add__20, False)

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x256x56x56xf16, 32x256x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(gelu_3, parameter_55, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__21 = paddle._C_ops.add(conv2d_10, parameter_56)

        # pd_op.multiply: (-1x32x56x56xf16) <- (32x1x1xf16, -1x32x56x56xf16)
        multiply_3 = parameter_57 * add__21

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        add__22 = paddle._C_ops.add(add__18, multiply_3)

        # pd_op.batch_norm_: (-1x32x56x56xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__22, parameter_58, parameter_59, parameter_60, parameter_61, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x32x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(batch_norm__30, parameter_62, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__23 = paddle._C_ops.add(conv2d_11, parameter_63)

        # pd_op.gelu: (-1x32x56x56xf16) <- (-1x32x56x56xf16)
        gelu_4 = paddle._C_ops.gelu(add__23, False)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x1x5x5xf16)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(gelu_4, parameter_64, [1, 1], [2, 2], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__24 = paddle._C_ops.add(depthwise_conv2d_6, parameter_65)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x1x7x7xf16)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(add__24, parameter_66, [1, 1], [9, 9], 'EXPLICIT', 32, [3, 3], 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__25 = paddle._C_ops.add(depthwise_conv2d_7, parameter_67)

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x32x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(add__25, parameter_68, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__26 = paddle._C_ops.add(conv2d_12, parameter_69)

        # pd_op.multiply_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        multiply__2 = paddle._C_ops.multiply(gelu_4, add__26)

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 32x32x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(multiply__2, parameter_70, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__27 = paddle._C_ops.add(conv2d_13, parameter_71)

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        add__28 = paddle._C_ops.add(add__27, batch_norm__30)

        # pd_op.multiply: (-1x32x56x56xf16) <- (32x1x1xf16, -1x32x56x56xf16)
        multiply_4 = parameter_72 * add__28

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        add__29 = paddle._C_ops.add(add__22, multiply_4)

        # pd_op.batch_norm_: (-1x32x56x56xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__29, parameter_73, parameter_74, parameter_75, parameter_76, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x32x56x56xf16, 256x32x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(batch_norm__36, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__30 = paddle._C_ops.add(conv2d_14, parameter_78)

        # pd_op.depthwise_conv2d: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 256x1x3x3xf16)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(add__30, parameter_79, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__31 = paddle._C_ops.add(depthwise_conv2d_8, parameter_80)

        # pd_op.gelu: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        gelu_5 = paddle._C_ops.gelu(add__31, False)

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x256x56x56xf16, 32x256x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(gelu_5, parameter_81, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__32 = paddle._C_ops.add(conv2d_15, parameter_82)

        # pd_op.multiply: (-1x32x56x56xf16) <- (32x1x1xf16, -1x32x56x56xf16)
        multiply_5 = parameter_83 * add__32

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, -1x32x56x56xf16)
        add__33 = paddle._C_ops.add(add__29, multiply_5)

        # pd_op.flatten_: (-1x32x3136xf16, None) <- (-1x32x56x56xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__33, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x32xf16) <- (-1x32x3136xf16)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.layer_norm: (-1x3136x32xf16, -3136xf32, -3136xf32) <- (-1x3136x32xf16, 32xf32, 32xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_0, parameter_84, parameter_85, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, constant_2, constant_2, constant_3]

        # pd_op.reshape_: (-1x56x56x32xf16, 0x-1x3136x32xf16) <- (-1x3136x32xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(layer_norm_0, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x32x56x56xf16) <- (-1x56x56x32xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x32x56x56xf16, 64x32x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(transpose_1, parameter_86, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__34 = paddle._C_ops.add(conv2d_16, parameter_87)

        # pd_op.batch_norm_: (-1x64x28x28xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__34, parameter_88, parameter_89, parameter_90, parameter_91, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.batch_norm_: (-1x64x28x28xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(batch_norm__42, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(batch_norm__48, parameter_96, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__35 = paddle._C_ops.add(conv2d_17, parameter_97)

        # pd_op.gelu: (-1x64x28x28xf16) <- (-1x64x28x28xf16)
        gelu_6 = paddle._C_ops.gelu(add__35, False)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x1x5x5xf16)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(gelu_6, parameter_98, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__36 = paddle._C_ops.add(depthwise_conv2d_9, parameter_99)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x1x7x7xf16)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(add__36, parameter_100, [1, 1], [9, 9], 'EXPLICIT', 64, [3, 3], 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__37 = paddle._C_ops.add(depthwise_conv2d_10, parameter_101)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(add__37, parameter_102, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__38 = paddle._C_ops.add(conv2d_18, parameter_103)

        # pd_op.multiply_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        multiply__3 = paddle._C_ops.multiply(gelu_6, add__38)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(multiply__3, parameter_104, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__39 = paddle._C_ops.add(conv2d_19, parameter_105)

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__40 = paddle._C_ops.add(add__39, batch_norm__48)

        # pd_op.multiply: (-1x64x28x28xf16) <- (64x1x1xf16, -1x64x28x28xf16)
        multiply_6 = parameter_106 * add__40

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__41 = paddle._C_ops.add(batch_norm__42, multiply_6)

        # pd_op.batch_norm_: (-1x64x28x28xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__41, parameter_107, parameter_108, parameter_109, parameter_110, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x64x28x28xf16, 512x64x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(batch_norm__54, parameter_111, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__42 = paddle._C_ops.add(conv2d_20, parameter_112)

        # pd_op.depthwise_conv2d: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 512x1x3x3xf16)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(add__42, parameter_113, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__43 = paddle._C_ops.add(depthwise_conv2d_11, parameter_114)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_7 = paddle._C_ops.gelu(add__43, False)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x512x28x28xf16, 64x512x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(gelu_7, parameter_115, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__44 = paddle._C_ops.add(conv2d_21, parameter_116)

        # pd_op.multiply: (-1x64x28x28xf16) <- (64x1x1xf16, -1x64x28x28xf16)
        multiply_7 = parameter_117 * add__44

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__45 = paddle._C_ops.add(add__41, multiply_7)

        # pd_op.batch_norm_: (-1x64x28x28xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__45, parameter_118, parameter_119, parameter_120, parameter_121, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(batch_norm__60, parameter_122, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__46 = paddle._C_ops.add(conv2d_22, parameter_123)

        # pd_op.gelu: (-1x64x28x28xf16) <- (-1x64x28x28xf16)
        gelu_8 = paddle._C_ops.gelu(add__46, False)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x1x5x5xf16)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(gelu_8, parameter_124, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__47 = paddle._C_ops.add(depthwise_conv2d_12, parameter_125)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x1x7x7xf16)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(add__47, parameter_126, [1, 1], [9, 9], 'EXPLICIT', 64, [3, 3], 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__48 = paddle._C_ops.add(depthwise_conv2d_13, parameter_127)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(add__48, parameter_128, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__49 = paddle._C_ops.add(conv2d_23, parameter_129)

        # pd_op.multiply_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        multiply__4 = paddle._C_ops.multiply(gelu_8, add__49)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(multiply__4, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__50 = paddle._C_ops.add(conv2d_24, parameter_131)

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__51 = paddle._C_ops.add(add__50, batch_norm__60)

        # pd_op.multiply: (-1x64x28x28xf16) <- (64x1x1xf16, -1x64x28x28xf16)
        multiply_8 = parameter_132 * add__51

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__52 = paddle._C_ops.add(add__45, multiply_8)

        # pd_op.batch_norm_: (-1x64x28x28xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__52, parameter_133, parameter_134, parameter_135, parameter_136, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x64x28x28xf16, 512x64x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(batch_norm__66, parameter_137, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__53 = paddle._C_ops.add(conv2d_25, parameter_138)

        # pd_op.depthwise_conv2d: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 512x1x3x3xf16)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(add__53, parameter_139, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__54 = paddle._C_ops.add(depthwise_conv2d_14, parameter_140)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_9 = paddle._C_ops.gelu(add__54, False)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x512x28x28xf16, 64x512x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(gelu_9, parameter_141, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__55 = paddle._C_ops.add(conv2d_26, parameter_142)

        # pd_op.multiply: (-1x64x28x28xf16) <- (64x1x1xf16, -1x64x28x28xf16)
        multiply_9 = parameter_143 * add__55

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__56 = paddle._C_ops.add(add__52, multiply_9)

        # pd_op.batch_norm_: (-1x64x28x28xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__56, parameter_144, parameter_145, parameter_146, parameter_147, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(batch_norm__72, parameter_148, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__57 = paddle._C_ops.add(conv2d_27, parameter_149)

        # pd_op.gelu: (-1x64x28x28xf16) <- (-1x64x28x28xf16)
        gelu_10 = paddle._C_ops.gelu(add__57, False)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x1x5x5xf16)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(gelu_10, parameter_150, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__58 = paddle._C_ops.add(depthwise_conv2d_15, parameter_151)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x1x7x7xf16)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(add__58, parameter_152, [1, 1], [9, 9], 'EXPLICIT', 64, [3, 3], 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__59 = paddle._C_ops.add(depthwise_conv2d_16, parameter_153)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(add__59, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__60 = paddle._C_ops.add(conv2d_28, parameter_155)

        # pd_op.multiply_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        multiply__5 = paddle._C_ops.multiply(gelu_10, add__60)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(multiply__5, parameter_156, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__61 = paddle._C_ops.add(conv2d_29, parameter_157)

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__62 = paddle._C_ops.add(add__61, batch_norm__72)

        # pd_op.multiply: (-1x64x28x28xf16) <- (64x1x1xf16, -1x64x28x28xf16)
        multiply_10 = parameter_158 * add__62

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__63 = paddle._C_ops.add(add__56, multiply_10)

        # pd_op.batch_norm_: (-1x64x28x28xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__63, parameter_159, parameter_160, parameter_161, parameter_162, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x64x28x28xf16, 512x64x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(batch_norm__78, parameter_163, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__64 = paddle._C_ops.add(conv2d_30, parameter_164)

        # pd_op.depthwise_conv2d: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 512x1x3x3xf16)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(add__64, parameter_165, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__65 = paddle._C_ops.add(depthwise_conv2d_17, parameter_166)

        # pd_op.gelu: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        gelu_11 = paddle._C_ops.gelu(add__65, False)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x512x28x28xf16, 64x512x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(gelu_11, parameter_167, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__66 = paddle._C_ops.add(conv2d_31, parameter_168)

        # pd_op.multiply: (-1x64x28x28xf16) <- (64x1x1xf16, -1x64x28x28xf16)
        multiply_11 = parameter_169 * add__66

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__67 = paddle._C_ops.add(add__63, multiply_11)

        # pd_op.flatten_: (-1x64x784xf16, None) <- (-1x64x28x28xf16)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__67, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x64xf16) <- (-1x64x784xf16)
        transpose_2 = paddle._C_ops.transpose(flatten__2, [0, 2, 1])

        # pd_op.layer_norm: (-1x784x64xf16, -784xf32, -784xf32) <- (-1x784x64xf16, 64xf32, 64xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_2, parameter_170, parameter_171, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_0, constant_4, constant_4, constant_5]

        # pd_op.reshape_: (-1x28x28x64xf16, 0x-1x784x64xf16) <- (-1x784x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(layer_norm_3, combine_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x28x28xf16) <- (-1x28x28x64xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__2, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x64x28x28xf16, 160x64x3x3xf16)
        conv2d_32 = paddle._C_ops.conv2d(transpose_3, parameter_172, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__68 = paddle._C_ops.add(conv2d_32, parameter_173)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__68, parameter_174, parameter_175, parameter_176, parameter_177, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(batch_norm__84, parameter_178, parameter_179, parameter_180, parameter_181, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(batch_norm__90, parameter_182, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__69 = paddle._C_ops.add(conv2d_33, parameter_183)

        # pd_op.gelu: (-1x160x14x14xf16) <- (-1x160x14x14xf16)
        gelu_12 = paddle._C_ops.gelu(add__69, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x5x5xf16)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(gelu_12, parameter_184, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__70 = paddle._C_ops.add(depthwise_conv2d_18, parameter_185)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x7x7xf16)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(add__70, parameter_186, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__71 = paddle._C_ops.add(depthwise_conv2d_19, parameter_187)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(add__71, parameter_188, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__72 = paddle._C_ops.add(conv2d_34, parameter_189)

        # pd_op.multiply_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        multiply__6 = paddle._C_ops.multiply(gelu_12, add__72)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(multiply__6, parameter_190, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__73 = paddle._C_ops.add(conv2d_35, parameter_191)

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__74 = paddle._C_ops.add(add__73, batch_norm__90)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_12 = parameter_192 * add__74

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__75 = paddle._C_ops.add(batch_norm__84, multiply_12)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__75, parameter_193, parameter_194, parameter_195, parameter_196, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x160x14x14xf16, 640x160x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(batch_norm__96, parameter_197, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__76 = paddle._C_ops.add(conv2d_36, parameter_198)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x1x3x3xf16)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(add__76, parameter_199, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__77 = paddle._C_ops.add(depthwise_conv2d_20, parameter_200)

        # pd_op.gelu: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        gelu_13 = paddle._C_ops.gelu(add__77, False)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x640x14x14xf16, 160x640x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(gelu_13, parameter_201, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__78 = paddle._C_ops.add(conv2d_37, parameter_202)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_13 = parameter_203 * add__78

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__79 = paddle._C_ops.add(add__75, multiply_13)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__79, parameter_204, parameter_205, parameter_206, parameter_207, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(batch_norm__102, parameter_208, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__80 = paddle._C_ops.add(conv2d_38, parameter_209)

        # pd_op.gelu: (-1x160x14x14xf16) <- (-1x160x14x14xf16)
        gelu_14 = paddle._C_ops.gelu(add__80, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x5x5xf16)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(gelu_14, parameter_210, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__81 = paddle._C_ops.add(depthwise_conv2d_21, parameter_211)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x7x7xf16)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(add__81, parameter_212, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__82 = paddle._C_ops.add(depthwise_conv2d_22, parameter_213)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(add__82, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__83 = paddle._C_ops.add(conv2d_39, parameter_215)

        # pd_op.multiply_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        multiply__7 = paddle._C_ops.multiply(gelu_14, add__83)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(multiply__7, parameter_216, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__84 = paddle._C_ops.add(conv2d_40, parameter_217)

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__85 = paddle._C_ops.add(add__84, batch_norm__102)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_14 = parameter_218 * add__85

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__86 = paddle._C_ops.add(add__79, multiply_14)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__86, parameter_219, parameter_220, parameter_221, parameter_222, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x160x14x14xf16, 640x160x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(batch_norm__108, parameter_223, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__87 = paddle._C_ops.add(conv2d_41, parameter_224)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x1x3x3xf16)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(add__87, parameter_225, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__88 = paddle._C_ops.add(depthwise_conv2d_23, parameter_226)

        # pd_op.gelu: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        gelu_15 = paddle._C_ops.gelu(add__88, False)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x640x14x14xf16, 160x640x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(gelu_15, parameter_227, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__89 = paddle._C_ops.add(conv2d_42, parameter_228)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_15 = parameter_229 * add__89

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__90 = paddle._C_ops.add(add__86, multiply_15)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__90, parameter_230, parameter_231, parameter_232, parameter_233, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(batch_norm__114, parameter_234, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__91 = paddle._C_ops.add(conv2d_43, parameter_235)

        # pd_op.gelu: (-1x160x14x14xf16) <- (-1x160x14x14xf16)
        gelu_16 = paddle._C_ops.gelu(add__91, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x5x5xf16)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(gelu_16, parameter_236, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__92 = paddle._C_ops.add(depthwise_conv2d_24, parameter_237)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x7x7xf16)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(add__92, parameter_238, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__93 = paddle._C_ops.add(depthwise_conv2d_25, parameter_239)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(add__93, parameter_240, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__94 = paddle._C_ops.add(conv2d_44, parameter_241)

        # pd_op.multiply_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        multiply__8 = paddle._C_ops.multiply(gelu_16, add__94)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(multiply__8, parameter_242, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__95 = paddle._C_ops.add(conv2d_45, parameter_243)

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__96 = paddle._C_ops.add(add__95, batch_norm__114)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_16 = parameter_244 * add__96

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__97 = paddle._C_ops.add(add__90, multiply_16)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__97, parameter_245, parameter_246, parameter_247, parameter_248, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x160x14x14xf16, 640x160x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(batch_norm__120, parameter_249, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__98 = paddle._C_ops.add(conv2d_46, parameter_250)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x1x3x3xf16)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(add__98, parameter_251, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__99 = paddle._C_ops.add(depthwise_conv2d_26, parameter_252)

        # pd_op.gelu: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        gelu_17 = paddle._C_ops.gelu(add__99, False)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x640x14x14xf16, 160x640x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(gelu_17, parameter_253, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__100 = paddle._C_ops.add(conv2d_47, parameter_254)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_17 = parameter_255 * add__100

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__101 = paddle._C_ops.add(add__97, multiply_17)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__101, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(batch_norm__126, parameter_260, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__102 = paddle._C_ops.add(conv2d_48, parameter_261)

        # pd_op.gelu: (-1x160x14x14xf16) <- (-1x160x14x14xf16)
        gelu_18 = paddle._C_ops.gelu(add__102, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x5x5xf16)
        depthwise_conv2d_27 = paddle._C_ops.depthwise_conv2d(gelu_18, parameter_262, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__103 = paddle._C_ops.add(depthwise_conv2d_27, parameter_263)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x7x7xf16)
        depthwise_conv2d_28 = paddle._C_ops.depthwise_conv2d(add__103, parameter_264, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__104 = paddle._C_ops.add(depthwise_conv2d_28, parameter_265)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(add__104, parameter_266, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__105 = paddle._C_ops.add(conv2d_49, parameter_267)

        # pd_op.multiply_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        multiply__9 = paddle._C_ops.multiply(gelu_18, add__105)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(multiply__9, parameter_268, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__106 = paddle._C_ops.add(conv2d_50, parameter_269)

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__107 = paddle._C_ops.add(add__106, batch_norm__126)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_18 = parameter_270 * add__107

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__108 = paddle._C_ops.add(add__101, multiply_18)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__108, parameter_271, parameter_272, parameter_273, parameter_274, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x160x14x14xf16, 640x160x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(batch_norm__132, parameter_275, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__109 = paddle._C_ops.add(conv2d_51, parameter_276)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x1x3x3xf16)
        depthwise_conv2d_29 = paddle._C_ops.depthwise_conv2d(add__109, parameter_277, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__110 = paddle._C_ops.add(depthwise_conv2d_29, parameter_278)

        # pd_op.gelu: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        gelu_19 = paddle._C_ops.gelu(add__110, False)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x640x14x14xf16, 160x640x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(gelu_19, parameter_279, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__111 = paddle._C_ops.add(conv2d_52, parameter_280)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_19 = parameter_281 * add__111

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__112 = paddle._C_ops.add(add__108, multiply_19)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__112, parameter_282, parameter_283, parameter_284, parameter_285, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_53 = paddle._C_ops.conv2d(batch_norm__138, parameter_286, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__113 = paddle._C_ops.add(conv2d_53, parameter_287)

        # pd_op.gelu: (-1x160x14x14xf16) <- (-1x160x14x14xf16)
        gelu_20 = paddle._C_ops.gelu(add__113, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x5x5xf16)
        depthwise_conv2d_30 = paddle._C_ops.depthwise_conv2d(gelu_20, parameter_288, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__114 = paddle._C_ops.add(depthwise_conv2d_30, parameter_289)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x1x7x7xf16)
        depthwise_conv2d_31 = paddle._C_ops.depthwise_conv2d(add__114, parameter_290, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__115 = paddle._C_ops.add(depthwise_conv2d_31, parameter_291)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(add__115, parameter_292, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__116 = paddle._C_ops.add(conv2d_54, parameter_293)

        # pd_op.multiply_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        multiply__10 = paddle._C_ops.multiply(gelu_20, add__116)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 160x160x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(multiply__10, parameter_294, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__117 = paddle._C_ops.add(conv2d_55, parameter_295)

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__118 = paddle._C_ops.add(add__117, batch_norm__138)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_20 = parameter_296 * add__118

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__119 = paddle._C_ops.add(add__112, multiply_20)

        # pd_op.batch_norm_: (-1x160x14x14xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__119, parameter_297, parameter_298, parameter_299, parameter_300, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x160x14x14xf16, 640x160x1x1xf16)
        conv2d_56 = paddle._C_ops.conv2d(batch_norm__144, parameter_301, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__120 = paddle._C_ops.add(conv2d_56, parameter_302)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x1x3x3xf16)
        depthwise_conv2d_32 = paddle._C_ops.depthwise_conv2d(add__120, parameter_303, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__121 = paddle._C_ops.add(depthwise_conv2d_32, parameter_304)

        # pd_op.gelu: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        gelu_21 = paddle._C_ops.gelu(add__121, False)

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x640x14x14xf16, 160x640x1x1xf16)
        conv2d_57 = paddle._C_ops.conv2d(gelu_21, parameter_305, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__122 = paddle._C_ops.add(conv2d_57, parameter_306)

        # pd_op.multiply: (-1x160x14x14xf16) <- (160x1x1xf16, -1x160x14x14xf16)
        multiply_21 = parameter_307 * add__122

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, -1x160x14x14xf16)
        add__123 = paddle._C_ops.add(add__119, multiply_21)

        # pd_op.flatten_: (-1x160x196xf16, None) <- (-1x160x14x14xf16)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__123, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x160xf16) <- (-1x160x196xf16)
        transpose_4 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x160xf16, -196xf32, -196xf32) <- (-1x196x160xf16, 160xf32, 160xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_4, parameter_308, parameter_309, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_0, constant_6, constant_6, constant_7]

        # pd_op.reshape_: (-1x14x14x160xf16, 0x-1x196x160xf16) <- (-1x196x160xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(layer_norm_6, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x160x14x14xf16) <- (-1x14x14x160xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__4, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x160x14x14xf16, 256x160x3x3xf16)
        conv2d_58 = paddle._C_ops.conv2d(transpose_5, parameter_310, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__124 = paddle._C_ops.add(conv2d_58, parameter_311)

        # pd_op.batch_norm_: (-1x256x7x7xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__124, parameter_312, parameter_313, parameter_314, parameter_315, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.batch_norm_: (-1x256x7x7xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(batch_norm__150, parameter_316, parameter_317, parameter_318, parameter_319, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x256x1x1xf16)
        conv2d_59 = paddle._C_ops.conv2d(batch_norm__156, parameter_320, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__125 = paddle._C_ops.add(conv2d_59, parameter_321)

        # pd_op.gelu: (-1x256x7x7xf16) <- (-1x256x7x7xf16)
        gelu_22 = paddle._C_ops.gelu(add__125, False)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x1x5x5xf16)
        depthwise_conv2d_33 = paddle._C_ops.depthwise_conv2d(gelu_22, parameter_322, [1, 1], [2, 2], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__126 = paddle._C_ops.add(depthwise_conv2d_33, parameter_323)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x1x7x7xf16)
        depthwise_conv2d_34 = paddle._C_ops.depthwise_conv2d(add__126, parameter_324, [1, 1], [9, 9], 'EXPLICIT', 256, [3, 3], 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__127 = paddle._C_ops.add(depthwise_conv2d_34, parameter_325)

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x256x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(add__127, parameter_326, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__128 = paddle._C_ops.add(conv2d_60, parameter_327)

        # pd_op.multiply_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, -1x256x7x7xf16)
        multiply__11 = paddle._C_ops.multiply(gelu_22, add__128)

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x256x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(multiply__11, parameter_328, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__129 = paddle._C_ops.add(conv2d_61, parameter_329)

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, -1x256x7x7xf16)
        add__130 = paddle._C_ops.add(add__129, batch_norm__156)

        # pd_op.multiply: (-1x256x7x7xf16) <- (256x1x1xf16, -1x256x7x7xf16)
        multiply_22 = parameter_330 * add__130

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, -1x256x7x7xf16)
        add__131 = paddle._C_ops.add(batch_norm__150, multiply_22)

        # pd_op.batch_norm_: (-1x256x7x7xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__131, parameter_331, parameter_332, parameter_333, parameter_334, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x256x7x7xf16, 1024x256x1x1xf16)
        conv2d_62 = paddle._C_ops.conv2d(batch_norm__162, parameter_335, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1x1024x1x1xf16)
        add__132 = paddle._C_ops.add(conv2d_62, parameter_336)

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1024x1x3x3xf16)
        depthwise_conv2d_35 = paddle._C_ops.depthwise_conv2d(add__132, parameter_337, [1, 1], [1, 1], 'EXPLICIT', 1024, [1, 1], 'NCHW')

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1x1024x1x1xf16)
        add__133 = paddle._C_ops.add(depthwise_conv2d_35, parameter_338)

        # pd_op.gelu: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        gelu_23 = paddle._C_ops.gelu(add__133, False)

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x1024x7x7xf16, 256x1024x1x1xf16)
        conv2d_63 = paddle._C_ops.conv2d(gelu_23, parameter_339, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__134 = paddle._C_ops.add(conv2d_63, parameter_340)

        # pd_op.multiply: (-1x256x7x7xf16) <- (256x1x1xf16, -1x256x7x7xf16)
        multiply_23 = parameter_341 * add__134

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, -1x256x7x7xf16)
        add__135 = paddle._C_ops.add(add__131, multiply_23)

        # pd_op.batch_norm_: (-1x256x7x7xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__135, parameter_342, parameter_343, parameter_344, parameter_345, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x256x1x1xf16)
        conv2d_64 = paddle._C_ops.conv2d(batch_norm__168, parameter_346, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__136 = paddle._C_ops.add(conv2d_64, parameter_347)

        # pd_op.gelu: (-1x256x7x7xf16) <- (-1x256x7x7xf16)
        gelu_24 = paddle._C_ops.gelu(add__136, False)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x1x5x5xf16)
        depthwise_conv2d_36 = paddle._C_ops.depthwise_conv2d(gelu_24, parameter_348, [1, 1], [2, 2], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__137 = paddle._C_ops.add(depthwise_conv2d_36, parameter_349)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x1x7x7xf16)
        depthwise_conv2d_37 = paddle._C_ops.depthwise_conv2d(add__137, parameter_350, [1, 1], [9, 9], 'EXPLICIT', 256, [3, 3], 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__138 = paddle._C_ops.add(depthwise_conv2d_37, parameter_351)

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x256x1x1xf16)
        conv2d_65 = paddle._C_ops.conv2d(add__138, parameter_352, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__139 = paddle._C_ops.add(conv2d_65, parameter_353)

        # pd_op.multiply_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, -1x256x7x7xf16)
        multiply__12 = paddle._C_ops.multiply(gelu_24, add__139)

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x256x1x1xf16)
        conv2d_66 = paddle._C_ops.conv2d(multiply__12, parameter_354, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__140 = paddle._C_ops.add(conv2d_66, parameter_355)

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, -1x256x7x7xf16)
        add__141 = paddle._C_ops.add(add__140, batch_norm__168)

        # pd_op.multiply: (-1x256x7x7xf16) <- (256x1x1xf16, -1x256x7x7xf16)
        multiply_24 = parameter_356 * add__141

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, -1x256x7x7xf16)
        add__142 = paddle._C_ops.add(add__135, multiply_24)

        # pd_op.batch_norm_: (-1x256x7x7xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__142, parameter_357, parameter_358, parameter_359, parameter_360, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x256x7x7xf16, 1024x256x1x1xf16)
        conv2d_67 = paddle._C_ops.conv2d(batch_norm__174, parameter_361, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1x1024x1x1xf16)
        add__143 = paddle._C_ops.add(conv2d_67, parameter_362)

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1024x1x3x3xf16)
        depthwise_conv2d_38 = paddle._C_ops.depthwise_conv2d(add__143, parameter_363, [1, 1], [1, 1], 'EXPLICIT', 1024, [1, 1], 'NCHW')

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1x1024x1x1xf16)
        add__144 = paddle._C_ops.add(depthwise_conv2d_38, parameter_364)

        # pd_op.gelu: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        gelu_25 = paddle._C_ops.gelu(add__144, False)

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x1024x7x7xf16, 256x1024x1x1xf16)
        conv2d_68 = paddle._C_ops.conv2d(gelu_25, parameter_365, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__145 = paddle._C_ops.add(conv2d_68, parameter_366)

        # pd_op.multiply: (-1x256x7x7xf16) <- (256x1x1xf16, -1x256x7x7xf16)
        multiply_25 = parameter_367 * add__145

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, -1x256x7x7xf16)
        add__146 = paddle._C_ops.add(add__142, multiply_25)

        # pd_op.flatten_: (-1x256x49xf16, None) <- (-1x256x7x7xf16)
        flatten__6, flatten__7 = (lambda x, f: f(x))(paddle._C_ops.flatten(add__146, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x256xf16) <- (-1x256x49xf16)
        transpose_6 = paddle._C_ops.transpose(flatten__6, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x256xf16, -49xf32, -49xf32) <- (-1x49x256xf16, 256xf32, 256xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_6, parameter_368, parameter_369, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.mean: (-1x256xf16) <- (-1x49x256xf16)
        mean_0 = paddle._C_ops.mean(layer_norm_9, [1], False)

        # pd_op.matmul: (-1x1000xf16) <- (-1x256xf16, 256x1000xf16)
        matmul_0 = paddle.matmul(mean_0, parameter_370, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__147 = paddle._C_ops.add(matmul_0, parameter_371)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__0 = paddle._C_ops.softmax(add__147, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__0, paddle.float32)
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

    def forward(self, parameter_366, parameter_364, parameter_362, parameter_355, parameter_353, parameter_351, parameter_349, parameter_347, parameter_340, parameter_338, parameter_336, parameter_329, parameter_327, parameter_325, parameter_323, parameter_321, parameter_311, constant_7, constant_6, parameter_306, parameter_304, parameter_302, parameter_295, parameter_293, parameter_291, parameter_289, parameter_287, parameter_280, parameter_278, parameter_276, parameter_269, parameter_267, parameter_265, parameter_263, parameter_261, parameter_254, parameter_252, parameter_250, parameter_243, parameter_241, parameter_239, parameter_237, parameter_235, parameter_228, parameter_226, parameter_224, parameter_217, parameter_215, parameter_213, parameter_211, parameter_209, parameter_202, parameter_200, parameter_198, parameter_191, parameter_189, parameter_187, parameter_185, parameter_183, parameter_173, constant_5, constant_4, parameter_168, parameter_166, parameter_164, parameter_157, parameter_155, parameter_153, parameter_151, parameter_149, parameter_142, parameter_140, parameter_138, parameter_131, parameter_129, parameter_127, parameter_125, parameter_123, parameter_116, parameter_114, parameter_112, parameter_105, parameter_103, parameter_101, parameter_99, parameter_97, parameter_87, constant_3, constant_2, parameter_82, parameter_80, parameter_78, parameter_71, parameter_69, parameter_67, parameter_65, parameter_63, parameter_56, parameter_54, parameter_52, parameter_45, parameter_43, parameter_41, parameter_39, parameter_37, parameter_30, parameter_28, parameter_26, parameter_19, parameter_17, parameter_15, parameter_13, parameter_11, parameter_1, constant_1, constant_0, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_12, parameter_14, parameter_16, parameter_18, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_27, parameter_29, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_38, parameter_40, parameter_42, parameter_44, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_53, parameter_55, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_64, parameter_66, parameter_68, parameter_70, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_79, parameter_81, parameter_83, parameter_85, parameter_84, parameter_86, parameter_91, parameter_88, parameter_90, parameter_89, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_98, parameter_100, parameter_102, parameter_104, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_113, parameter_115, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_124, parameter_126, parameter_128, parameter_130, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_139, parameter_141, parameter_143, parameter_147, parameter_144, parameter_146, parameter_145, parameter_148, parameter_150, parameter_152, parameter_154, parameter_156, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_165, parameter_167, parameter_169, parameter_171, parameter_170, parameter_172, parameter_177, parameter_174, parameter_176, parameter_175, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_184, parameter_186, parameter_188, parameter_190, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_199, parameter_201, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_210, parameter_212, parameter_214, parameter_216, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_225, parameter_227, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_236, parameter_238, parameter_240, parameter_242, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_251, parameter_253, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_262, parameter_264, parameter_266, parameter_268, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_277, parameter_279, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_288, parameter_290, parameter_292, parameter_294, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_303, parameter_305, parameter_307, parameter_309, parameter_308, parameter_310, parameter_315, parameter_312, parameter_314, parameter_313, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_322, parameter_324, parameter_326, parameter_328, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_337, parameter_339, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_348, parameter_350, parameter_352, parameter_354, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_363, parameter_365, parameter_367, parameter_369, parameter_368, parameter_370, parameter_371, feed_0):
        return self.builtin_module_2602_0_0(parameter_366, parameter_364, parameter_362, parameter_355, parameter_353, parameter_351, parameter_349, parameter_347, parameter_340, parameter_338, parameter_336, parameter_329, parameter_327, parameter_325, parameter_323, parameter_321, parameter_311, constant_7, constant_6, parameter_306, parameter_304, parameter_302, parameter_295, parameter_293, parameter_291, parameter_289, parameter_287, parameter_280, parameter_278, parameter_276, parameter_269, parameter_267, parameter_265, parameter_263, parameter_261, parameter_254, parameter_252, parameter_250, parameter_243, parameter_241, parameter_239, parameter_237, parameter_235, parameter_228, parameter_226, parameter_224, parameter_217, parameter_215, parameter_213, parameter_211, parameter_209, parameter_202, parameter_200, parameter_198, parameter_191, parameter_189, parameter_187, parameter_185, parameter_183, parameter_173, constant_5, constant_4, parameter_168, parameter_166, parameter_164, parameter_157, parameter_155, parameter_153, parameter_151, parameter_149, parameter_142, parameter_140, parameter_138, parameter_131, parameter_129, parameter_127, parameter_125, parameter_123, parameter_116, parameter_114, parameter_112, parameter_105, parameter_103, parameter_101, parameter_99, parameter_97, parameter_87, constant_3, constant_2, parameter_82, parameter_80, parameter_78, parameter_71, parameter_69, parameter_67, parameter_65, parameter_63, parameter_56, parameter_54, parameter_52, parameter_45, parameter_43, parameter_41, parameter_39, parameter_37, parameter_30, parameter_28, parameter_26, parameter_19, parameter_17, parameter_15, parameter_13, parameter_11, parameter_1, constant_1, constant_0, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_12, parameter_14, parameter_16, parameter_18, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_27, parameter_29, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_38, parameter_40, parameter_42, parameter_44, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_53, parameter_55, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_64, parameter_66, parameter_68, parameter_70, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_79, parameter_81, parameter_83, parameter_85, parameter_84, parameter_86, parameter_91, parameter_88, parameter_90, parameter_89, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_98, parameter_100, parameter_102, parameter_104, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_113, parameter_115, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_124, parameter_126, parameter_128, parameter_130, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_139, parameter_141, parameter_143, parameter_147, parameter_144, parameter_146, parameter_145, parameter_148, parameter_150, parameter_152, parameter_154, parameter_156, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_165, parameter_167, parameter_169, parameter_171, parameter_170, parameter_172, parameter_177, parameter_174, parameter_176, parameter_175, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_184, parameter_186, parameter_188, parameter_190, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_199, parameter_201, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_210, parameter_212, parameter_214, parameter_216, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_225, parameter_227, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_236, parameter_238, parameter_240, parameter_242, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_251, parameter_253, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_262, parameter_264, parameter_266, parameter_268, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_277, parameter_279, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_288, parameter_290, parameter_292, parameter_294, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_303, parameter_305, parameter_307, parameter_309, parameter_308, parameter_310, parameter_315, parameter_312, parameter_314, parameter_313, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_322, parameter_324, parameter_326, parameter_328, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_337, parameter_339, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_348, parameter_350, parameter_352, parameter_354, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_363, parameter_365, parameter_367, parameter_369, parameter_368, parameter_370, parameter_371, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2602_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_366
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_362
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_355
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_353
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_351
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_347
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_340
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_338
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_336
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_329
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_327
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_325
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_7
            paddle.to_tensor([160], dtype='int32').reshape([1]),
            # constant_6
            paddle.to_tensor([14], dtype='int32').reshape([1]),
            # parameter_306
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_302
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_295
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_291
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_287
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_280
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_276
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_267
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_265
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_261
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_252
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_250
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_237
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_235
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_228
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_226
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_200
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_198
            paddle.uniform([1, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_187
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_185
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([1, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_5
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([28], dtype='int32').reshape([1]),
            # parameter_168
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_142
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_140
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_112
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_3
            paddle.to_tensor([32], dtype='int32').reshape([1]),
            # constant_2
            paddle.to_tensor([56], dtype='int32').reshape([1]),
            # parameter_82
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_30
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_26
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_17
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_0
            paddle.uniform([32, 3, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([32, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([32, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([32, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_40
            paddle.uniform([32, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_42
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_46
            paddle.uniform([32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([256, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([32, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([32, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_66
            paddle.uniform([32, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_70
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([32, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([64, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_98
            paddle.uniform([64, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_100
            paddle.uniform([64, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_110
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([64, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([64, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([64, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_132
            paddle.uniform([64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([512, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([64, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_150
            paddle.uniform([64, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([64, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_162
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([64, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([160, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([160, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([160, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_192
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([640, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([640, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([160, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_210
            paddle.uniform([160, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_212
            paddle.uniform([160, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_216
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_222
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([640, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([640, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([160, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_233
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_236
            paddle.uniform([160, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([160, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_240
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_242
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_248
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([640, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([640, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([160, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_255
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_262
            paddle.uniform([160, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([160, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_266
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_270
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_274
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([640, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([640, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([160, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_281
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_285
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([160, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_290
            paddle.uniform([160, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_296
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_300
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([640, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_303
            paddle.uniform([640, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([160, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_307
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_309
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([256, 160, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_315
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_322
            paddle.uniform([256, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_324
            paddle.uniform([256, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_326
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_328
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_330
            paddle.uniform([256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1024, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_341
            paddle.uniform([256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_345
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_348
            paddle.uniform([256, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_350
            paddle.uniform([256, 1, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_352
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_354
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_356
            paddle.uniform([256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_360
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_363
            paddle.uniform([1024, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_365
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_367
            paddle.uniform([256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_369
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([256, 1000], dtype='float16', min=0, max=0.5),
            # parameter_371
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_366
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[1, 1024, 1, 1], dtype='float16'),
            # parameter_362
            paddle.static.InputSpec(shape=[1, 1024, 1, 1], dtype='float16'),
            # parameter_355
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_353
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_351
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_349
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_347
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_340
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_338
            paddle.static.InputSpec(shape=[1, 1024, 1, 1], dtype='float16'),
            # parameter_336
            paddle.static.InputSpec(shape=[1, 1024, 1, 1], dtype='float16'),
            # parameter_329
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_327
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_325
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_323
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_321
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_306
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_302
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_295
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_291
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_287
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_280
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_276
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_267
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_265
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_261
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_252
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_250
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_237
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_235
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_228
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_226
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_202
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_200
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_198
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_187
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_185
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float16'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_168
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_142
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_140
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_112
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_82
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_30
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_26
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_17
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 7, 7], dtype='float16'),
            # parameter_5
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_12
            paddle.static.InputSpec(shape=[32, 1, 5, 5], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[32, 1, 7, 7], dtype='float16'),
            # parameter_16
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256, 32, 1, 1], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[32, 256, 1, 1], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[32, 1, 5, 5], dtype='float16'),
            # parameter_40
            paddle.static.InputSpec(shape=[32, 1, 7, 7], dtype='float16'),
            # parameter_42
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_46
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[256, 32, 1, 1], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[32, 256, 1, 1], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[32, 1, 5, 5], dtype='float16'),
            # parameter_66
            paddle.static.InputSpec(shape=[32, 1, 7, 7], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_70
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[256, 32, 1, 1], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[32, 256, 1, 1], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_98
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float16'),
            # parameter_100
            paddle.static.InputSpec(shape=[64, 1, 7, 7], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float16'),
            # parameter_110
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512, 64, 1, 1], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[64, 512, 1, 1], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[64, 1, 7, 7], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_132
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[512, 64, 1, 1], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[64, 512, 1, 1], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_150
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[64, 1, 7, 7], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float16'),
            # parameter_162
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512, 64, 1, 1], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[64, 512, 1, 1], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[160, 64, 3, 3], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_192
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_210
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float16'),
            # parameter_212
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_216
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_222
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_233
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_236
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float16'),
            # parameter_240
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_242
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_248
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float16'),
            # parameter_255
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_262
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float16'),
            # parameter_266
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_270
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_274
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float16'),
            # parameter_281
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_285
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float16'),
            # parameter_290
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_296
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_300
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float16'),
            # parameter_303
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float16'),
            # parameter_307
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            # parameter_309
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[256, 160, 3, 3], dtype='float16'),
            # parameter_315
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_322
            paddle.static.InputSpec(shape=[256, 1, 5, 5], dtype='float16'),
            # parameter_324
            paddle.static.InputSpec(shape=[256, 1, 7, 7], dtype='float16'),
            # parameter_326
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_328
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_330
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_337
            paddle.static.InputSpec(shape=[1024, 1, 3, 3], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_341
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float16'),
            # parameter_345
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_348
            paddle.static.InputSpec(shape=[256, 1, 5, 5], dtype='float16'),
            # parameter_350
            paddle.static.InputSpec(shape=[256, 1, 7, 7], dtype='float16'),
            # parameter_352
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_354
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_356
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float16'),
            # parameter_360
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_363
            paddle.static.InputSpec(shape=[1024, 1, 3, 3], dtype='float16'),
            # parameter_365
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_367
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float16'),
            # parameter_369
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[256, 1000], dtype='float16'),
            # parameter_371
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