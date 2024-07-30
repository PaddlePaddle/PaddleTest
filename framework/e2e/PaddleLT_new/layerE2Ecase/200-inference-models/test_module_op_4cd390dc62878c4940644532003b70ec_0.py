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
    return [521][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_823_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_286, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, feed_1, feed_0):

        # pd_op.cast: (-1x3x640x640xf16) <- (-1x3x640x640xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x32x320x320xf16) <- (-1x3x640x640xf16, 32x3x6x6xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [2, 2], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x320x320xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x320x320xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x32x320x320xf16) <- (-1x32x320x320xf16)
        sigmoid_0 = paddle.nn.functional.sigmoid(batch_norm__0)

        # pd_op.multiply_: (-1x32x320x320xf16) <- (-1x32x320x320xf16, -1x32x320x320xf16)
        multiply__0 = paddle._C_ops.multiply_(batch_norm__0, sigmoid_0)

        # pd_op.conv2d: (-1x64x160x160xf16) <- (-1x32x320x320xf16, 64x32x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(multiply__0, parameter_5, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x160x160xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x160x160xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x160x160xf16) <- (-1x64x160x160xf16)
        sigmoid_1 = paddle.nn.functional.sigmoid(batch_norm__6)

        # pd_op.multiply_: (-1x64x160x160xf16) <- (-1x64x160x160xf16, -1x64x160x160xf16)
        multiply__1 = paddle._C_ops.multiply_(batch_norm__6, sigmoid_1)

        # pd_op.conv2d: (-1x32x160x160xf16) <- (-1x64x160x160xf16, 32x64x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(multiply__1, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x160x160xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x160x160xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x32x160x160xf16) <- (-1x32x160x160xf16)
        sigmoid_2 = paddle.nn.functional.sigmoid(batch_norm__12)

        # pd_op.multiply_: (-1x32x160x160xf16) <- (-1x32x160x160xf16, -1x32x160x160xf16)
        multiply__2 = paddle._C_ops.multiply_(batch_norm__12, sigmoid_2)

        # pd_op.conv2d: (-1x32x160x160xf16) <- (-1x32x160x160xf16, 32x32x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(multiply__2, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x160x160xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x160x160xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x32x160x160xf16) <- (-1x32x160x160xf16)
        sigmoid_3 = paddle.nn.functional.sigmoid(batch_norm__18)

        # pd_op.multiply_: (-1x32x160x160xf16) <- (-1x32x160x160xf16, -1x32x160x160xf16)
        multiply__3 = paddle._C_ops.multiply_(batch_norm__18, sigmoid_3)

        # pd_op.conv2d: (-1x32x160x160xf16) <- (-1x32x160x160xf16, 32x32x3x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(multiply__3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x160x160xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x160x160xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x32x160x160xf16) <- (-1x32x160x160xf16)
        sigmoid_4 = paddle.nn.functional.sigmoid(batch_norm__24)

        # pd_op.multiply_: (-1x32x160x160xf16) <- (-1x32x160x160xf16, -1x32x160x160xf16)
        multiply__4 = paddle._C_ops.multiply_(batch_norm__24, sigmoid_4)

        # pd_op.add_: (-1x32x160x160xf16) <- (-1x32x160x160xf16, -1x32x160x160xf16)
        add__0 = paddle._C_ops.add_(multiply__4, multiply__2)

        # pd_op.conv2d: (-1x32x160x160xf16) <- (-1x64x160x160xf16, 32x64x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(multiply__1, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x160x160xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x160x160xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x32x160x160xf16) <- (-1x32x160x160xf16)
        sigmoid_5 = paddle.nn.functional.sigmoid(batch_norm__30)

        # pd_op.multiply_: (-1x32x160x160xf16) <- (-1x32x160x160xf16, -1x32x160x160xf16)
        multiply__5 = paddle._C_ops.multiply_(batch_norm__30, sigmoid_5)

        # builtin.combine: ([-1x32x160x160xf16, -1x32x160x160xf16]) <- (-1x32x160x160xf16, -1x32x160x160xf16)
        combine_0 = [add__0, multiply__5]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x64x160x160xf16) <- ([-1x32x160x160xf16, -1x32x160x160xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (-1x64x160x160xf16) <- (-1x64x160x160xf16, 64x64x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(concat_0, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x160x160xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x160x160xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x160x160xf16) <- (-1x64x160x160xf16)
        sigmoid_6 = paddle.nn.functional.sigmoid(batch_norm__36)

        # pd_op.multiply_: (-1x64x160x160xf16) <- (-1x64x160x160xf16, -1x64x160x160xf16)
        multiply__6 = paddle._C_ops.multiply_(batch_norm__36, sigmoid_6)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x64x160x160xf16, 128x64x3x3xf16)
        conv2d_7 = paddle._C_ops.conv2d(multiply__6, parameter_35, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        sigmoid_7 = paddle.nn.functional.sigmoid(batch_norm__42)

        # pd_op.multiply_: (-1x128x80x80xf16) <- (-1x128x80x80xf16, -1x128x80x80xf16)
        multiply__7 = paddle._C_ops.multiply_(batch_norm__42, sigmoid_7)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x128x80x80xf16, 64x128x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(multiply__7, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_8 = paddle.nn.functional.sigmoid(batch_norm__48)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__8 = paddle._C_ops.multiply_(batch_norm__48, sigmoid_8)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x64x80x80xf16, 64x64x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(multiply__8, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_9 = paddle.nn.functional.sigmoid(batch_norm__54)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__9 = paddle._C_ops.multiply_(batch_norm__54, sigmoid_9)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x64x80x80xf16, 64x64x3x3xf16)
        conv2d_10 = paddle._C_ops.conv2d(multiply__9, parameter_50, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_10 = paddle.nn.functional.sigmoid(batch_norm__60)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__10 = paddle._C_ops.multiply_(batch_norm__60, sigmoid_10)

        # pd_op.add_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        add__1 = paddle._C_ops.add_(multiply__10, multiply__8)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x64x80x80xf16, 64x64x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(add__1, parameter_55, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_11 = paddle.nn.functional.sigmoid(batch_norm__66)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__11 = paddle._C_ops.multiply_(batch_norm__66, sigmoid_11)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x64x80x80xf16, 64x64x3x3xf16)
        conv2d_12 = paddle._C_ops.conv2d(multiply__11, parameter_60, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_12 = paddle.nn.functional.sigmoid(batch_norm__72)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__12 = paddle._C_ops.multiply_(batch_norm__72, sigmoid_12)

        # pd_op.add_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        add__2 = paddle._C_ops.add_(multiply__12, add__1)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x128x80x80xf16, 64x128x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(multiply__7, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_13 = paddle.nn.functional.sigmoid(batch_norm__78)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__13 = paddle._C_ops.multiply_(batch_norm__78, sigmoid_13)

        # builtin.combine: ([-1x64x80x80xf16, -1x64x80x80xf16]) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        combine_1 = [add__2, multiply__13]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x80x80xf16) <- ([-1x64x80x80xf16, -1x64x80x80xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(concat_1, parameter_70, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        sigmoid_14 = paddle.nn.functional.sigmoid(batch_norm__84)

        # pd_op.multiply_: (-1x128x80x80xf16) <- (-1x128x80x80xf16, -1x128x80x80xf16)
        multiply__14 = paddle._C_ops.multiply_(batch_norm__84, sigmoid_14)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x128x80x80xf16, 256x128x3x3xf16)
        conv2d_15 = paddle._C_ops.conv2d(multiply__14, parameter_75, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        sigmoid_15 = paddle.nn.functional.sigmoid(batch_norm__90)

        # pd_op.multiply_: (-1x256x40x40xf16) <- (-1x256x40x40xf16, -1x256x40x40xf16)
        multiply__15 = paddle._C_ops.multiply_(batch_norm__90, sigmoid_15)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x256x40x40xf16, 128x256x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(multiply__15, parameter_80, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_16 = paddle.nn.functional.sigmoid(batch_norm__96)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__16 = paddle._C_ops.multiply_(batch_norm__96, sigmoid_16)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(multiply__16, parameter_85, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_17 = paddle.nn.functional.sigmoid(batch_norm__102)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__17 = paddle._C_ops.multiply_(batch_norm__102, sigmoid_17)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x3x3xf16)
        conv2d_18 = paddle._C_ops.conv2d(multiply__17, parameter_90, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_18 = paddle.nn.functional.sigmoid(batch_norm__108)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__18 = paddle._C_ops.multiply_(batch_norm__108, sigmoid_18)

        # pd_op.add_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        add__3 = paddle._C_ops.add_(multiply__18, multiply__16)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(add__3, parameter_95, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_19 = paddle.nn.functional.sigmoid(batch_norm__114)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__19 = paddle._C_ops.multiply_(batch_norm__114, sigmoid_19)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x3x3xf16)
        conv2d_20 = paddle._C_ops.conv2d(multiply__19, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_20 = paddle.nn.functional.sigmoid(batch_norm__120)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__20 = paddle._C_ops.multiply_(batch_norm__120, sigmoid_20)

        # pd_op.add_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        add__4 = paddle._C_ops.add_(multiply__20, add__3)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(add__4, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_21 = paddle.nn.functional.sigmoid(batch_norm__126)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__21 = paddle._C_ops.multiply_(batch_norm__126, sigmoid_21)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(multiply__21, parameter_110, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_22 = paddle.nn.functional.sigmoid(batch_norm__132)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__22 = paddle._C_ops.multiply_(batch_norm__132, sigmoid_22)

        # pd_op.add_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        add__5 = paddle._C_ops.add_(multiply__22, add__4)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x256x40x40xf16, 128x256x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(multiply__15, parameter_115, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_23 = paddle.nn.functional.sigmoid(batch_norm__138)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__23 = paddle._C_ops.multiply_(batch_norm__138, sigmoid_23)

        # builtin.combine: ([-1x128x40x40xf16, -1x128x40x40xf16]) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        combine_2 = [add__5, multiply__23]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x40x40xf16) <- ([-1x128x40x40xf16, -1x128x40x40xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_2)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(concat_2, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        sigmoid_24 = paddle.nn.functional.sigmoid(batch_norm__144)

        # pd_op.multiply_: (-1x256x40x40xf16) <- (-1x256x40x40xf16, -1x256x40x40xf16)
        multiply__24 = paddle._C_ops.multiply_(batch_norm__144, sigmoid_24)

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x256x40x40xf16, 512x256x3x3xf16)
        conv2d_25 = paddle._C_ops.conv2d(multiply__24, parameter_125, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        sigmoid_25 = paddle.nn.functional.sigmoid(batch_norm__150)

        # pd_op.multiply_: (-1x512x20x20xf16) <- (-1x512x20x20xf16, -1x512x20x20xf16)
        multiply__25 = paddle._C_ops.multiply_(batch_norm__150, sigmoid_25)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x512x20x20xf16, 256x512x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(multiply__25, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_26 = paddle.nn.functional.sigmoid(batch_norm__156)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__26 = paddle._C_ops.multiply_(batch_norm__156, sigmoid_26)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x256x20x20xf16, 256x256x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(multiply__26, parameter_135, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_27 = paddle.nn.functional.sigmoid(batch_norm__162)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__27 = paddle._C_ops.multiply_(batch_norm__162, sigmoid_27)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x256x20x20xf16, 256x256x3x3xf16)
        conv2d_28 = paddle._C_ops.conv2d(multiply__27, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_28 = paddle.nn.functional.sigmoid(batch_norm__168)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__28 = paddle._C_ops.multiply_(batch_norm__168, sigmoid_28)

        # pd_op.add_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        add__6 = paddle._C_ops.add_(multiply__28, multiply__26)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x512x20x20xf16, 256x512x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(multiply__25, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_29 = paddle.nn.functional.sigmoid(batch_norm__174)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__29 = paddle._C_ops.multiply_(batch_norm__174, sigmoid_29)

        # builtin.combine: ([-1x256x20x20xf16, -1x256x20x20xf16]) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        combine_3 = [add__6, multiply__29]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x20x20xf16) <- ([-1x256x20x20xf16, -1x256x20x20xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_3)

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(concat_3, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        sigmoid_30 = paddle.nn.functional.sigmoid(batch_norm__180)

        # pd_op.multiply_: (-1x512x20x20xf16) <- (-1x512x20x20xf16, -1x512x20x20xf16)
        multiply__30 = paddle._C_ops.multiply_(batch_norm__180, sigmoid_30)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x512x20x20xf16, 256x512x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(multiply__30, parameter_155, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_31 = paddle.nn.functional.sigmoid(batch_norm__186)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__31 = paddle._C_ops.multiply_(batch_norm__186, sigmoid_31)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [5, 5]

        # pd_op.pool2d: (-1x256x20x20xf16) <- (-1x256x20x20xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(multiply__31, full_int_array_0, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [5, 5]

        # pd_op.pool2d: (-1x256x20x20xf16) <- (-1x256x20x20xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(pool2d_0, full_int_array_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [5, 5]

        # pd_op.pool2d: (-1x256x20x20xf16) <- (-1x256x20x20xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(pool2d_1, full_int_array_2, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # builtin.combine: ([-1x256x20x20xf16, -1x256x20x20xf16, -1x256x20x20xf16, -1x256x20x20xf16]) <- (-1x256x20x20xf16, -1x256x20x20xf16, -1x256x20x20xf16, -1x256x20x20xf16)
        combine_4 = [multiply__31, pool2d_0, pool2d_1, pool2d_2]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024x20x20xf16) <- ([-1x256x20x20xf16, -1x256x20x20xf16, -1x256x20x20xf16, -1x256x20x20xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_4)

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x1024x20x20xf16, 512x1024x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(concat_4, parameter_160, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        sigmoid_32 = paddle.nn.functional.sigmoid(batch_norm__192)

        # pd_op.multiply_: (-1x512x20x20xf16) <- (-1x512x20x20xf16, -1x512x20x20xf16)
        multiply__32 = paddle._C_ops.multiply_(batch_norm__192, sigmoid_32)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x512x20x20xf16, 256x512x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(multiply__32, parameter_165, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_33 = paddle.nn.functional.sigmoid(batch_norm__198)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__33 = paddle._C_ops.multiply_(batch_norm__198, sigmoid_33)

        # pd_op.nearest_interp: (-1x256x40x40xf16) <- (-1x256x20x20xf16, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(multiply__33, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # builtin.combine: ([-1x256x40x40xf16, -1x256x40x40xf16]) <- (-1x256x40x40xf16, -1x256x40x40xf16)
        combine_5 = [nearest_interp_0, multiply__24]

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x40x40xf16) <- ([-1x256x40x40xf16, -1x256x40x40xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_5)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x512x40x40xf16, 128x512x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(concat_5, parameter_170, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_34 = paddle.nn.functional.sigmoid(batch_norm__204)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__34 = paddle._C_ops.multiply_(batch_norm__204, sigmoid_34)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(multiply__34, parameter_175, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_35 = paddle.nn.functional.sigmoid(batch_norm__210)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__35 = paddle._C_ops.multiply_(batch_norm__210, sigmoid_35)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x3x3xf16)
        conv2d_36 = paddle._C_ops.conv2d(multiply__35, parameter_180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_181, parameter_182, parameter_183, parameter_184, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_36 = paddle.nn.functional.sigmoid(batch_norm__216)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__36 = paddle._C_ops.multiply_(batch_norm__216, sigmoid_36)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x512x40x40xf16, 128x512x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(concat_5, parameter_185, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_37 = paddle.nn.functional.sigmoid(batch_norm__222)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__37 = paddle._C_ops.multiply_(batch_norm__222, sigmoid_37)

        # builtin.combine: ([-1x128x40x40xf16, -1x128x40x40xf16]) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        combine_6 = [multiply__36, multiply__37]

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x40x40xf16) <- ([-1x128x40x40xf16, -1x128x40x40xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_6)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(concat_6, parameter_190, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_191, parameter_192, parameter_193, parameter_194, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        sigmoid_38 = paddle.nn.functional.sigmoid(batch_norm__228)

        # pd_op.multiply_: (-1x256x40x40xf16) <- (-1x256x40x40xf16, -1x256x40x40xf16)
        multiply__38 = paddle._C_ops.multiply_(batch_norm__228, sigmoid_38)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x256x40x40xf16, 128x256x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(multiply__38, parameter_195, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_196, parameter_197, parameter_198, parameter_199, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_39 = paddle.nn.functional.sigmoid(batch_norm__234)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__39 = paddle._C_ops.multiply_(batch_norm__234, sigmoid_39)

        # pd_op.nearest_interp: (-1x128x80x80xf16) <- (-1x128x40x40xf16, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(multiply__39, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # builtin.combine: ([-1x128x80x80xf16, -1x128x80x80xf16]) <- (-1x128x80x80xf16, -1x128x80x80xf16)
        combine_7 = [nearest_interp_1, multiply__14]

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x80x80xf16) <- ([-1x128x80x80xf16, -1x128x80x80xf16], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_7)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x256x80x80xf16, 64x256x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(concat_7, parameter_200, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_201, parameter_202, parameter_203, parameter_204, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_40 = paddle.nn.functional.sigmoid(batch_norm__240)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__40 = paddle._C_ops.multiply_(batch_norm__240, sigmoid_40)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x64x80x80xf16, 64x64x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(multiply__40, parameter_205, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_206, parameter_207, parameter_208, parameter_209, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_41 = paddle.nn.functional.sigmoid(batch_norm__246)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__41 = paddle._C_ops.multiply_(batch_norm__246, sigmoid_41)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x64x80x80xf16, 64x64x3x3xf16)
        conv2d_42 = paddle._C_ops.conv2d(multiply__41, parameter_210, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_211, parameter_212, parameter_213, parameter_214, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_42 = paddle.nn.functional.sigmoid(batch_norm__252)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__42 = paddle._C_ops.multiply_(batch_norm__252, sigmoid_42)

        # pd_op.conv2d: (-1x64x80x80xf16) <- (-1x256x80x80xf16, 64x256x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(concat_7, parameter_215, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x80x80xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x80x80xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x64x80x80xf16) <- (-1x64x80x80xf16)
        sigmoid_43 = paddle.nn.functional.sigmoid(batch_norm__258)

        # pd_op.multiply_: (-1x64x80x80xf16) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        multiply__43 = paddle._C_ops.multiply_(batch_norm__258, sigmoid_43)

        # builtin.combine: ([-1x64x80x80xf16, -1x64x80x80xf16]) <- (-1x64x80x80xf16, -1x64x80x80xf16)
        combine_8 = [multiply__42, multiply__43]

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x80x80xf16) <- ([-1x64x80x80xf16, -1x64x80x80xf16], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_8)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(concat_8, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_221, parameter_222, parameter_223, parameter_224, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        sigmoid_44 = paddle.nn.functional.sigmoid(batch_norm__264)

        # pd_op.multiply_: (-1x128x80x80xf16) <- (-1x128x80x80xf16, -1x128x80x80xf16)
        multiply__44 = paddle._C_ops.multiply_(batch_norm__264, sigmoid_44)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x80x80xf16, 128x128x3x3xf16)
        conv2d_45 = paddle._C_ops.conv2d(multiply__44, parameter_225, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_226, parameter_227, parameter_228, parameter_229, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_45 = paddle.nn.functional.sigmoid(batch_norm__270)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__45 = paddle._C_ops.multiply_(batch_norm__270, sigmoid_45)

        # builtin.combine: ([-1x128x40x40xf16, -1x128x40x40xf16]) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        combine_9 = [multiply__45, multiply__39]

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x40x40xf16) <- ([-1x128x40x40xf16, -1x128x40x40xf16], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_9)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x256x40x40xf16, 128x256x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(concat_9, parameter_230, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_46 = paddle.nn.functional.sigmoid(batch_norm__276)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__46 = paddle._C_ops.multiply_(batch_norm__276, sigmoid_46)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(multiply__46, parameter_235, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_47 = paddle.nn.functional.sigmoid(batch_norm__282)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__47 = paddle._C_ops.multiply_(batch_norm__282, sigmoid_47)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x128x40x40xf16, 128x128x3x3xf16)
        conv2d_48 = paddle._C_ops.conv2d(multiply__47, parameter_240, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_48 = paddle.nn.functional.sigmoid(batch_norm__288)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__48 = paddle._C_ops.multiply_(batch_norm__288, sigmoid_48)

        # pd_op.conv2d: (-1x128x40x40xf16) <- (-1x256x40x40xf16, 128x256x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(concat_9, parameter_245, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x40x40xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x40x40xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x128x40x40xf16) <- (-1x128x40x40xf16)
        sigmoid_49 = paddle.nn.functional.sigmoid(batch_norm__294)

        # pd_op.multiply_: (-1x128x40x40xf16) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        multiply__49 = paddle._C_ops.multiply_(batch_norm__294, sigmoid_49)

        # builtin.combine: ([-1x128x40x40xf16, -1x128x40x40xf16]) <- (-1x128x40x40xf16, -1x128x40x40xf16)
        combine_10 = [multiply__48, multiply__49]

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x40x40xf16) <- ([-1x128x40x40xf16, -1x128x40x40xf16], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_10)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(concat_10, parameter_250, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        sigmoid_50 = paddle.nn.functional.sigmoid(batch_norm__300)

        # pd_op.multiply_: (-1x256x40x40xf16) <- (-1x256x40x40xf16, -1x256x40x40xf16)
        multiply__50 = paddle._C_ops.multiply_(batch_norm__300, sigmoid_50)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_51 = paddle._C_ops.conv2d(multiply__50, parameter_255, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_51 = paddle.nn.functional.sigmoid(batch_norm__306)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__51 = paddle._C_ops.multiply_(batch_norm__306, sigmoid_51)

        # builtin.combine: ([-1x256x20x20xf16, -1x256x20x20xf16]) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        combine_11 = [multiply__51, multiply__33]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x20x20xf16) <- ([-1x256x20x20xf16, -1x256x20x20xf16], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_11)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x512x20x20xf16, 256x512x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(concat_11, parameter_260, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_261, parameter_262, parameter_263, parameter_264, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_52 = paddle.nn.functional.sigmoid(batch_norm__312)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__52 = paddle._C_ops.multiply_(batch_norm__312, sigmoid_52)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x256x20x20xf16, 256x256x1x1xf16)
        conv2d_53 = paddle._C_ops.conv2d(multiply__52, parameter_265, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_266, parameter_267, parameter_268, parameter_269, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_53 = paddle.nn.functional.sigmoid(batch_norm__318)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__53 = paddle._C_ops.multiply_(batch_norm__318, sigmoid_53)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x256x20x20xf16, 256x256x3x3xf16)
        conv2d_54 = paddle._C_ops.conv2d(multiply__53, parameter_270, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_271, parameter_272, parameter_273, parameter_274, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_54 = paddle.nn.functional.sigmoid(batch_norm__324)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__54 = paddle._C_ops.multiply_(batch_norm__324, sigmoid_54)

        # pd_op.conv2d: (-1x256x20x20xf16) <- (-1x512x20x20xf16, 256x512x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(concat_11, parameter_275, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x20x20xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x20x20xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_276, parameter_277, parameter_278, parameter_279, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x256x20x20xf16) <- (-1x256x20x20xf16)
        sigmoid_55 = paddle.nn.functional.sigmoid(batch_norm__330)

        # pd_op.multiply_: (-1x256x20x20xf16) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        multiply__55 = paddle._C_ops.multiply_(batch_norm__330, sigmoid_55)

        # builtin.combine: ([-1x256x20x20xf16, -1x256x20x20xf16]) <- (-1x256x20x20xf16, -1x256x20x20xf16)
        combine_12 = [multiply__54, multiply__55]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x20x20xf16) <- ([-1x256x20x20xf16, -1x256x20x20xf16], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_12)

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x1x1xf16)
        conv2d_56 = paddle._C_ops.conv2d(concat_12, parameter_280, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_281, parameter_282, parameter_283, parameter_284, True, float('0.97'), float('0.001'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.sigmoid: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        sigmoid_56 = paddle.nn.functional.sigmoid(batch_norm__336)

        # pd_op.multiply_: (-1x512x20x20xf16) <- (-1x512x20x20xf16, -1x512x20x20xf16)
        multiply__56 = paddle._C_ops.multiply_(batch_norm__336, sigmoid_56)

        # pd_op.conv2d: (-1x255x80x80xf16) <- (-1x128x80x80xf16, 255x128x1x1xf16)
        conv2d_57 = paddle._C_ops.conv2d(multiply__44, parameter_285, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 255, 1, 1]

        # pd_op.reshape: (1x255x1x1xf16, 0x255xf16) <- (255xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_286, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x255x80x80xf16) <- (-1x255x80x80xf16, 1x255x1x1xf16)
        add__7 = paddle._C_ops.add_(conv2d_57, reshape_0)

        # pd_op.conv2d: (-1x255x40x40xf16) <- (-1x256x40x40xf16, 255x256x1x1xf16)
        conv2d_58 = paddle._C_ops.conv2d(multiply__50, parameter_287, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 255, 1, 1]

        # pd_op.reshape: (1x255x1x1xf16, 0x255xf16) <- (255xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_288, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x255x40x40xf16) <- (-1x255x40x40xf16, 1x255x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_58, reshape_2)

        # pd_op.conv2d: (-1x255x20x20xf16) <- (-1x512x20x20xf16, 255x512x1x1xf16)
        conv2d_59 = paddle._C_ops.conv2d(multiply__56, parameter_289, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 255, 1, 1]

        # pd_op.reshape: (1x255x1x1xf16, 0x255xf16) <- (255xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_290, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x255x20x20xf16) <- (-1x255x20x20xf16, 1x255x1x1xf16)
        add__9 = paddle._C_ops.add_(conv2d_59, reshape_4)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_6 = [-1, 3, 85, 80, 80]

        # pd_op.reshape_: (-1x3x85x80x80xf16, 0x-1x255x80x80xf16) <- (-1x255x80x80xf16, 5xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__7, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3x80x80x85xf16) <- (-1x3x85x80x80xf16)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 1, 3, 4, 2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [1]

        # pd_op.slice: (3x2xf16) <- (3x3x2xf16, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(parameter_291, [0], full_int_array_7, full_int_array_8, [1], [0])

        # pd_op.full: (1xf16) <- ()
        full_13 = paddle._C_ops.full([1], float('0'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_14 = paddle._C_ops.full([1], float('80'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.arange: (80xf16) <- (1xf16, 1xf16, 1xf16)
        arange_0 = paddle.arange(full_13, full_14, full_15, dtype='float16')

        # pd_op.full: (1xf16) <- ()
        full_16 = paddle._C_ops.full([1], float('0'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_17 = paddle._C_ops.full([1], float('80'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_18 = paddle._C_ops.full([1], float('1'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.arange: (80xf16) <- (1xf16, 1xf16, 1xf16)
        arange_1 = paddle.arange(full_16, full_17, full_18, dtype='float16')

        # builtin.combine: ([80xf16, 80xf16]) <- (80xf16, 80xf16)
        combine_13 = [arange_0, arange_1]

        # pd_op.meshgrid: ([80x80xf16, 80x80xf16]) <- ([80xf16, 80xf16])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_13)

        # builtin.slice: (80x80xf16) <- ([80x80xf16, 80x80xf16])
        slice_1 = meshgrid_0[1]

        # builtin.slice: (80x80xf16) <- ([80x80xf16, 80x80xf16])
        slice_2 = meshgrid_0[0]

        # builtin.combine: ([80x80xf16, 80x80xf16]) <- (80x80xf16, 80x80xf16)
        combine_14 = [slice_1, slice_2]

        # pd_op.stack: (80x80x2xf16) <- ([80x80xf16, 80x80xf16])
        stack_0 = paddle._C_ops.stack(combine_14, 2)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_9 = [1, 3, 80, 80, 2]

        # pd_op.expand: (1x3x80x80x2xf16) <- (80x80x2xf16, 5xi64)
        expand_0 = paddle._C_ops.expand(stack_0, full_int_array_9)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_10 = [1, 3, 1, 1, 2]

        # pd_op.reshape_: (1x3x1x1x2xf16, 0x3x2xf16) <- (3x2xf16, 5xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_0, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_11 = [1, 3, 80, 80, 2]

        # pd_op.expand: (1x3x80x80x2xf16) <- (1x3x1x1x2xf16, 5xi64)
        expand_1 = paddle._C_ops.expand(reshape__2, full_int_array_11)

        # pd_op.sigmoid_: (-1x3x80x80x85xf16) <- (-1x3x80x80x85xf16)
        sigmoid__0 = paddle._C_ops.sigmoid_(transpose_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [2]

        # pd_op.slice: (-1x3x80x80x2xf16) <- (-1x3x80x80x85xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(sigmoid__0, [4], full_int_array_12, full_int_array_13, [1], [])

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(slice_3, full_19, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(scale__0, full_20, float('-0.5'), True)

        # pd_op.add_: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, 1x3x80x80x2xf16)
        add__10 = paddle._C_ops.add_(scale__1, expand_0)

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full([1], float('8'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(add__10, full_21, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [4]

        # pd_op.slice: (-1x3x80x80x2xf16) <- (-1x3x80x80x85xf16, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(sigmoid__0, [4], full_int_array_14, full_int_array_15, [1], [])

        # pd_op.full: (1xf32) <- ()
        full_22 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(slice_4, full_22, float('0'), True)

        # pd_op.full: (xf16) <- ()
        full_23 = paddle._C_ops.full([], float('2'), paddle.float16, paddle.framework._current_expected_place())

        # pd_op.elementwise_pow: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, xf16)
        elementwise_pow_0 = paddle.pow(scale__3, full_23)

        # pd_op.multiply_: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, 1x3x80x80x2xf16)
        multiply__57 = paddle._C_ops.multiply_(elementwise_pow_0, expand_1)

        # pd_op.full: (1xf32) <- ()
        full_24 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, 1xf32)
        scale_0 = paddle._C_ops.scale(multiply__57, full_24, float('0'), True)

        # pd_op.subtract: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, -1x3x80x80x2xf16)
        subtract_0 = scale__2 - scale_0

        # pd_op.full: (1xf32) <- ()
        full_25 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(multiply__57, full_25, float('0'), True)

        # pd_op.add_: (-1x3x80x80x2xf16) <- (-1x3x80x80x2xf16, -1x3x80x80x2xf16)
        add__11 = paddle._C_ops.add_(scale__2, scale__4)

        # builtin.combine: ([-1x3x80x80x2xf16, -1x3x80x80x2xf16]) <- (-1x3x80x80x2xf16, -1x3x80x80x2xf16)
        combine_15 = [subtract_0, add__11]

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x3x80x80x4xf16) <- ([-1x3x80x80x2xf16, -1x3x80x80x2xf16], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_15, full_26)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [2147483647]

        # pd_op.slice: (-1x3x80x80x80xf16) <- (-1x3x80x80x85xf16, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(sigmoid__0, [4], full_int_array_16, full_int_array_17, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [5]

        # pd_op.slice: (-1x3x80x80xf16) <- (-1x3x80x80x85xf16, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(sigmoid__0, [4], full_int_array_18, full_int_array_19, [1], [4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [-1]

        # pd_op.unsqueeze_: (-1x3x80x80x1xf16, None) <- (-1x3x80x80xf16, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(slice_6, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x3x80x80x80xf16) <- (-1x3x80x80x80xf16, -1x3x80x80x1xf16)
        multiply__58 = paddle._C_ops.multiply_(slice_5, unsqueeze__0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_21 = [-1, 19200, 4]

        # pd_op.reshape_: (-1x19200x4xf16, 0x-1x3x80x80x4xf16) <- (-1x3x80x80x4xf16, 3xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_13, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_22 = [-1, 19200, 80]

        # pd_op.reshape_: (-1x19200x80xf16, 0x-1x3x80x80x80xf16) <- (-1x3x80x80x80xf16, 3xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(multiply__58, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x19200xf16) <- (-1x19200x80xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__6, [0, 2, 1])

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_23 = [-1, 3, 85, 40, 40]

        # pd_op.reshape_: (-1x3x85x40x40xf16, 0x-1x255x40x40xf16) <- (-1x255x40x40xf16, 5xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__8, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3x40x40x85xf16) <- (-1x3x85x40x40xf16)
        transpose_2 = paddle._C_ops.transpose(reshape__8, [0, 1, 3, 4, 2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [2]

        # pd_op.slice: (3x2xf16) <- (3x3x2xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(parameter_291, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.full: (1xf16) <- ()
        full_27 = paddle._C_ops.full([1], float('0'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_28 = paddle._C_ops.full([1], float('40'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_29 = paddle._C_ops.full([1], float('1'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.arange: (40xf16) <- (1xf16, 1xf16, 1xf16)
        arange_2 = paddle.arange(full_27, full_28, full_29, dtype='float16')

        # pd_op.full: (1xf16) <- ()
        full_30 = paddle._C_ops.full([1], float('0'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_31 = paddle._C_ops.full([1], float('40'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.arange: (40xf16) <- (1xf16, 1xf16, 1xf16)
        arange_3 = paddle.arange(full_30, full_31, full_32, dtype='float16')

        # builtin.combine: ([40xf16, 40xf16]) <- (40xf16, 40xf16)
        combine_16 = [arange_2, arange_3]

        # pd_op.meshgrid: ([40x40xf16, 40x40xf16]) <- ([40xf16, 40xf16])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_16)

        # builtin.slice: (40x40xf16) <- ([40x40xf16, 40x40xf16])
        slice_8 = meshgrid_1[1]

        # builtin.slice: (40x40xf16) <- ([40x40xf16, 40x40xf16])
        slice_9 = meshgrid_1[0]

        # builtin.combine: ([40x40xf16, 40x40xf16]) <- (40x40xf16, 40x40xf16)
        combine_17 = [slice_8, slice_9]

        # pd_op.stack: (40x40x2xf16) <- ([40x40xf16, 40x40xf16])
        stack_1 = paddle._C_ops.stack(combine_17, 2)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_26 = [1, 3, 40, 40, 2]

        # pd_op.expand: (1x3x40x40x2xf16) <- (40x40x2xf16, 5xi64)
        expand_2 = paddle._C_ops.expand(stack_1, full_int_array_26)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_27 = [1, 3, 1, 1, 2]

        # pd_op.reshape_: (1x3x1x1x2xf16, 0x3x2xf16) <- (3x2xf16, 5xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_7, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_28 = [1, 3, 40, 40, 2]

        # pd_op.expand: (1x3x40x40x2xf16) <- (1x3x1x1x2xf16, 5xi64)
        expand_3 = paddle._C_ops.expand(reshape__10, full_int_array_28)

        # pd_op.sigmoid_: (-1x3x40x40x85xf16) <- (-1x3x40x40x85xf16)
        sigmoid__1 = paddle._C_ops.sigmoid_(transpose_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [2]

        # pd_op.slice: (-1x3x40x40x2xf16) <- (-1x3x40x40x85xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(sigmoid__1, [4], full_int_array_29, full_int_array_30, [1], [])

        # pd_op.full: (1xf32) <- ()
        full_33 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(slice_10, full_33, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_34 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(scale__5, full_34, float('-0.5'), True)

        # pd_op.add_: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, 1x3x40x40x2xf16)
        add__12 = paddle._C_ops.add_(scale__6, expand_2)

        # pd_op.full: (1xf32) <- ()
        full_35 = paddle._C_ops.full([1], float('16'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(add__12, full_35, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [4]

        # pd_op.slice: (-1x3x40x40x2xf16) <- (-1x3x40x40x85xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(sigmoid__1, [4], full_int_array_31, full_int_array_32, [1], [])

        # pd_op.full: (1xf32) <- ()
        full_36 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, 1xf32)
        scale__8 = paddle._C_ops.scale_(slice_11, full_36, float('0'), True)

        # pd_op.full: (xf16) <- ()
        full_37 = paddle._C_ops.full([], float('2'), paddle.float16, paddle.framework._current_expected_place())

        # pd_op.elementwise_pow: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, xf16)
        elementwise_pow_1 = paddle.pow(scale__8, full_37)

        # pd_op.multiply_: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, 1x3x40x40x2xf16)
        multiply__59 = paddle._C_ops.multiply_(elementwise_pow_1, expand_3)

        # pd_op.full: (1xf32) <- ()
        full_38 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply__59, full_38, float('0'), True)

        # pd_op.subtract: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, -1x3x40x40x2xf16)
        subtract_1 = scale__7 - scale_1

        # pd_op.full: (1xf32) <- ()
        full_39 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, 1xf32)
        scale__9 = paddle._C_ops.scale_(multiply__59, full_39, float('0'), True)

        # pd_op.add_: (-1x3x40x40x2xf16) <- (-1x3x40x40x2xf16, -1x3x40x40x2xf16)
        add__13 = paddle._C_ops.add_(scale__7, scale__9)

        # builtin.combine: ([-1x3x40x40x2xf16, -1x3x40x40x2xf16]) <- (-1x3x40x40x2xf16, -1x3x40x40x2xf16)
        combine_18 = [subtract_1, add__13]

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x3x40x40x4xf16) <- ([-1x3x40x40x2xf16, -1x3x40x40x2xf16], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_18, full_40)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [2147483647]

        # pd_op.slice: (-1x3x40x40x80xf16) <- (-1x3x40x40x85xf16, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(sigmoid__1, [4], full_int_array_33, full_int_array_34, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [5]

        # pd_op.slice: (-1x3x40x40xf16) <- (-1x3x40x40x85xf16, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(sigmoid__1, [4], full_int_array_35, full_int_array_36, [1], [4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [-1]

        # pd_op.unsqueeze_: (-1x3x40x40x1xf16, None) <- (-1x3x40x40xf16, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(slice_13, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x3x40x40x80xf16) <- (-1x3x40x40x80xf16, -1x3x40x40x1xf16)
        multiply__60 = paddle._C_ops.multiply_(slice_12, unsqueeze__2)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_38 = [-1, 4800, 4]

        # pd_op.reshape_: (-1x4800x4xf16, 0x-1x3x40x40x4xf16) <- (-1x3x40x40x4xf16, 3xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_14, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_39 = [-1, 4800, 80]

        # pd_op.reshape_: (-1x4800x80xf16, 0x-1x3x40x40x80xf16) <- (-1x3x40x40x80xf16, 3xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(multiply__60, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x4800xf16) <- (-1x4800x80xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__14, [0, 2, 1])

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_40 = [-1, 3, 85, 20, 20]

        # pd_op.reshape_: (-1x3x85x20x20xf16, 0x-1x255x20x20xf16) <- (-1x255x20x20xf16, 5xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__9, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3x20x20x85xf16) <- (-1x3x85x20x20xf16)
        transpose_4 = paddle._C_ops.transpose(reshape__16, [0, 1, 3, 4, 2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [3]

        # pd_op.slice: (3x2xf16) <- (3x3x2xf16, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(parameter_291, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.full: (1xf16) <- ()
        full_41 = paddle._C_ops.full([1], float('0'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_42 = paddle._C_ops.full([1], float('20'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_43 = paddle._C_ops.full([1], float('1'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.arange: (20xf16) <- (1xf16, 1xf16, 1xf16)
        arange_4 = paddle.arange(full_41, full_42, full_43, dtype='float16')

        # pd_op.full: (1xf16) <- ()
        full_44 = paddle._C_ops.full([1], float('0'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_45 = paddle._C_ops.full([1], float('20'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.full: (1xf16) <- ()
        full_46 = paddle._C_ops.full([1], float('1'), paddle.float16, paddle.core.CPUPlace())

        # pd_op.arange: (20xf16) <- (1xf16, 1xf16, 1xf16)
        arange_5 = paddle.arange(full_44, full_45, full_46, dtype='float16')

        # builtin.combine: ([20xf16, 20xf16]) <- (20xf16, 20xf16)
        combine_19 = [arange_4, arange_5]

        # pd_op.meshgrid: ([20x20xf16, 20x20xf16]) <- ([20xf16, 20xf16])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_19)

        # builtin.slice: (20x20xf16) <- ([20x20xf16, 20x20xf16])
        slice_15 = meshgrid_2[1]

        # builtin.slice: (20x20xf16) <- ([20x20xf16, 20x20xf16])
        slice_16 = meshgrid_2[0]

        # builtin.combine: ([20x20xf16, 20x20xf16]) <- (20x20xf16, 20x20xf16)
        combine_20 = [slice_15, slice_16]

        # pd_op.stack: (20x20x2xf16) <- ([20x20xf16, 20x20xf16])
        stack_2 = paddle._C_ops.stack(combine_20, 2)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_43 = [1, 3, 20, 20, 2]

        # pd_op.expand: (1x3x20x20x2xf16) <- (20x20x2xf16, 5xi64)
        expand_4 = paddle._C_ops.expand(stack_2, full_int_array_43)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_44 = [1, 3, 1, 1, 2]

        # pd_op.reshape_: (1x3x1x1x2xf16, 0x3x2xf16) <- (3x2xf16, 5xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_14, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_45 = [1, 3, 20, 20, 2]

        # pd_op.expand: (1x3x20x20x2xf16) <- (1x3x1x1x2xf16, 5xi64)
        expand_5 = paddle._C_ops.expand(reshape__18, full_int_array_45)

        # pd_op.sigmoid_: (-1x3x20x20x85xf16) <- (-1x3x20x20x85xf16)
        sigmoid__2 = paddle._C_ops.sigmoid_(transpose_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [2]

        # pd_op.slice: (-1x3x20x20x2xf16) <- (-1x3x20x20x85xf16, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(sigmoid__2, [4], full_int_array_46, full_int_array_47, [1], [])

        # pd_op.full: (1xf32) <- ()
        full_47 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, 1xf32)
        scale__10 = paddle._C_ops.scale_(slice_17, full_47, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_48 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, 1xf32)
        scale__11 = paddle._C_ops.scale_(scale__10, full_48, float('-0.5'), True)

        # pd_op.add_: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, 1x3x20x20x2xf16)
        add__14 = paddle._C_ops.add_(scale__11, expand_4)

        # pd_op.full: (1xf32) <- ()
        full_49 = paddle._C_ops.full([1], float('32'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, 1xf32)
        scale__12 = paddle._C_ops.scale_(add__14, full_49, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [4]

        # pd_op.slice: (-1x3x20x20x2xf16) <- (-1x3x20x20x85xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(sigmoid__2, [4], full_int_array_48, full_int_array_49, [1], [])

        # pd_op.full: (1xf32) <- ()
        full_50 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, 1xf32)
        scale__13 = paddle._C_ops.scale_(slice_18, full_50, float('0'), True)

        # pd_op.full: (xf16) <- ()
        full_51 = paddle._C_ops.full([], float('2'), paddle.float16, paddle.framework._current_expected_place())

        # pd_op.elementwise_pow: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, xf16)
        elementwise_pow_2 = paddle.pow(scale__13, full_51)

        # pd_op.multiply_: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, 1x3x20x20x2xf16)
        multiply__61 = paddle._C_ops.multiply_(elementwise_pow_2, expand_5)

        # pd_op.full: (1xf32) <- ()
        full_52 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, 1xf32)
        scale_2 = paddle._C_ops.scale(multiply__61, full_52, float('0'), True)

        # pd_op.subtract: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, -1x3x20x20x2xf16)
        subtract_2 = scale__12 - scale_2

        # pd_op.full: (1xf32) <- ()
        full_53 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, 1xf32)
        scale__14 = paddle._C_ops.scale_(multiply__61, full_53, float('0'), True)

        # pd_op.add_: (-1x3x20x20x2xf16) <- (-1x3x20x20x2xf16, -1x3x20x20x2xf16)
        add__15 = paddle._C_ops.add_(scale__12, scale__14)

        # builtin.combine: ([-1x3x20x20x2xf16, -1x3x20x20x2xf16]) <- (-1x3x20x20x2xf16, -1x3x20x20x2xf16)
        combine_21 = [subtract_2, add__15]

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x3x20x20x4xf16) <- ([-1x3x20x20x2xf16, -1x3x20x20x2xf16], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_21, full_54)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [2147483647]

        # pd_op.slice: (-1x3x20x20x80xf16) <- (-1x3x20x20x85xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(sigmoid__2, [4], full_int_array_50, full_int_array_51, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [5]

        # pd_op.slice: (-1x3x20x20xf16) <- (-1x3x20x20x85xf16, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(sigmoid__2, [4], full_int_array_52, full_int_array_53, [1], [4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [-1]

        # pd_op.unsqueeze_: (-1x3x20x20x1xf16, None) <- (-1x3x20x20xf16, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(slice_20, full_int_array_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x3x20x20x80xf16) <- (-1x3x20x20x80xf16, -1x3x20x20x1xf16)
        multiply__62 = paddle._C_ops.multiply_(slice_19, unsqueeze__4)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_55 = [-1, 1200, 4]

        # pd_op.reshape_: (-1x1200x4xf16, 0x-1x3x20x20x4xf16) <- (-1x3x20x20x4xf16, 3xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_15, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_56 = [-1, 1200, 80]

        # pd_op.reshape_: (-1x1200x80xf16, 0x-1x3x20x20x80xf16) <- (-1x3x20x20x80xf16, 3xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(multiply__62, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x1200xf16) <- (-1x1200x80xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__22, [0, 2, 1])

        # builtin.combine: ([-1x19200x4xf16, -1x4800x4xf16, -1x1200x4xf16]) <- (-1x19200x4xf16, -1x4800x4xf16, -1x1200x4xf16)
        combine_22 = [reshape__4, reshape__12, reshape__20]

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x25200x4xf16) <- ([-1x19200x4xf16, -1x4800x4xf16, -1x1200x4xf16], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_22, full_55)

        # builtin.combine: ([-1x80x19200xf16, -1x80x4800xf16, -1x80x1200xf16]) <- (-1x80x19200xf16, -1x80x4800xf16, -1x80x1200xf16)
        combine_23 = [transpose_1, transpose_3, transpose_5]

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x25200xf16) <- ([-1x80x19200xf16, -1x80x4800xf16, -1x80x1200xf16], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_23, full_56)

        # pd_op.cast: (-1x2xf16) <- (-1x2xf32)
        cast_1 = paddle._C_ops.cast(feed_1, paddle.float16)

        # pd_op.flip: (-1x2xf16) <- (-1x2xf16)
        flip_0 = paddle._C_ops.flip(cast_1, [-1])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_57 = [1, 2]

        # pd_op.tile: (-1x4xf16) <- (-1x2xf16, 2xi64)
        tile_0 = paddle._C_ops.tile(flip_0, full_int_array_57)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [1]

        # pd_op.unsqueeze_: (-1x1x4xf16, None) <- (-1x4xf16, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(tile_0, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.divide_: (-1x25200x4xf16) <- (-1x25200x4xf16, -1x1x4xf16)
        divide__0 = paddle._C_ops.divide_(concat_16, unsqueeze__6)

        # pd_op.cast: (-1x25200x4xf32) <- (-1x25200x4xf16)
        cast_2 = paddle._C_ops.cast(divide__0, paddle.float32)

        # pd_op.cast: (-1x80x25200xf32) <- (-1x80x25200xf16)
        cast_3 = paddle._C_ops.cast(concat_17, paddle.float32)

        # pd_op.multiclass_nms3: (-1x6xf32, -1x1xi32, -1xi32) <- (-1x25200x4xf32, -1x80x25200xf32, None)
        multiclass_nms3_0, multiclass_nms3_1, multiclass_nms3_2 = (lambda x, f: f(x))(paddle._C_ops.multiclass_nms3(cast_2, cast_3, None, float('0.001'), 3000, 300, float('0.65'), True, float('1'), -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.cast: (-1x6xf16) <- (-1x6xf32)
        cast_4 = paddle._C_ops.cast(multiclass_nms3_0, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_57 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6xf16) <- (-1x6xf16, 1xf32)
        scale__15 = paddle._C_ops.scale_(cast_4, full_57, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_58 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1xi32) <- (-1xi32, 1xf32)
        scale_3 = paddle._C_ops.scale(multiclass_nms3_2, full_58, float('0'), True)

        # pd_op.cast: (-1x6xf32) <- (-1x6xf16)
        cast_5 = paddle._C_ops.cast(scale__15, paddle.float32)
        return cast_5, scale_3



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_286, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, feed_1, feed_0):
        return self.builtin_module_823_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_286, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, feed_1, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_823_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([32, 3, 6, 6], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([128, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([256, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([512, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_274
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_284
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([255, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_286
            paddle.uniform([255], dtype='float16', min=0, max=0.5),
            # parameter_287
            paddle.uniform([255, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([255], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([255, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_290
            paddle.uniform([255], dtype='float16', min=0, max=0.5),
            # parameter_291
            paddle.uniform([3, 3, 2], dtype='float16', min=0, max=0.5),
            # feed_1
            paddle.to_tensor([1.0, 1.0], dtype='float32').reshape([1, 2]),
            # feed_0
            paddle.uniform([1, 3, 640, 640], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 6, 6], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
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
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
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
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
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
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_274
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_284
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[255, 128, 1, 1], dtype='float16'),
            # parameter_286
            paddle.static.InputSpec(shape=[255], dtype='float16'),
            # parameter_287
            paddle.static.InputSpec(shape=[255, 256, 1, 1], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[255], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[255, 512, 1, 1], dtype='float16'),
            # parameter_290
            paddle.static.InputSpec(shape=[255], dtype='float16'),
            # parameter_291
            paddle.static.InputSpec(shape=[3, 3, 2], dtype='float16'),
            # feed_1
            paddle.static.InputSpec(shape=[None, 2], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 640, 640], dtype='float32'),
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