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
    return [432][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1186_0_0(self, constant_39, constant_38, constant_37, constant_36, constant_35, constant_34, constant_33, constant_32, constant_31, constant_30, constant_29, constant_28, constant_27, constant_26, constant_25, constant_24, constant_23, constant_22, constant_21, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_144, parameter_143, parameter_145, parameter_146, parameter_147, parameter_148, parameter_150, parameter_149, parameter_151, parameter_152, parameter_153, parameter_154, parameter_156, parameter_155, parameter_157, parameter_158, parameter_159, parameter_160, parameter_162, parameter_161, parameter_163, parameter_164, parameter_165, parameter_166, parameter_168, parameter_167, parameter_169, parameter_170, parameter_171, parameter_172, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_225, parameter_224, parameter_226, parameter_227, parameter_228, parameter_229, parameter_231, parameter_230, parameter_232, parameter_233, parameter_234, parameter_235, parameter_237, parameter_236, parameter_238, parameter_239, parameter_240, parameter_241, parameter_243, parameter_242, parameter_244, parameter_245, parameter_246, parameter_247, parameter_249, parameter_248, parameter_250, parameter_251, parameter_252, parameter_253, parameter_255, parameter_254, parameter_256, parameter_257, parameter_258, parameter_259, parameter_261, parameter_260, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_278, feed_0):

        # pd_op.cast: (-1x3x256x256xf16) <- (-1x3x256x256xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x16x128x128xf16) <- (-1x3x256x256xf16, 16x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x128x128xf16, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x128x128xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x16x128x128xf16) <- (-1x16x128x128xf16)
        silu_0 = paddle._C_ops.silu(batch_norm__0)

        # pd_op.conv2d: (-1x64x128x128xf16) <- (-1x16x128x128xf16, 64x16x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(silu_0, parameter_5, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x128x128xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x128x128xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x64x128x128xf16) <- (-1x64x128x128xf16)
        silu_1 = paddle._C_ops.silu(batch_norm__6)

        # pd_op.depthwise_conv2d: (-1x64x128x128xf16) <- (-1x64x128x128xf16, 64x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(silu_1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x64x128x128xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x128x128xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x64x128x128xf16) <- (-1x64x128x128xf16)
        silu_2 = paddle._C_ops.silu(batch_norm__12)

        # pd_op.conv2d: (-1x32x128x128xf16) <- (-1x64x128x128xf16, 32x64x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(silu_2, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x128x128xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x128x128xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x128x128xf16) <- (-1x32x128x128xf16, 128x32x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(batch_norm__18, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x128x128xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x128x128xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x128x128xf16) <- (-1x128x128x128xf16)
        silu_3 = paddle._C_ops.silu(batch_norm__24)

        # pd_op.depthwise_conv2d: (-1x128x64x64xf16) <- (-1x128x128x128xf16, 128x1x3x3xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(silu_3, parameter_25, [2, 2], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x128x64x64xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x64x64xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x64x64xf16) <- (-1x128x64x64xf16)
        silu_4 = paddle._C_ops.silu(batch_norm__30)

        # pd_op.conv2d: (-1x64x64x64xf16) <- (-1x128x64x64xf16, 64x128x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(silu_4, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x64x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x64x64xf16) <- (-1x64x64x64xf16, 256x64x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(batch_norm__36, parameter_35, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf16) <- (-1x256x64x64xf16)
        silu_5 = paddle._C_ops.silu(batch_norm__42)

        # pd_op.depthwise_conv2d: (-1x256x64x64xf16) <- (-1x256x64x64xf16, 256x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(silu_5, parameter_40, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf16) <- (-1x256x64x64xf16)
        silu_6 = paddle._C_ops.silu(batch_norm__48)

        # pd_op.conv2d: (-1x64x64x64xf16) <- (-1x256x64x64xf16, 64x256x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(silu_6, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x64x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x64x64xf16) <- (-1x64x64x64xf16, -1x64x64x64xf16)
        add__0 = paddle._C_ops.add_(batch_norm__36, batch_norm__54)

        # pd_op.conv2d: (-1x256x64x64xf16) <- (-1x64x64x64xf16, 256x64x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(add__0, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf16) <- (-1x256x64x64xf16)
        silu_7 = paddle._C_ops.silu(batch_norm__60)

        # pd_op.depthwise_conv2d: (-1x256x64x64xf16) <- (-1x256x64x64xf16, 256x1x3x3xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(silu_7, parameter_55, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf16) <- (-1x256x64x64xf16)
        silu_8 = paddle._C_ops.silu(batch_norm__66)

        # pd_op.conv2d: (-1x64x64x64xf16) <- (-1x256x64x64xf16, 64x256x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(silu_8, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x64x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x64x64xf16) <- (-1x64x64x64xf16, -1x64x64x64xf16)
        add__1 = paddle._C_ops.add_(add__0, batch_norm__72)

        # pd_op.conv2d: (-1x256x64x64xf16) <- (-1x64x64x64xf16, 256x64x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(add__1, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf16) <- (-1x256x64x64xf16)
        silu_9 = paddle._C_ops.silu(batch_norm__78)

        # pd_op.depthwise_conv2d: (-1x256x32x32xf16) <- (-1x256x64x64xf16, 256x1x3x3xf16)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(silu_9, parameter_70, [2, 2], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x32x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x32x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x32x32xf16) <- (-1x256x32x32xf16)
        silu_10 = paddle._C_ops.silu(batch_norm__84)

        # pd_op.conv2d: (-1x96x32x32xf16) <- (-1x256x32x32xf16, 96x256x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(silu_10, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x32x32xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x32x32xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x96x32x32xf16) <- (-1x96x32x32xf16, 96x96x3x3xf16)
        conv2d_11 = paddle._C_ops.conv2d(batch_norm__90, parameter_80, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x32x32xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x32x32xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x96x32x32xf16) <- (-1x96x32x32xf16)
        silu_11 = paddle._C_ops.silu(batch_norm__96)

        # pd_op.conv2d: (-1x144x32x32xf16) <- (-1x96x32x32xf16, 144x96x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(silu_11, parameter_85, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape_: (-1x2x16x2xf16, 0x-1x144x32x32xf16) <- (-1x144x32x32xf16, 4xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_12, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x2x2xf16) <- (-1x2x16x2xf16)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x144x256x4xf16, 0x-1x16x2x2xf16) <- (-1x16x2x2xf16, 4xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x256x144xf16) <- (-1x144x256x4xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__2, [0, 3, 2, 1])

        # pd_op.reshape_: (-1x256x144xf16, 0x-1x4x256x144xf16) <- (-1x4x256x144xf16, 3xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x256x144xf16, -256xf32, -256xf32) <- (-1x256x144xf16, 144xf32, 144xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__4, parameter_86, parameter_87, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x256x144xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(layer_norm_0, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_3, constant_4, [1], [0])

        # pd_op.matmul: (-1x256x432xf16) <- (-1x256x144xf16, 144x432xf16)
        matmul_0 = paddle.matmul(layer_norm_0, parameter_88, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x432xf16) <- (-1x256x432xf16, 432xf16)
        add__2 = paddle._C_ops.add_(matmul_0, parameter_89)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, constant_5, constant_6, constant_7, constant_8]

        # pd_op.reshape_: (-1x256x3x4x36xf16, 0x-1x256x432xf16) <- (-1x256x432xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__2, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x256x36xf16) <- (-1x256x3x4x36xf16)
        transpose_2 = paddle._C_ops.transpose(reshape__6, [0, 3, 2, 1, 4])

        # pd_op.slice: (-1x4x256x36xf16) <- (-1x4x3x256x36xf16, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(transpose_2, [2], constant_3, constant_4, [1], [2])

        # pd_op.slice: (-1x4x256x36xf16) <- (-1x4x3x256x36xf16, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_2, [2], constant_4, constant_9, [1], [2])

        # pd_op.slice: (-1x4x256x36xf16) <- (-1x4x3x256x36xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_2, [2], constant_9, constant_10, [1], [2])

        # pd_op.scale_: (-1x4x256x36xf16) <- (-1x4x256x36xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(slice_1, constant_11, float('0'), True)

        # pd_op.transpose: (-1x4x36x256xf16) <- (-1x4x256x36xf16)
        transpose_3 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x256x256xf16) <- (-1x4x256x36xf16, -1x4x36x256xf16)
        matmul_1 = paddle.matmul(scale__0, transpose_3, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x4x256x256xf16) <- (-1x4x256x256xf16)
        softmax__0 = paddle._C_ops.softmax_(matmul_1, -1)

        # pd_op.matmul: (-1x4x256x36xf16) <- (-1x4x256x256xf16, -1x4x256x36xf16)
        matmul_2 = paddle.matmul(softmax__0, slice_3, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x256x4x36xf16) <- (-1x4x256x36xf16)
        transpose_4 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_1 = [slice_0, constant_5, constant_12]

        # pd_op.reshape_: (-1x256x144xf16, 0x-1x256x4x36xf16) <- (-1x256x4x36xf16, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, combine_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x144xf16) <- (-1x256x144xf16, 144x144xf16)
        matmul_3 = paddle.matmul(reshape__8, parameter_90, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x144xf16) <- (-1x256x144xf16, 144xf16)
        add__3 = paddle._C_ops.add_(matmul_3, parameter_91)

        # pd_op.dropout: (-1x256x144xf16, None) <- (-1x256x144xf16, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__3, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x144xf16) <- (-1x256x144xf16, -1x256x144xf16)
        add__4 = paddle._C_ops.add_(reshape__4, dropout_0)

        # pd_op.layer_norm: (-1x256x144xf16, -256xf32, -256xf32) <- (-1x256x144xf16, 144xf32, 144xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__4, parameter_92, parameter_93, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x256x288xf16) <- (-1x256x144xf16, 144x288xf16)
        matmul_4 = paddle.matmul(layer_norm_3, parameter_94, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x288xf16) <- (-1x256x288xf16, 288xf16)
        add__5 = paddle._C_ops.add_(matmul_4, parameter_95)

        # pd_op.silu: (-1x256x288xf16) <- (-1x256x288xf16)
        silu_12 = paddle._C_ops.silu(add__5)

        # pd_op.dropout: (-1x256x288xf16, None) <- (-1x256x288xf16, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_12, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x144xf16) <- (-1x256x288xf16, 288x144xf16)
        matmul_5 = paddle.matmul(dropout_2, parameter_96, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x144xf16) <- (-1x256x144xf16, 144xf16)
        add__6 = paddle._C_ops.add_(matmul_5, parameter_97)

        # pd_op.dropout: (-1x256x144xf16, None) <- (-1x256x144xf16, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__6, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x144xf16) <- (-1x256x144xf16, -1x256x144xf16)
        add__7 = paddle._C_ops.add_(dropout_4, add__4)

        # pd_op.layer_norm: (-1x256x144xf16, -256xf32, -256xf32) <- (-1x256x144xf16, 144xf32, 144xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__7, parameter_98, parameter_99, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x256x144xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(layer_norm_6, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_1, [0], constant_3, constant_4, [1], [0])

        # pd_op.matmul: (-1x256x432xf16) <- (-1x256x144xf16, 144x432xf16)
        matmul_6 = paddle.matmul(layer_norm_6, parameter_100, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x432xf16) <- (-1x256x432xf16, 432xf16)
        add__8 = paddle._C_ops.add_(matmul_6, parameter_101)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_4, constant_5, constant_6, constant_7, constant_8]

        # pd_op.reshape_: (-1x256x3x4x36xf16, 0x-1x256x432xf16) <- (-1x256x432xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__8, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x256x36xf16) <- (-1x256x3x4x36xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__10, [0, 3, 2, 1, 4])

        # pd_op.slice: (-1x4x256x36xf16) <- (-1x4x3x256x36xf16, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(transpose_5, [2], constant_3, constant_4, [1], [2])

        # pd_op.slice: (-1x4x256x36xf16) <- (-1x4x3x256x36xf16, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_5, [2], constant_4, constant_9, [1], [2])

        # pd_op.slice: (-1x4x256x36xf16) <- (-1x4x3x256x36xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_5, [2], constant_9, constant_10, [1], [2])

        # pd_op.scale_: (-1x4x256x36xf16) <- (-1x4x256x36xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(slice_5, constant_11, float('0'), True)

        # pd_op.transpose: (-1x4x36x256xf16) <- (-1x4x256x36xf16)
        transpose_6 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x256x256xf16) <- (-1x4x256x36xf16, -1x4x36x256xf16)
        matmul_7 = paddle.matmul(scale__1, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x4x256x256xf16) <- (-1x4x256x256xf16)
        softmax__1 = paddle._C_ops.softmax_(matmul_7, -1)

        # pd_op.matmul: (-1x4x256x36xf16) <- (-1x4x256x256xf16, -1x4x256x36xf16)
        matmul_8 = paddle.matmul(softmax__1, slice_7, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x256x4x36xf16) <- (-1x4x256x36xf16)
        transpose_7 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_3 = [slice_4, constant_5, constant_12]

        # pd_op.reshape_: (-1x256x144xf16, 0x-1x256x4x36xf16) <- (-1x256x4x36xf16, [1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x144xf16) <- (-1x256x144xf16, 144x144xf16)
        matmul_9 = paddle.matmul(reshape__12, parameter_102, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x144xf16) <- (-1x256x144xf16, 144xf16)
        add__9 = paddle._C_ops.add_(matmul_9, parameter_103)

        # pd_op.dropout: (-1x256x144xf16, None) <- (-1x256x144xf16, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__9, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x144xf16) <- (-1x256x144xf16, -1x256x144xf16)
        add__10 = paddle._C_ops.add_(add__7, dropout_6)

        # pd_op.layer_norm: (-1x256x144xf16, -256xf32, -256xf32) <- (-1x256x144xf16, 144xf32, 144xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__10, parameter_104, parameter_105, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x256x288xf16) <- (-1x256x144xf16, 144x288xf16)
        matmul_10 = paddle.matmul(layer_norm_9, parameter_106, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x288xf16) <- (-1x256x288xf16, 288xf16)
        add__11 = paddle._C_ops.add_(matmul_10, parameter_107)

        # pd_op.silu: (-1x256x288xf16) <- (-1x256x288xf16)
        silu_13 = paddle._C_ops.silu(add__11)

        # pd_op.dropout: (-1x256x288xf16, None) <- (-1x256x288xf16, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_13, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x144xf16) <- (-1x256x288xf16, 288x144xf16)
        matmul_11 = paddle.matmul(dropout_8, parameter_108, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256x144xf16) <- (-1x256x144xf16, 144xf16)
        add__12 = paddle._C_ops.add_(matmul_11, parameter_109)

        # pd_op.dropout: (-1x256x144xf16, None) <- (-1x256x144xf16, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__12, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x144xf16) <- (-1x256x144xf16, -1x256x144xf16)
        add__13 = paddle._C_ops.add_(dropout_10, add__10)

        # pd_op.layer_norm: (-1x256x144xf16, -256xf32, -256xf32) <- (-1x256x144xf16, 144xf32, 144xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__13, parameter_110, parameter_111, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.reshape_: (-1x4x256x144xf16, 0x-1x256x144xf16) <- (-1x256x144xf16, 4xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_12, constant_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x144x256x4xf16) <- (-1x4x256x144xf16)
        transpose_8 = paddle._C_ops.transpose(reshape__14, [0, 3, 2, 1])

        # pd_op.reshape_: (-1x16x2x2xf16, 0x-1x144x256x4xf16) <- (-1x144x256x4xf16, 4xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_8, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x16x2xf16) <- (-1x16x2x2xf16)
        transpose_9 = paddle._C_ops.transpose(reshape__16, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x144x32x32xf16, 0x-1x2x16x2xf16) <- (-1x2x16x2xf16, 4xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, constant_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x96x32x32xf16) <- (-1x144x32x32xf16, 96x144x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(reshape__18, parameter_112, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x32x32xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x32x32xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_113, parameter_114, parameter_115, parameter_116, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x96x32x32xf16) <- (-1x96x32x32xf16)
        silu_14 = paddle._C_ops.silu(batch_norm__102)

        # builtin.combine: ([-1x96x32x32xf16, -1x96x32x32xf16]) <- (-1x96x32x32xf16, -1x96x32x32xf16)
        combine_4 = [batch_norm__90, silu_14]

        # pd_op.concat: (-1x192x32x32xf16) <- ([-1x96x32x32xf16, -1x96x32x32xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_4, constant_17)

        # pd_op.conv2d: (-1x96x32x32xf16) <- (-1x192x32x32xf16, 96x192x3x3xf16)
        conv2d_14 = paddle._C_ops.conv2d(concat_0, parameter_117, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x32x32xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x32x32xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_118, parameter_119, parameter_120, parameter_121, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x96x32x32xf16) <- (-1x96x32x32xf16)
        silu_15 = paddle._C_ops.silu(batch_norm__108)

        # pd_op.conv2d: (-1x384x32x32xf16) <- (-1x96x32x32xf16, 384x96x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(silu_15, parameter_122, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x32x32xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x32x32xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_123, parameter_124, parameter_125, parameter_126, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x384x32x32xf16) <- (-1x384x32x32xf16)
        silu_16 = paddle._C_ops.silu(batch_norm__114)

        # pd_op.depthwise_conv2d: (-1x384x16x16xf16) <- (-1x384x32x32xf16, 384x1x3x3xf16)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(silu_16, parameter_127, [2, 2], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x384x16x16xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x16x16xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_128, parameter_129, parameter_130, parameter_131, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x384x16x16xf16) <- (-1x384x16x16xf16)
        silu_17 = paddle._C_ops.silu(batch_norm__120)

        # pd_op.conv2d: (-1x128x16x16xf16) <- (-1x384x16x16xf16, 128x384x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(silu_17, parameter_132, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x16xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x16xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_133, parameter_134, parameter_135, parameter_136, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x16x16xf16) <- (-1x128x16x16xf16, 128x128x3x3xf16)
        conv2d_17 = paddle._C_ops.conv2d(batch_norm__126, parameter_137, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x16xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x16xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_138, parameter_139, parameter_140, parameter_141, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x16x16xf16) <- (-1x128x16x16xf16)
        silu_18 = paddle._C_ops.silu(batch_norm__132)

        # pd_op.conv2d: (-1x192x16x16xf16) <- (-1x128x16x16xf16, 192x128x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(silu_18, parameter_142, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape_: (-1x2x8x2xf16, 0x-1x192x16x16xf16) <- (-1x192x16x16xf16, 4xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_18, constant_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x2x2xf16) <- (-1x2x8x2xf16)
        transpose_10 = paddle._C_ops.transpose(reshape__20, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x192x64x4xf16, 0x-1x8x2x2xf16) <- (-1x8x2x2xf16, 4xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x64x192xf16) <- (-1x192x64x4xf16)
        transpose_11 = paddle._C_ops.transpose(reshape__22, [0, 3, 2, 1])

        # pd_op.reshape_: (-1x64x192xf16, 0x-1x4x64x192xf16) <- (-1x4x64x192xf16, 3xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_11, constant_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x64x192xf16, -64xf32, -64xf32) <- (-1x64x192xf16, 192xf32, 192xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__24, parameter_143, parameter_144, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x64x192xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(layer_norm_15, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_2, [0], constant_3, constant_4, [1], [0])

        # pd_op.matmul: (-1x64x576xf16) <- (-1x64x192xf16, 192x576xf16)
        matmul_12 = paddle.matmul(layer_norm_15, parameter_145, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x576xf16) <- (-1x64x576xf16, 576xf16)
        add__14 = paddle._C_ops.add_(matmul_12, parameter_146)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_8, constant_21, constant_6, constant_7, constant_22]

        # pd_op.reshape_: (-1x64x3x4x48xf16, 0x-1x64x576xf16) <- (-1x64x576xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__14, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x64x48xf16) <- (-1x64x3x4x48xf16)
        transpose_12 = paddle._C_ops.transpose(reshape__26, [0, 3, 2, 1, 4])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_12, [2], constant_3, constant_4, [1], [2])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_12, [2], constant_4, constant_9, [1], [2])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_12, [2], constant_9, constant_10, [1], [2])

        # pd_op.scale_: (-1x4x64x48xf16) <- (-1x4x64x48xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(slice_9, constant_23, float('0'), True)

        # pd_op.transpose: (-1x4x48x64xf16) <- (-1x4x64x48xf16)
        transpose_13 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x64x64xf16) <- (-1x4x64x48xf16, -1x4x48x64xf16)
        matmul_13 = paddle.matmul(scale__2, transpose_13, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x4x64x64xf16) <- (-1x4x64x64xf16)
        softmax__2 = paddle._C_ops.softmax_(matmul_13, -1)

        # pd_op.matmul: (-1x4x64x48xf16) <- (-1x4x64x64xf16, -1x4x64x48xf16)
        matmul_14 = paddle.matmul(softmax__2, slice_11, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x64x4x48xf16) <- (-1x4x64x48xf16)
        transpose_14 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_6 = [slice_8, constant_21, constant_24]

        # pd_op.reshape_: (-1x64x192xf16, 0x-1x64x4x48xf16) <- (-1x64x4x48xf16, [1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf16) <- (-1x64x192xf16, 192x192xf16)
        matmul_15 = paddle.matmul(reshape__28, parameter_147, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, 192xf16)
        add__15 = paddle._C_ops.add_(matmul_15, parameter_148)

        # pd_op.dropout: (-1x64x192xf16, None) <- (-1x64x192xf16, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__15, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, -1x64x192xf16)
        add__16 = paddle._C_ops.add_(reshape__24, dropout_12)

        # pd_op.layer_norm: (-1x64x192xf16, -64xf32, -64xf32) <- (-1x64x192xf16, 192xf32, 192xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__16, parameter_149, parameter_150, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x64x384xf16) <- (-1x64x192xf16, 192x384xf16)
        matmul_16 = paddle.matmul(layer_norm_18, parameter_151, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x384xf16) <- (-1x64x384xf16, 384xf16)
        add__17 = paddle._C_ops.add_(matmul_16, parameter_152)

        # pd_op.silu: (-1x64x384xf16) <- (-1x64x384xf16)
        silu_19 = paddle._C_ops.silu(add__17)

        # pd_op.dropout: (-1x64x384xf16, None) <- (-1x64x384xf16, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_19, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf16) <- (-1x64x384xf16, 384x192xf16)
        matmul_17 = paddle.matmul(dropout_14, parameter_153, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, 192xf16)
        add__18 = paddle._C_ops.add_(matmul_17, parameter_154)

        # pd_op.dropout: (-1x64x192xf16, None) <- (-1x64x192xf16, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__18, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, -1x64x192xf16)
        add__19 = paddle._C_ops.add_(dropout_16, add__16)

        # pd_op.layer_norm: (-1x64x192xf16, -64xf32, -64xf32) <- (-1x64x192xf16, 192xf32, 192xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__19, parameter_155, parameter_156, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x64x192xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(layer_norm_21, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_3, [0], constant_3, constant_4, [1], [0])

        # pd_op.matmul: (-1x64x576xf16) <- (-1x64x192xf16, 192x576xf16)
        matmul_18 = paddle.matmul(layer_norm_21, parameter_157, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x576xf16) <- (-1x64x576xf16, 576xf16)
        add__20 = paddle._C_ops.add_(matmul_18, parameter_158)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_12, constant_21, constant_6, constant_7, constant_22]

        # pd_op.reshape_: (-1x64x3x4x48xf16, 0x-1x64x576xf16) <- (-1x64x576xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__20, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x64x48xf16) <- (-1x64x3x4x48xf16)
        transpose_15 = paddle._C_ops.transpose(reshape__30, [0, 3, 2, 1, 4])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(transpose_15, [2], constant_3, constant_4, [1], [2])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(transpose_15, [2], constant_4, constant_9, [1], [2])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_15, [2], constant_9, constant_10, [1], [2])

        # pd_op.scale_: (-1x4x64x48xf16) <- (-1x4x64x48xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(slice_13, constant_23, float('0'), True)

        # pd_op.transpose: (-1x4x48x64xf16) <- (-1x4x64x48xf16)
        transpose_16 = paddle._C_ops.transpose(slice_14, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x64x64xf16) <- (-1x4x64x48xf16, -1x4x48x64xf16)
        matmul_19 = paddle.matmul(scale__3, transpose_16, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x4x64x64xf16) <- (-1x4x64x64xf16)
        softmax__3 = paddle._C_ops.softmax_(matmul_19, -1)

        # pd_op.matmul: (-1x4x64x48xf16) <- (-1x4x64x64xf16, -1x4x64x48xf16)
        matmul_20 = paddle.matmul(softmax__3, slice_15, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x64x4x48xf16) <- (-1x4x64x48xf16)
        transpose_17 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_8 = [slice_12, constant_21, constant_24]

        # pd_op.reshape_: (-1x64x192xf16, 0x-1x64x4x48xf16) <- (-1x64x4x48xf16, [1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_17, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf16) <- (-1x64x192xf16, 192x192xf16)
        matmul_21 = paddle.matmul(reshape__32, parameter_159, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, 192xf16)
        add__21 = paddle._C_ops.add_(matmul_21, parameter_160)

        # pd_op.dropout: (-1x64x192xf16, None) <- (-1x64x192xf16, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__21, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, -1x64x192xf16)
        add__22 = paddle._C_ops.add_(add__19, dropout_18)

        # pd_op.layer_norm: (-1x64x192xf16, -64xf32, -64xf32) <- (-1x64x192xf16, 192xf32, 192xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__22, parameter_161, parameter_162, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x64x384xf16) <- (-1x64x192xf16, 192x384xf16)
        matmul_22 = paddle.matmul(layer_norm_24, parameter_163, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x384xf16) <- (-1x64x384xf16, 384xf16)
        add__23 = paddle._C_ops.add_(matmul_22, parameter_164)

        # pd_op.silu: (-1x64x384xf16) <- (-1x64x384xf16)
        silu_20 = paddle._C_ops.silu(add__23)

        # pd_op.dropout: (-1x64x384xf16, None) <- (-1x64x384xf16, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_20, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf16) <- (-1x64x384xf16, 384x192xf16)
        matmul_23 = paddle.matmul(dropout_20, parameter_165, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, 192xf16)
        add__24 = paddle._C_ops.add_(matmul_23, parameter_166)

        # pd_op.dropout: (-1x64x192xf16, None) <- (-1x64x192xf16, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__24, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, -1x64x192xf16)
        add__25 = paddle._C_ops.add_(dropout_22, add__22)

        # pd_op.layer_norm: (-1x64x192xf16, -64xf32, -64xf32) <- (-1x64x192xf16, 192xf32, 192xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__25, parameter_167, parameter_168, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x64x192xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(layer_norm_27, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_4, [0], constant_3, constant_4, [1], [0])

        # pd_op.matmul: (-1x64x576xf16) <- (-1x64x192xf16, 192x576xf16)
        matmul_24 = paddle.matmul(layer_norm_27, parameter_169, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x576xf16) <- (-1x64x576xf16, 576xf16)
        add__26 = paddle._C_ops.add_(matmul_24, parameter_170)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_16, constant_21, constant_6, constant_7, constant_22]

        # pd_op.reshape_: (-1x64x3x4x48xf16, 0x-1x64x576xf16) <- (-1x64x576xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__26, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x64x48xf16) <- (-1x64x3x4x48xf16)
        transpose_18 = paddle._C_ops.transpose(reshape__34, [0, 3, 2, 1, 4])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_18, [2], constant_3, constant_4, [1], [2])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_18, [2], constant_4, constant_9, [1], [2])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_18, [2], constant_9, constant_10, [1], [2])

        # pd_op.scale_: (-1x4x64x48xf16) <- (-1x4x64x48xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(slice_17, constant_23, float('0'), True)

        # pd_op.transpose: (-1x4x48x64xf16) <- (-1x4x64x48xf16)
        transpose_19 = paddle._C_ops.transpose(slice_18, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x64x64xf16) <- (-1x4x64x48xf16, -1x4x48x64xf16)
        matmul_25 = paddle.matmul(scale__4, transpose_19, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x4x64x64xf16) <- (-1x4x64x64xf16)
        softmax__4 = paddle._C_ops.softmax_(matmul_25, -1)

        # pd_op.matmul: (-1x4x64x48xf16) <- (-1x4x64x64xf16, -1x4x64x48xf16)
        matmul_26 = paddle.matmul(softmax__4, slice_19, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x64x4x48xf16) <- (-1x4x64x48xf16)
        transpose_20 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_16, constant_21, constant_24]

        # pd_op.reshape_: (-1x64x192xf16, 0x-1x64x4x48xf16) <- (-1x64x4x48xf16, [1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf16) <- (-1x64x192xf16, 192x192xf16)
        matmul_27 = paddle.matmul(reshape__36, parameter_171, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, 192xf16)
        add__27 = paddle._C_ops.add_(matmul_27, parameter_172)

        # pd_op.dropout: (-1x64x192xf16, None) <- (-1x64x192xf16, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__27, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, -1x64x192xf16)
        add__28 = paddle._C_ops.add_(add__25, dropout_24)

        # pd_op.layer_norm: (-1x64x192xf16, -64xf32, -64xf32) <- (-1x64x192xf16, 192xf32, 192xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__28, parameter_173, parameter_174, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x64x384xf16) <- (-1x64x192xf16, 192x384xf16)
        matmul_28 = paddle.matmul(layer_norm_30, parameter_175, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x384xf16) <- (-1x64x384xf16, 384xf16)
        add__29 = paddle._C_ops.add_(matmul_28, parameter_176)

        # pd_op.silu: (-1x64x384xf16) <- (-1x64x384xf16)
        silu_21 = paddle._C_ops.silu(add__29)

        # pd_op.dropout: (-1x64x384xf16, None) <- (-1x64x384xf16, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_21, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf16) <- (-1x64x384xf16, 384x192xf16)
        matmul_29 = paddle.matmul(dropout_26, parameter_177, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, 192xf16)
        add__30 = paddle._C_ops.add_(matmul_29, parameter_178)

        # pd_op.dropout: (-1x64x192xf16, None) <- (-1x64x192xf16, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__30, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, -1x64x192xf16)
        add__31 = paddle._C_ops.add_(dropout_28, add__28)

        # pd_op.layer_norm: (-1x64x192xf16, -64xf32, -64xf32) <- (-1x64x192xf16, 192xf32, 192xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__31, parameter_179, parameter_180, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x64x192xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(layer_norm_33, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_5, [0], constant_3, constant_4, [1], [0])

        # pd_op.matmul: (-1x64x576xf16) <- (-1x64x192xf16, 192x576xf16)
        matmul_30 = paddle.matmul(layer_norm_33, parameter_181, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x576xf16) <- (-1x64x576xf16, 576xf16)
        add__32 = paddle._C_ops.add_(matmul_30, parameter_182)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_20, constant_21, constant_6, constant_7, constant_22]

        # pd_op.reshape_: (-1x64x3x4x48xf16, 0x-1x64x576xf16) <- (-1x64x576xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__32, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x64x48xf16) <- (-1x64x3x4x48xf16)
        transpose_21 = paddle._C_ops.transpose(reshape__38, [0, 3, 2, 1, 4])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(transpose_21, [2], constant_3, constant_4, [1], [2])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_21, [2], constant_4, constant_9, [1], [2])

        # pd_op.slice: (-1x4x64x48xf16) <- (-1x4x3x64x48xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_21, [2], constant_9, constant_10, [1], [2])

        # pd_op.scale_: (-1x4x64x48xf16) <- (-1x4x64x48xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(slice_21, constant_23, float('0'), True)

        # pd_op.transpose: (-1x4x48x64xf16) <- (-1x4x64x48xf16)
        transpose_22 = paddle._C_ops.transpose(slice_22, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x64x64xf16) <- (-1x4x64x48xf16, -1x4x48x64xf16)
        matmul_31 = paddle.matmul(scale__5, transpose_22, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x4x64x64xf16) <- (-1x4x64x64xf16)
        softmax__5 = paddle._C_ops.softmax_(matmul_31, -1)

        # pd_op.matmul: (-1x4x64x48xf16) <- (-1x4x64x64xf16, -1x4x64x48xf16)
        matmul_32 = paddle.matmul(softmax__5, slice_23, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x64x4x48xf16) <- (-1x4x64x48xf16)
        transpose_23 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_12 = [slice_20, constant_21, constant_24]

        # pd_op.reshape_: (-1x64x192xf16, 0x-1x64x4x48xf16) <- (-1x64x4x48xf16, [1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_23, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf16) <- (-1x64x192xf16, 192x192xf16)
        matmul_33 = paddle.matmul(reshape__40, parameter_183, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, 192xf16)
        add__33 = paddle._C_ops.add_(matmul_33, parameter_184)

        # pd_op.dropout: (-1x64x192xf16, None) <- (-1x64x192xf16, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__33, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, -1x64x192xf16)
        add__34 = paddle._C_ops.add_(add__31, dropout_30)

        # pd_op.layer_norm: (-1x64x192xf16, -64xf32, -64xf32) <- (-1x64x192xf16, 192xf32, 192xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__34, parameter_185, parameter_186, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x64x384xf16) <- (-1x64x192xf16, 192x384xf16)
        matmul_34 = paddle.matmul(layer_norm_36, parameter_187, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x384xf16) <- (-1x64x384xf16, 384xf16)
        add__35 = paddle._C_ops.add_(matmul_34, parameter_188)

        # pd_op.silu: (-1x64x384xf16) <- (-1x64x384xf16)
        silu_22 = paddle._C_ops.silu(add__35)

        # pd_op.dropout: (-1x64x384xf16, None) <- (-1x64x384xf16, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_22, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf16) <- (-1x64x384xf16, 384x192xf16)
        matmul_35 = paddle.matmul(dropout_32, parameter_189, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, 192xf16)
        add__36 = paddle._C_ops.add_(matmul_35, parameter_190)

        # pd_op.dropout: (-1x64x192xf16, None) <- (-1x64x192xf16, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__36, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf16) <- (-1x64x192xf16, -1x64x192xf16)
        add__37 = paddle._C_ops.add_(dropout_34, add__34)

        # pd_op.layer_norm: (-1x64x192xf16, -64xf32, -64xf32) <- (-1x64x192xf16, 192xf32, 192xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__37, parameter_191, parameter_192, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.reshape_: (-1x4x64x192xf16, 0x-1x64x192xf16) <- (-1x64x192xf16, 4xi64)
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_39, constant_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x192x64x4xf16) <- (-1x4x64x192xf16)
        transpose_24 = paddle._C_ops.transpose(reshape__42, [0, 3, 2, 1])

        # pd_op.reshape_: (-1x8x2x2xf16, 0x-1x192x64x4xf16) <- (-1x192x64x4xf16, 4xi64)
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, constant_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x8x2xf16) <- (-1x8x2x2xf16)
        transpose_25 = paddle._C_ops.transpose(reshape__44, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x192x16x16xf16, 0x-1x2x8x2xf16) <- (-1x2x8x2xf16, 4xi64)
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_25, constant_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x16x16xf16) <- (-1x192x16x16xf16, 128x192x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(reshape__46, parameter_193, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x16xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x16xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_194, parameter_195, parameter_196, parameter_197, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x16x16xf16) <- (-1x128x16x16xf16)
        silu_23 = paddle._C_ops.silu(batch_norm__138)

        # builtin.combine: ([-1x128x16x16xf16, -1x128x16x16xf16]) <- (-1x128x16x16xf16, -1x128x16x16xf16)
        combine_13 = [batch_norm__126, silu_23]

        # pd_op.concat: (-1x256x16x16xf16) <- ([-1x128x16x16xf16, -1x128x16x16xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_13, constant_17)

        # pd_op.conv2d: (-1x128x16x16xf16) <- (-1x256x16x16xf16, 128x256x3x3xf16)
        conv2d_20 = paddle._C_ops.conv2d(concat_1, parameter_198, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x16xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x16xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_199, parameter_200, parameter_201, parameter_202, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x16x16xf16) <- (-1x128x16x16xf16)
        silu_24 = paddle._C_ops.silu(batch_norm__144)

        # pd_op.conv2d: (-1x512x16x16xf16) <- (-1x128x16x16xf16, 512x128x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(silu_24, parameter_203, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x16x16xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_204, parameter_205, parameter_206, parameter_207, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x512x16x16xf16) <- (-1x512x16x16xf16)
        silu_25 = paddle._C_ops.silu(batch_norm__150)

        # pd_op.depthwise_conv2d: (-1x512x8x8xf16) <- (-1x512x16x16xf16, 512x1x3x3xf16)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(silu_25, parameter_208, [2, 2], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x8xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x8xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_209, parameter_210, parameter_211, parameter_212, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x512x8x8xf16) <- (-1x512x8x8xf16)
        silu_26 = paddle._C_ops.silu(batch_norm__156)

        # pd_op.conv2d: (-1x160x8x8xf16) <- (-1x512x8x8xf16, 160x512x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(silu_26, parameter_213, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x8xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x8xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_214, parameter_215, parameter_216, parameter_217, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x8xf16) <- (-1x160x8x8xf16, 160x160x3x3xf16)
        conv2d_23 = paddle._C_ops.conv2d(batch_norm__162, parameter_218, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x8xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x8xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_219, parameter_220, parameter_221, parameter_222, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x160x8x8xf16) <- (-1x160x8x8xf16)
        silu_27 = paddle._C_ops.silu(batch_norm__168)

        # pd_op.conv2d: (-1x240x8x8xf16) <- (-1x160x8x8xf16, 240x160x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(silu_27, parameter_223, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape_: (-1x2x4x2xf16, 0x-1x240x8x8xf16) <- (-1x240x8x8xf16, 4xi64)
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_24, constant_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x2x2xf16) <- (-1x2x4x2xf16)
        transpose_26 = paddle._C_ops.transpose(reshape__48, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x240x16x4xf16, 0x-1x4x2x2xf16) <- (-1x4x2x2xf16, 4xi64)
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_26, constant_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x240xf16) <- (-1x240x16x4xf16)
        transpose_27 = paddle._C_ops.transpose(reshape__50, [0, 3, 2, 1])

        # pd_op.reshape_: (-1x16x240xf16, 0x-1x4x16x240xf16) <- (-1x4x16x240xf16, 3xi64)
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_27, constant_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x16x240xf16, -16xf32, -16xf32) <- (-1x16x240xf16, 240xf32, 240xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__52, parameter_224, parameter_225, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x240xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(layer_norm_42, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_6, [0], constant_3, constant_4, [1], [0])

        # pd_op.matmul: (-1x16x720xf16) <- (-1x16x240xf16, 240x720xf16)
        matmul_36 = paddle.matmul(layer_norm_42, parameter_226, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x720xf16) <- (-1x16x720xf16, 720xf16)
        add__38 = paddle._C_ops.add_(matmul_36, parameter_227)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_24, constant_31, constant_6, constant_7, constant_32]

        # pd_op.reshape_: (-1x16x3x4x60xf16, 0x-1x16x720xf16) <- (-1x16x720xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__38, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x16x60xf16) <- (-1x16x3x4x60xf16)
        transpose_28 = paddle._C_ops.transpose(reshape__54, [0, 3, 2, 1, 4])

        # pd_op.slice: (-1x4x16x60xf16) <- (-1x4x3x16x60xf16, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_28, [2], constant_3, constant_4, [1], [2])

        # pd_op.slice: (-1x4x16x60xf16) <- (-1x4x3x16x60xf16, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_28, [2], constant_4, constant_9, [1], [2])

        # pd_op.slice: (-1x4x16x60xf16) <- (-1x4x3x16x60xf16, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_28, [2], constant_9, constant_10, [1], [2])

        # pd_op.scale_: (-1x4x16x60xf16) <- (-1x4x16x60xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(slice_25, constant_33, float('0'), True)

        # pd_op.transpose: (-1x4x60x16xf16) <- (-1x4x16x60xf16)
        transpose_29 = paddle._C_ops.transpose(slice_26, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x60xf16, -1x4x60x16xf16)
        matmul_37 = paddle.matmul(scale__6, transpose_29, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__6 = paddle._C_ops.softmax_(matmul_37, -1)

        # pd_op.matmul: (-1x4x16x60xf16) <- (-1x4x16x16xf16, -1x4x16x60xf16)
        matmul_38 = paddle.matmul(softmax__6, slice_27, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x60xf16) <- (-1x4x16x60xf16)
        transpose_30 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_15 = [slice_24, constant_31, constant_34]

        # pd_op.reshape_: (-1x16x240xf16, 0x-1x16x4x60xf16) <- (-1x16x4x60xf16, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_30, combine_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf16) <- (-1x16x240xf16, 240x240xf16)
        matmul_39 = paddle.matmul(reshape__56, parameter_228, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, 240xf16)
        add__39 = paddle._C_ops.add_(matmul_39, parameter_229)

        # pd_op.dropout: (-1x16x240xf16, None) <- (-1x16x240xf16, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__39, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, -1x16x240xf16)
        add__40 = paddle._C_ops.add_(reshape__52, dropout_36)

        # pd_op.layer_norm: (-1x16x240xf16, -16xf32, -16xf32) <- (-1x16x240xf16, 240xf32, 240xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__40, parameter_230, parameter_231, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x480xf16) <- (-1x16x240xf16, 240x480xf16)
        matmul_40 = paddle.matmul(layer_norm_45, parameter_232, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x480xf16) <- (-1x16x480xf16, 480xf16)
        add__41 = paddle._C_ops.add_(matmul_40, parameter_233)

        # pd_op.silu: (-1x16x480xf16) <- (-1x16x480xf16)
        silu_28 = paddle._C_ops.silu(add__41)

        # pd_op.dropout: (-1x16x480xf16, None) <- (-1x16x480xf16, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_28, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf16) <- (-1x16x480xf16, 480x240xf16)
        matmul_41 = paddle.matmul(dropout_38, parameter_234, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, 240xf16)
        add__42 = paddle._C_ops.add_(matmul_41, parameter_235)

        # pd_op.dropout: (-1x16x240xf16, None) <- (-1x16x240xf16, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__42, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, -1x16x240xf16)
        add__43 = paddle._C_ops.add_(dropout_40, add__40)

        # pd_op.layer_norm: (-1x16x240xf16, -16xf32, -16xf32) <- (-1x16x240xf16, 240xf32, 240xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__43, parameter_236, parameter_237, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x240xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(layer_norm_48, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_7, [0], constant_3, constant_4, [1], [0])

        # pd_op.matmul: (-1x16x720xf16) <- (-1x16x240xf16, 240x720xf16)
        matmul_42 = paddle.matmul(layer_norm_48, parameter_238, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x720xf16) <- (-1x16x720xf16, 720xf16)
        add__44 = paddle._C_ops.add_(matmul_42, parameter_239)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_28, constant_31, constant_6, constant_7, constant_32]

        # pd_op.reshape_: (-1x16x3x4x60xf16, 0x-1x16x720xf16) <- (-1x16x720xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__44, combine_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x16x60xf16) <- (-1x16x3x4x60xf16)
        transpose_31 = paddle._C_ops.transpose(reshape__58, [0, 3, 2, 1, 4])

        # pd_op.slice: (-1x4x16x60xf16) <- (-1x4x3x16x60xf16, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_31, [2], constant_3, constant_4, [1], [2])

        # pd_op.slice: (-1x4x16x60xf16) <- (-1x4x3x16x60xf16, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_31, [2], constant_4, constant_9, [1], [2])

        # pd_op.slice: (-1x4x16x60xf16) <- (-1x4x3x16x60xf16, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_31, [2], constant_9, constant_10, [1], [2])

        # pd_op.scale_: (-1x4x16x60xf16) <- (-1x4x16x60xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(slice_29, constant_33, float('0'), True)

        # pd_op.transpose: (-1x4x60x16xf16) <- (-1x4x16x60xf16)
        transpose_32 = paddle._C_ops.transpose(slice_30, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x60xf16, -1x4x60x16xf16)
        matmul_43 = paddle.matmul(scale__7, transpose_32, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__7 = paddle._C_ops.softmax_(matmul_43, -1)

        # pd_op.matmul: (-1x4x16x60xf16) <- (-1x4x16x16xf16, -1x4x16x60xf16)
        matmul_44 = paddle.matmul(softmax__7, slice_31, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x60xf16) <- (-1x4x16x60xf16)
        transpose_33 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_17 = [slice_28, constant_31, constant_34]

        # pd_op.reshape_: (-1x16x240xf16, 0x-1x16x4x60xf16) <- (-1x16x4x60xf16, [1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_33, combine_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf16) <- (-1x16x240xf16, 240x240xf16)
        matmul_45 = paddle.matmul(reshape__60, parameter_240, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, 240xf16)
        add__45 = paddle._C_ops.add_(matmul_45, parameter_241)

        # pd_op.dropout: (-1x16x240xf16, None) <- (-1x16x240xf16, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__45, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, -1x16x240xf16)
        add__46 = paddle._C_ops.add_(add__43, dropout_42)

        # pd_op.layer_norm: (-1x16x240xf16, -16xf32, -16xf32) <- (-1x16x240xf16, 240xf32, 240xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__46, parameter_242, parameter_243, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x480xf16) <- (-1x16x240xf16, 240x480xf16)
        matmul_46 = paddle.matmul(layer_norm_51, parameter_244, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x480xf16) <- (-1x16x480xf16, 480xf16)
        add__47 = paddle._C_ops.add_(matmul_46, parameter_245)

        # pd_op.silu: (-1x16x480xf16) <- (-1x16x480xf16)
        silu_29 = paddle._C_ops.silu(add__47)

        # pd_op.dropout: (-1x16x480xf16, None) <- (-1x16x480xf16, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_29, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf16) <- (-1x16x480xf16, 480x240xf16)
        matmul_47 = paddle.matmul(dropout_44, parameter_246, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, 240xf16)
        add__48 = paddle._C_ops.add_(matmul_47, parameter_247)

        # pd_op.dropout: (-1x16x240xf16, None) <- (-1x16x240xf16, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__48, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, -1x16x240xf16)
        add__49 = paddle._C_ops.add_(dropout_46, add__46)

        # pd_op.layer_norm: (-1x16x240xf16, -16xf32, -16xf32) <- (-1x16x240xf16, 240xf32, 240xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__49, parameter_248, parameter_249, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x240xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(layer_norm_54, 'float32'))

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_8, [0], constant_3, constant_4, [1], [0])

        # pd_op.matmul: (-1x16x720xf16) <- (-1x16x240xf16, 240x720xf16)
        matmul_48 = paddle.matmul(layer_norm_54, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x720xf16) <- (-1x16x720xf16, 720xf16)
        add__50 = paddle._C_ops.add_(matmul_48, parameter_251)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_32, constant_31, constant_6, constant_7, constant_32]

        # pd_op.reshape_: (-1x16x3x4x60xf16, 0x-1x16x720xf16) <- (-1x16x720xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__50, combine_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x16x60xf16) <- (-1x16x3x4x60xf16)
        transpose_34 = paddle._C_ops.transpose(reshape__62, [0, 3, 2, 1, 4])

        # pd_op.slice: (-1x4x16x60xf16) <- (-1x4x3x16x60xf16, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_34, [2], constant_3, constant_4, [1], [2])

        # pd_op.slice: (-1x4x16x60xf16) <- (-1x4x3x16x60xf16, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_34, [2], constant_4, constant_9, [1], [2])

        # pd_op.slice: (-1x4x16x60xf16) <- (-1x4x3x16x60xf16, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_34, [2], constant_9, constant_10, [1], [2])

        # pd_op.scale_: (-1x4x16x60xf16) <- (-1x4x16x60xf16, 1xf32)
        scale__8 = paddle._C_ops.scale_(slice_33, constant_33, float('0'), True)

        # pd_op.transpose: (-1x4x60x16xf16) <- (-1x4x16x60xf16)
        transpose_35 = paddle._C_ops.transpose(slice_34, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf16) <- (-1x4x16x60xf16, -1x4x60x16xf16)
        matmul_49 = paddle.matmul(scale__8, transpose_35, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x4x16x16xf16) <- (-1x4x16x16xf16)
        softmax__8 = paddle._C_ops.softmax_(matmul_49, -1)

        # pd_op.matmul: (-1x4x16x60xf16) <- (-1x4x16x16xf16, -1x4x16x60xf16)
        matmul_50 = paddle.matmul(softmax__8, slice_35, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x4x60xf16) <- (-1x4x16x60xf16)
        transpose_36 = paddle._C_ops.transpose(matmul_50, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_19 = [slice_32, constant_31, constant_34]

        # pd_op.reshape_: (-1x16x240xf16, 0x-1x16x4x60xf16) <- (-1x16x4x60xf16, [1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_36, combine_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf16) <- (-1x16x240xf16, 240x240xf16)
        matmul_51 = paddle.matmul(reshape__64, parameter_252, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, 240xf16)
        add__51 = paddle._C_ops.add_(matmul_51, parameter_253)

        # pd_op.dropout: (-1x16x240xf16, None) <- (-1x16x240xf16, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__51, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, -1x16x240xf16)
        add__52 = paddle._C_ops.add_(add__49, dropout_48)

        # pd_op.layer_norm: (-1x16x240xf16, -16xf32, -16xf32) <- (-1x16x240xf16, 240xf32, 240xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__52, parameter_254, parameter_255, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x480xf16) <- (-1x16x240xf16, 240x480xf16)
        matmul_52 = paddle.matmul(layer_norm_57, parameter_256, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x480xf16) <- (-1x16x480xf16, 480xf16)
        add__53 = paddle._C_ops.add_(matmul_52, parameter_257)

        # pd_op.silu: (-1x16x480xf16) <- (-1x16x480xf16)
        silu_30 = paddle._C_ops.silu(add__53)

        # pd_op.dropout: (-1x16x480xf16, None) <- (-1x16x480xf16, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_30, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf16) <- (-1x16x480xf16, 480x240xf16)
        matmul_53 = paddle.matmul(dropout_50, parameter_258, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, 240xf16)
        add__54 = paddle._C_ops.add_(matmul_53, parameter_259)

        # pd_op.dropout: (-1x16x240xf16, None) <- (-1x16x240xf16, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__54, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf16) <- (-1x16x240xf16, -1x16x240xf16)
        add__55 = paddle._C_ops.add_(dropout_52, add__52)

        # pd_op.layer_norm: (-1x16x240xf16, -16xf32, -16xf32) <- (-1x16x240xf16, 240xf32, 240xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__55, parameter_260, parameter_261, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.reshape_: (-1x4x16x240xf16, 0x-1x16x240xf16) <- (-1x16x240xf16, 4xi64)
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_60, constant_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x240x16x4xf16) <- (-1x4x16x240xf16)
        transpose_37 = paddle._C_ops.transpose(reshape__66, [0, 3, 2, 1])

        # pd_op.reshape_: (-1x4x2x2xf16, 0x-1x240x16x4xf16) <- (-1x240x16x4xf16, 4xi64)
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_37, constant_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x4x2xf16) <- (-1x4x2x2xf16)
        transpose_38 = paddle._C_ops.transpose(reshape__68, [0, 2, 1, 3])

        # pd_op.reshape_: (-1x240x8x8xf16, 0x-1x2x4x2xf16) <- (-1x2x4x2xf16, 4xi64)
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_38, constant_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x160x8x8xf16) <- (-1x240x8x8xf16, 160x240x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(reshape__70, parameter_262, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x8xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x8xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_263, parameter_264, parameter_265, parameter_266, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x160x8x8xf16) <- (-1x160x8x8xf16)
        silu_31 = paddle._C_ops.silu(batch_norm__174)

        # builtin.combine: ([-1x160x8x8xf16, -1x160x8x8xf16]) <- (-1x160x8x8xf16, -1x160x8x8xf16)
        combine_20 = [batch_norm__162, silu_31]

        # pd_op.concat: (-1x320x8x8xf16) <- ([-1x160x8x8xf16, -1x160x8x8xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_20, constant_17)

        # pd_op.conv2d: (-1x160x8x8xf16) <- (-1x320x8x8xf16, 160x320x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(concat_2, parameter_267, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x8xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x8xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_268, parameter_269, parameter_270, parameter_271, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x160x8x8xf16) <- (-1x160x8x8xf16)
        silu_32 = paddle._C_ops.silu(batch_norm__180)

        # pd_op.conv2d: (-1x640x8x8xf16) <- (-1x160x8x8xf16, 640x160x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(silu_32, parameter_272, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x8x8xf16, 640xf32, 640xf32, xf32, xf32, None) <- (-1x640x8x8xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_273, parameter_274, parameter_275, parameter_276, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x640x8x8xf16) <- (-1x640x8x8xf16)
        silu_33 = paddle._C_ops.silu(batch_norm__186)

        # pd_op.pool2d: (-1x640x1x1xf16) <- (-1x640x8x8xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(silu_33, constant_38, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.shape: (4xi32) <- (-1x640x1x1xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(pool2d_0, 'float32'))

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(shape_9, [0], constant_3, constant_4, [1], [0])

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_21 = [slice_36, constant_39]

        # pd_op.reshape_: (-1x640xf16, 0x-1x640x1x1xf16) <- (-1x640x1x1xf16, [1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(pool2d_0, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x640xf16, None) <- (-1x640xf16, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(paddle._C_ops.dropout(reshape__72, None, constant_13, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x640xf16, 640x1000xf16)
        matmul_54 = paddle.matmul(dropout_54, parameter_277, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__56 = paddle._C_ops.add_(matmul_54, parameter_278)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__9 = paddle._C_ops.softmax_(add__56, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__9, paddle.float32)
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

    def forward(self, constant_39, constant_38, constant_37, constant_36, constant_35, constant_34, constant_33, constant_32, constant_31, constant_30, constant_29, constant_28, constant_27, constant_26, constant_25, constant_24, constant_23, constant_22, constant_21, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_144, parameter_143, parameter_145, parameter_146, parameter_147, parameter_148, parameter_150, parameter_149, parameter_151, parameter_152, parameter_153, parameter_154, parameter_156, parameter_155, parameter_157, parameter_158, parameter_159, parameter_160, parameter_162, parameter_161, parameter_163, parameter_164, parameter_165, parameter_166, parameter_168, parameter_167, parameter_169, parameter_170, parameter_171, parameter_172, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_225, parameter_224, parameter_226, parameter_227, parameter_228, parameter_229, parameter_231, parameter_230, parameter_232, parameter_233, parameter_234, parameter_235, parameter_237, parameter_236, parameter_238, parameter_239, parameter_240, parameter_241, parameter_243, parameter_242, parameter_244, parameter_245, parameter_246, parameter_247, parameter_249, parameter_248, parameter_250, parameter_251, parameter_252, parameter_253, parameter_255, parameter_254, parameter_256, parameter_257, parameter_258, parameter_259, parameter_261, parameter_260, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_278, feed_0):
        return self.builtin_module_1186_0_0(constant_39, constant_38, constant_37, constant_36, constant_35, constant_34, constant_33, constant_32, constant_31, constant_30, constant_29, constant_28, constant_27, constant_26, constant_25, constant_24, constant_23, constant_22, constant_21, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_144, parameter_143, parameter_145, parameter_146, parameter_147, parameter_148, parameter_150, parameter_149, parameter_151, parameter_152, parameter_153, parameter_154, parameter_156, parameter_155, parameter_157, parameter_158, parameter_159, parameter_160, parameter_162, parameter_161, parameter_163, parameter_164, parameter_165, parameter_166, parameter_168, parameter_167, parameter_169, parameter_170, parameter_171, parameter_172, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_225, parameter_224, parameter_226, parameter_227, parameter_228, parameter_229, parameter_231, parameter_230, parameter_232, parameter_233, parameter_234, parameter_235, parameter_237, parameter_236, parameter_238, parameter_239, parameter_240, parameter_241, parameter_243, parameter_242, parameter_244, parameter_245, parameter_246, parameter_247, parameter_249, parameter_248, parameter_250, parameter_251, parameter_252, parameter_253, parameter_255, parameter_254, parameter_256, parameter_257, parameter_258, parameter_259, parameter_261, parameter_260, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_278, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1186_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_39
            paddle.to_tensor([640], dtype='int32').reshape([1]),
            # constant_38
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_37
            paddle.to_tensor([-1, 240, 8, 8], dtype='int64').reshape([4]),
            # constant_36
            paddle.to_tensor([-1, 4, 2, 2], dtype='int64').reshape([4]),
            # constant_35
            paddle.to_tensor([-1, 4, 16, 240], dtype='int64').reshape([4]),
            # constant_34
            paddle.to_tensor([240], dtype='int32').reshape([1]),
            # constant_33
            paddle.to_tensor([0.129099], dtype='float32').reshape([1]),
            # constant_32
            paddle.to_tensor([60], dtype='int32').reshape([1]),
            # constant_31
            paddle.to_tensor([16], dtype='int32').reshape([1]),
            # constant_30
            paddle.to_tensor([-1, 16, 240], dtype='int64').reshape([3]),
            # constant_29
            paddle.to_tensor([-1, 240, 16, 4], dtype='int64').reshape([4]),
            # constant_28
            paddle.to_tensor([-1, 2, 4, 2], dtype='int64').reshape([4]),
            # constant_27
            paddle.to_tensor([-1, 192, 16, 16], dtype='int64').reshape([4]),
            # constant_26
            paddle.to_tensor([-1, 8, 2, 2], dtype='int64').reshape([4]),
            # constant_25
            paddle.to_tensor([-1, 4, 64, 192], dtype='int64').reshape([4]),
            # constant_24
            paddle.to_tensor([192], dtype='int32').reshape([1]),
            # constant_23
            paddle.to_tensor([0.144338], dtype='float32').reshape([1]),
            # constant_22
            paddle.to_tensor([48], dtype='int32').reshape([1]),
            # constant_21
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_20
            paddle.to_tensor([-1, 64, 192], dtype='int64').reshape([3]),
            # constant_19
            paddle.to_tensor([-1, 192, 64, 4], dtype='int64').reshape([4]),
            # constant_18
            paddle.to_tensor([-1, 2, 8, 2], dtype='int64').reshape([4]),
            # constant_17
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_16
            paddle.to_tensor([-1, 144, 32, 32], dtype='int64').reshape([4]),
            # constant_15
            paddle.to_tensor([-1, 16, 2, 2], dtype='int64').reshape([4]),
            # constant_14
            paddle.to_tensor([-1, 4, 256, 144], dtype='int64').reshape([4]),
            # constant_13
            paddle.to_tensor([0.1], dtype='float32').reshape([1]),
            # constant_12
            paddle.to_tensor([144], dtype='int32').reshape([1]),
            # constant_11
            paddle.to_tensor([0.166667], dtype='float32').reshape([1]),
            # constant_10
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            # constant_9
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_8
            paddle.to_tensor([36], dtype='int32').reshape([1]),
            # constant_7
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            # constant_6
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            # constant_5
            paddle.to_tensor([256], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_3
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # constant_2
            paddle.to_tensor([-1, 256, 144], dtype='int64').reshape([3]),
            # constant_1
            paddle.to_tensor([-1, 144, 256, 4], dtype='int64').reshape([4]),
            # constant_0
            paddle.to_tensor([-1, 2, 16, 2], dtype='int64').reshape([4]),
            # parameter_0
            paddle.uniform([16, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([128, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([96, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([96, 96, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([144, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([144, 432], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([432], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([144, 144], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([144], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([144, 288], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([288], dtype='float16', min=0, max=0.5),
            # parameter_96
            paddle.uniform([288, 144], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([144], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([144, 432], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([432], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([144, 144], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([144], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([144, 288], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([288], dtype='float16', min=0, max=0.5),
            # parameter_108
            paddle.uniform([288, 144], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([144], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([96, 144, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([96, 192, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([384, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([384, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([192, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([192, 576], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([576], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_150
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([192, 384], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([384, 192], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([192, 576], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([576], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_160
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_162
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([192, 384], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([384, 192], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_168
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([192, 576], dtype='float16', min=0, max=0.5),
            # parameter_170
            paddle.uniform([576], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([192, 384], dtype='float16', min=0, max=0.5),
            # parameter_176
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([384, 192], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_180
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([192, 576], dtype='float16', min=0, max=0.5),
            # parameter_182
            paddle.uniform([576], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([192, 384], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([384, 192], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_192
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([128, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_197
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_202
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([160, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([160, 160, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_222
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([240, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([240, 720], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([720], dtype='float16', min=0, max=0.5),
            # parameter_228
            paddle.uniform([240, 240], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([240], dtype='float16', min=0, max=0.5),
            # parameter_231
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([240, 480], dtype='float16', min=0, max=0.5),
            # parameter_233
            paddle.uniform([480], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([480, 240], dtype='float16', min=0, max=0.5),
            # parameter_235
            paddle.uniform([240], dtype='float16', min=0, max=0.5),
            # parameter_237
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([240, 720], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([720], dtype='float16', min=0, max=0.5),
            # parameter_240
            paddle.uniform([240, 240], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([240], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([240, 480], dtype='float16', min=0, max=0.5),
            # parameter_245
            paddle.uniform([480], dtype='float16', min=0, max=0.5),
            # parameter_246
            paddle.uniform([480, 240], dtype='float16', min=0, max=0.5),
            # parameter_247
            paddle.uniform([240], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([240, 720], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([720], dtype='float16', min=0, max=0.5),
            # parameter_252
            paddle.uniform([240, 240], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([240], dtype='float16', min=0, max=0.5),
            # parameter_255
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([240, 480], dtype='float16', min=0, max=0.5),
            # parameter_257
            paddle.uniform([480], dtype='float16', min=0, max=0.5),
            # parameter_258
            paddle.uniform([480, 240], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([240], dtype='float16', min=0, max=0.5),
            # parameter_261
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([160, 240, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_266
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([160, 320, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([640, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_276
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([640, 1000], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 256, 256], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_39
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_38
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_37
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_36
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_35
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_34
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_33
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_32
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_31
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_30
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_29
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_28
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_27
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_26
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_25
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_24
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_23
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_22
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_21
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_20
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_19
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_18
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_17
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_16
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_15
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_14
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='float32'),
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
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_2
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_1
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[16, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[96, 256, 1, 1], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[144, 96, 1, 1], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[144, 432], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[432], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[144, 144], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[144], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[144, 288], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[288], dtype='float16'),
            # parameter_96
            paddle.static.InputSpec(shape=[288, 144], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[144], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[144, 432], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[432], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[144, 144], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[144], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[144, 288], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[288], dtype='float16'),
            # parameter_108
            paddle.static.InputSpec(shape=[288, 144], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[144], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[96, 144, 1, 1], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[96, 192, 3, 3], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[384, 96, 1, 1], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128, 384, 1, 1], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[192, 128, 1, 1], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[192, 576], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[576], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_150
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[192, 384], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[384, 192], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[192, 576], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[576], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_160
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_162
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[192, 384], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[384, 192], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_168
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[192, 576], dtype='float16'),
            # parameter_170
            paddle.static.InputSpec(shape=[576], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[192, 384], dtype='float16'),
            # parameter_176
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[384, 192], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_180
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[192, 576], dtype='float16'),
            # parameter_182
            paddle.static.InputSpec(shape=[576], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[192, 384], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[384, 192], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_192
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[128, 192, 1, 1], dtype='float16'),
            # parameter_197
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float16'),
            # parameter_202
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[160, 512, 1, 1], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[160, 160, 3, 3], dtype='float16'),
            # parameter_222
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[240, 160, 1, 1], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[240, 720], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[720], dtype='float16'),
            # parameter_228
            paddle.static.InputSpec(shape=[240, 240], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[240], dtype='float16'),
            # parameter_231
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[240, 480], dtype='float16'),
            # parameter_233
            paddle.static.InputSpec(shape=[480], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[480, 240], dtype='float16'),
            # parameter_235
            paddle.static.InputSpec(shape=[240], dtype='float16'),
            # parameter_237
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[240, 720], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[720], dtype='float16'),
            # parameter_240
            paddle.static.InputSpec(shape=[240, 240], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[240], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[240, 480], dtype='float16'),
            # parameter_245
            paddle.static.InputSpec(shape=[480], dtype='float16'),
            # parameter_246
            paddle.static.InputSpec(shape=[480, 240], dtype='float16'),
            # parameter_247
            paddle.static.InputSpec(shape=[240], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[240, 720], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[720], dtype='float16'),
            # parameter_252
            paddle.static.InputSpec(shape=[240, 240], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[240], dtype='float16'),
            # parameter_255
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[240, 480], dtype='float16'),
            # parameter_257
            paddle.static.InputSpec(shape=[480], dtype='float16'),
            # parameter_258
            paddle.static.InputSpec(shape=[480, 240], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[240], dtype='float16'),
            # parameter_261
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[160, 240, 1, 1], dtype='float16'),
            # parameter_266
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[160, 320, 3, 3], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float16'),
            # parameter_276
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[640, 1000], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[1000], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 256, 256], dtype='float32'),
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