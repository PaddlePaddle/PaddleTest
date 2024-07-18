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
    return [229][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1595_0_0(self, constant_7, constant_6, parameter_323, constant_5, parameter_317, parameter_311, constant_4, constant_3, parameter_305, parameter_303, parameter_297, parameter_291, parameter_289, parameter_283, parameter_281, parameter_275, constant_2, parameter_265, parameter_259, parameter_253, parameter_247, parameter_241, parameter_235, parameter_229, parameter_223, parameter_217, parameter_211, parameter_205, parameter_199, parameter_193, parameter_187, parameter_181, parameter_175, parameter_169, parameter_163, parameter_157, parameter_151, parameter_145, parameter_139, parameter_133, parameter_127, parameter_121, parameter_115, parameter_109, parameter_103, parameter_97, parameter_91, parameter_85, parameter_79, parameter_73, parameter_67, constant_1, constant_0, parameter_61, parameter_55, parameter_49, parameter_43, parameter_37, parameter_31, parameter_25, parameter_19, parameter_13, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_17, parameter_14, parameter_16, parameter_15, parameter_18, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_137, parameter_134, parameter_136, parameter_135, parameter_138, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_167, parameter_164, parameter_166, parameter_165, parameter_168, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_185, parameter_182, parameter_184, parameter_183, parameter_186, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_257, parameter_254, parameter_256, parameter_255, parameter_258, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_269, parameter_266, parameter_268, parameter_267, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_282, parameter_287, parameter_284, parameter_286, parameter_285, parameter_288, parameter_290, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_301, parameter_298, parameter_300, parameter_299, parameter_302, parameter_304, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_321, parameter_318, parameter_320, parameter_319, parameter_322, feed_0):

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x3x-1x-1xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_0 = conv2d_0 + parameter_1

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_0, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_6, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_1 = conv2d_1 + parameter_7

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_1, parameter_8, parameter_9, parameter_10, parameter_11, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu_1, parameter_12, [2, 2], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_2 = conv2d_2 + parameter_13

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_2, parameter_14, parameter_15, parameter_16, parameter_17, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu_2, parameter_18, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_3 = conv2d_3 + parameter_19

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_3, parameter_20, parameter_21, parameter_22, parameter_23, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu_3, parameter_24, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_4 = conv2d_4 + parameter_25

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_4, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__24)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu_4, parameter_30, [2, 2], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_5 = conv2d_5 + parameter_31

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_5, parameter_32, parameter_33, parameter_34, parameter_35, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__30)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu_5, parameter_36, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_6 = conv2d_6 + parameter_37

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_6, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__36)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu_6, parameter_42, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_7 = conv2d_7 + parameter_43

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_7, parameter_44, parameter_45, parameter_46, parameter_47, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__42)

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x3x-1x-1xf32, 16x3x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(feed_0, parameter_48, [2, 2], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_8 = conv2d_8 + parameter_49

        # pd_op.batch_norm_: (-1x16x-1x-1xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_8, parameter_50, parameter_51, parameter_52, parameter_53, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__48)

        # pd_op.conv2d: (-1x8x-1x-1xf32) <- (-1x16x-1x-1xf32, 8x16x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu_8, parameter_54, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1x8x1x1xf32)
        add_9 = conv2d_9 + parameter_55

        # pd_op.batch_norm_: (-1x8x-1x-1xf32, 8xf32, 8xf32, xf32, xf32, None) <- (-1x8x-1x-1xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_9, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__54)

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x8x-1x-1xf32, 16x8x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu_9, parameter_60, [2, 2], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_10 = conv2d_10 + parameter_61

        # pd_op.batch_norm_: (-1x16x-1x-1xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_10, parameter_62, parameter_63, parameter_64, parameter_65, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__60)

        # pd_op.pool2d: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu_8, constant_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # builtin.combine: ([-1x16x-1x-1xf32, -1x16x-1x-1xf32]) <- (-1x16x-1x-1xf32, -1x16x-1x-1xf32)
        combine_0 = [relu_10, pool2d_0]

        # pd_op.concat: (-1x32x-1x-1xf32) <- ([-1x16x-1x-1xf32, -1x16x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, constant_1)

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x32x-1x-1xf32, 16x32x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(concat_0, parameter_66, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_11 = conv2d_11 + parameter_67

        # pd_op.batch_norm_: (-1x16x-1x-1xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_11, parameter_68, parameter_69, parameter_70, parameter_71, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__66)

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 16x16x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu_11, parameter_72, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_12 = conv2d_12 + parameter_73

        # pd_op.batch_norm_: (-1x16x-1x-1xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_12, parameter_74, parameter_75, parameter_76, parameter_77, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__72)

        # pd_op.depthwise_conv2d: (-1x96x-1x-1xf32) <- (-1x16x-1x-1xf32, 96x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(relu_12, parameter_78, [2, 2], [0, 0], 'SAME', 16, [1, 1], 'NCHW')

        # pd_op.add: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, 1x96x1x1xf32)
        add_13 = depthwise_conv2d_0 + parameter_79

        # pd_op.batch_norm_: (-1x96x-1x-1xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_13, parameter_80, parameter_81, parameter_82, parameter_83, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, 96x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(batch_norm__78, parameter_84, [1, 1], [0, 0], 'SAME', 96, [1, 1], 'NCHW')

        # pd_op.add: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, 1x96x1x1xf32)
        add_14 = depthwise_conv2d_1 + parameter_85

        # pd_op.batch_norm_: (-1x96x-1x-1xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_14, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x96x-1x-1xf32, 32x96x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(batch_norm__84, parameter_90, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 1x32x1x1xf32)
        add_15 = conv2d_13 + parameter_91

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_15, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 16x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(relu_11, parameter_96, [2, 2], [0, 0], 'SAME', 16, [1, 1], 'NCHW')

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_16 = depthwise_conv2d_2 + parameter_97

        # pd_op.batch_norm_: (-1x16x-1x-1xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_16, parameter_98, parameter_99, parameter_100, parameter_101, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x16x-1x-1xf32, 32x16x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(batch_norm__96, parameter_102, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 1x32x1x1xf32)
        add_17 = conv2d_14 + parameter_103

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_17, parameter_104, parameter_105, parameter_106, parameter_107, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, -1x32x-1x-1xf32)
        add_18 = batch_norm__90 + batch_norm__102

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(add_18)

        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu_13, parameter_108, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 1x32x1x1xf32)
        add_19 = conv2d_15 + parameter_109

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_19, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__108)

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf32) <- (-1x32x-1x-1xf32, 192x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(relu_14, parameter_114, [1, 1], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

        # pd_op.add: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, 1x192x1x1xf32)
        add_20 = depthwise_conv2d_3 + parameter_115

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_20, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x192x-1x-1xf32, 32x192x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(batch_norm__114, parameter_120, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 1x32x1x1xf32)
        add_21 = conv2d_16 + parameter_121

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_21, parameter_122, parameter_123, parameter_124, parameter_125, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, -1x32x-1x-1xf32)
        add_22 = batch_norm__120 + relu_13

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(add_22)

        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu_15, parameter_126, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 1x32x1x1xf32)
        add_23 = conv2d_17 + parameter_127

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_23, parameter_128, parameter_129, parameter_130, parameter_131, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__126)

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf32) <- (-1x32x-1x-1xf32, 192x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(relu_16, parameter_132, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

        # pd_op.add: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, 1x192x1x1xf32)
        add_24 = depthwise_conv2d_4 + parameter_133

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_24, parameter_134, parameter_135, parameter_136, parameter_137, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, 192x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(batch_norm__132, parameter_138, [1, 1], [0, 0], 'SAME', 192, [1, 1], 'NCHW')

        # pd_op.add: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, 1x192x1x1xf32)
        add_25 = depthwise_conv2d_5 + parameter_139

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_25, parameter_140, parameter_141, parameter_142, parameter_143, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x192x-1x-1xf32, 64x192x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(batch_norm__138, parameter_144, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_26 = conv2d_18 + parameter_145

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_26, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 32x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(relu_15, parameter_150, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

        # pd_op.add: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 1x32x1x1xf32)
        add_27 = depthwise_conv2d_6 + parameter_151

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_27, parameter_152, parameter_153, parameter_154, parameter_155, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x32x-1x-1xf32, 64x32x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(batch_norm__150, parameter_156, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_28 = conv2d_19 + parameter_157

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_28, parameter_158, parameter_159, parameter_160, parameter_161, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_29 = batch_norm__144 + batch_norm__156

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(add_29)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu_17, parameter_162, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_30 = conv2d_20 + parameter_163

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_30, parameter_164, parameter_165, parameter_166, parameter_167, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__162)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x64x-1x-1xf32, 384x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(relu_18, parameter_168, [1, 1], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 1x384x1x1xf32)
        add_31 = depthwise_conv2d_7 + parameter_169

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_31, parameter_170, parameter_171, parameter_172, parameter_173, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x384x-1x-1xf32, 64x384x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(batch_norm__168, parameter_174, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_32 = conv2d_21 + parameter_175

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_32, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_33 = batch_norm__174 + relu_17

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(add_33)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu_19, parameter_180, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_34 = conv2d_22 + parameter_181

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_34, parameter_182, parameter_183, parameter_184, parameter_185, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__180)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x64x-1x-1xf32, 384x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(relu_20, parameter_186, [2, 2], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 1x384x1x1xf32)
        add_35 = depthwise_conv2d_8 + parameter_187

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_35, parameter_188, parameter_189, parameter_190, parameter_191, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x3x3xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(batch_norm__186, parameter_192, [1, 1], [0, 0], 'SAME', 384, [1, 1], 'NCHW')

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 1x384x1x1xf32)
        add_36 = depthwise_conv2d_9 + parameter_193

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_36, parameter_194, parameter_195, parameter_196, parameter_197, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x384x-1x-1xf32, 128x384x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(batch_norm__192, parameter_198, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_37 = conv2d_23 + parameter_199

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_37, parameter_200, parameter_201, parameter_202, parameter_203, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x1x3x3xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(relu_19, parameter_204, [2, 2], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_38 = depthwise_conv2d_10 + parameter_205

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_38, parameter_206, parameter_207, parameter_208, parameter_209, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(batch_norm__204, parameter_210, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_39 = conv2d_24 + parameter_211

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_39, parameter_212, parameter_213, parameter_214, parameter_215, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_40 = batch_norm__198 + batch_norm__210

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(add_40)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu_21, parameter_216, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_41 = conv2d_25 + parameter_217

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_41, parameter_218, parameter_219, parameter_220, parameter_221, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__216)

        # pd_op.depthwise_conv2d: (-1x768x-1x-1xf32) <- (-1x128x-1x-1xf32, 768x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(relu_22, parameter_222, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        # pd_op.add: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, 1x768x1x1xf32)
        add_42 = depthwise_conv2d_11 + parameter_223

        # pd_op.batch_norm_: (-1x768x-1x-1xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_42, parameter_224, parameter_225, parameter_226, parameter_227, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x768x-1x-1xf32, 128x768x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(batch_norm__222, parameter_228, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_43 = conv2d_26 + parameter_229

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_43, parameter_230, parameter_231, parameter_232, parameter_233, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_44 = batch_norm__228 + relu_21

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_23 = paddle._C_ops.relu(add_44)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu_23, parameter_234, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_45 = conv2d_27 + parameter_235

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_45, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__234)

        # pd_op.depthwise_conv2d: (-1x768x-1x-1xf32) <- (-1x128x-1x-1xf32, 768x1x3x3xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(relu_24, parameter_240, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        # pd_op.add: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, 1x768x1x1xf32)
        add_46 = depthwise_conv2d_12 + parameter_241

        # pd_op.batch_norm_: (-1x768x-1x-1xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_46, parameter_242, parameter_243, parameter_244, parameter_245, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x768x-1x-1xf32, 128x768x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(batch_norm__240, parameter_246, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_47 = conv2d_28 + parameter_247

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_47, parameter_248, parameter_249, parameter_250, parameter_251, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_48 = batch_norm__246 + relu_23

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_25 = paddle._C_ops.relu(add_48)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu_25, parameter_252, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_49 = conv2d_29 + parameter_253

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_49, parameter_254, parameter_255, parameter_256, parameter_257, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__252)

        # pd_op.depthwise_conv2d: (-1x768x-1x-1xf32) <- (-1x128x-1x-1xf32, 768x1x3x3xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(relu_26, parameter_258, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        # pd_op.add: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, 1x768x1x1xf32)
        add_50 = depthwise_conv2d_13 + parameter_259

        # pd_op.batch_norm_: (-1x768x-1x-1xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_50, parameter_260, parameter_261, parameter_262, parameter_263, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x768x-1x-1xf32, 128x768x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(batch_norm__258, parameter_264, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_51 = conv2d_30 + parameter_265

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_51, parameter_266, parameter_267, parameter_268, parameter_269, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_52 = batch_norm__264 + relu_25

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_27 = paddle._C_ops.relu(add_52)

        # pd_op.pool2d: (-1x128x1x1xf32) <- (-1x128x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu_27, constant_2, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(pool2d_1, parameter_270, parameter_271, parameter_272, parameter_273, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 128x128x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(batch_norm__270, parameter_274, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add__0 = paddle._C_ops.add(conv2d_31, parameter_275)

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_276, parameter_277, parameter_278, parameter_279, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__0 = paddle._C_ops.relu(batch_norm__276)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x1x1xf32, -1x128x-1x-1xf32)
        add_53 = relu__0 + relu_27

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(add_53, parameter_280, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_54 = conv2d_32 + parameter_281

        # pd_op.depthwise_conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(relu_7, parameter_282, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_55 = depthwise_conv2d_14 + parameter_283

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_55, parameter_284, parameter_285, parameter_286, parameter_287, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(batch_norm__282, parameter_288, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_56 = conv2d_33 + parameter_289

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu_7, parameter_290, [2, 2], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_57 = conv2d_34 + parameter_291

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_57, parameter_292, parameter_293, parameter_294, parameter_295, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.pool2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(batch_norm__288, constant_0, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.depthwise_conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x1x3x3xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(add_54, parameter_296, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_58 = depthwise_conv2d_15 + parameter_297

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_58, parameter_298, parameter_299, parameter_300, parameter_301, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(batch_norm__294, parameter_302, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_59 = conv2d_35 + parameter_303

        # pd_op.sigmoid: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        sigmoid_0 = paddle.nn.functional.sigmoid(add_59)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(add_54, parameter_304, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_60 = conv2d_36 + parameter_305

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_60, parameter_306, parameter_307, parameter_308, parameter_309, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x128x-1x-1xf32)
        shape_0 = paddle._C_ops.shape(add_56)

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_3, constant_4, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__0 = paddle._C_ops.cast(slice_0, paddle.int32)

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(batch_norm__300, cast__0, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.sigmoid: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        sigmoid_1 = paddle.nn.functional.sigmoid(bilinear_interp_0)

        # pd_op.multiply: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        multiply_0 = add_56 * sigmoid_1

        # pd_op.multiply: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        multiply_1 = pool2d_2 * sigmoid_0

        # pd_op.shape: (4xi32) <- (-1x128x-1x-1xf32)
        shape_1 = paddle._C_ops.shape(multiply_0)

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], constant_3, constant_4, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__1 = paddle._C_ops.cast(slice_1, paddle.int32)

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(multiply_1, cast__1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_61 = multiply_0 + bilinear_interp_1

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(add_61, parameter_310, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_62 = conv2d_37 + parameter_311

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_62, parameter_312, parameter_313, parameter_314, parameter_315, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_38 = paddle._C_ops.conv2d(batch_norm__306, parameter_316, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_63 = conv2d_38 + parameter_317

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_63, parameter_318, parameter_319, parameter_320, parameter_321, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__312)

        # pd_op.dropout: (-1x128x-1x-1xf32, None) <- (-1x128x-1x-1xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_28, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x19x-1x-1xf32) <- (-1x128x-1x-1xf32, 19x128x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(dropout_0, parameter_322, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x19x-1x-1xf32) <- (-1x19x-1x-1xf32, 1x19x1x1xf32)
        add_64 = conv2d_39 + parameter_323

        # pd_op.shape: (4xi32) <- (-1x3x-1x-1xf32)
        shape_2 = paddle._C_ops.shape(feed_0)

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], constant_3, constant_4, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__2 = paddle._C_ops.cast(slice_2, paddle.int32)

        # pd_op.bilinear_interp: (-1x19x-1x-1xf32) <- (-1x19x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(add_64, cast__2, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.argmax: (-1x-1x-1xi32) <- (-1x19x-1x-1xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(bilinear_interp_2, constant_6, False, False, paddle.int32)

        # pd_op.scale: (-1x-1x-1xi32) <- (-1x-1x-1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(argmax_0, constant_7, float('0'), True)
        return scale_0



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

    def forward(self, constant_7, constant_6, parameter_323, constant_5, parameter_317, parameter_311, constant_4, constant_3, parameter_305, parameter_303, parameter_297, parameter_291, parameter_289, parameter_283, parameter_281, parameter_275, constant_2, parameter_265, parameter_259, parameter_253, parameter_247, parameter_241, parameter_235, parameter_229, parameter_223, parameter_217, parameter_211, parameter_205, parameter_199, parameter_193, parameter_187, parameter_181, parameter_175, parameter_169, parameter_163, parameter_157, parameter_151, parameter_145, parameter_139, parameter_133, parameter_127, parameter_121, parameter_115, parameter_109, parameter_103, parameter_97, parameter_91, parameter_85, parameter_79, parameter_73, parameter_67, constant_1, constant_0, parameter_61, parameter_55, parameter_49, parameter_43, parameter_37, parameter_31, parameter_25, parameter_19, parameter_13, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_17, parameter_14, parameter_16, parameter_15, parameter_18, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_137, parameter_134, parameter_136, parameter_135, parameter_138, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_167, parameter_164, parameter_166, parameter_165, parameter_168, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_185, parameter_182, parameter_184, parameter_183, parameter_186, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_257, parameter_254, parameter_256, parameter_255, parameter_258, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_269, parameter_266, parameter_268, parameter_267, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_282, parameter_287, parameter_284, parameter_286, parameter_285, parameter_288, parameter_290, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_301, parameter_298, parameter_300, parameter_299, parameter_302, parameter_304, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_321, parameter_318, parameter_320, parameter_319, parameter_322, feed_0):
        return self.builtin_module_1595_0_0(constant_7, constant_6, parameter_323, constant_5, parameter_317, parameter_311, constant_4, constant_3, parameter_305, parameter_303, parameter_297, parameter_291, parameter_289, parameter_283, parameter_281, parameter_275, constant_2, parameter_265, parameter_259, parameter_253, parameter_247, parameter_241, parameter_235, parameter_229, parameter_223, parameter_217, parameter_211, parameter_205, parameter_199, parameter_193, parameter_187, parameter_181, parameter_175, parameter_169, parameter_163, parameter_157, parameter_151, parameter_145, parameter_139, parameter_133, parameter_127, parameter_121, parameter_115, parameter_109, parameter_103, parameter_97, parameter_91, parameter_85, parameter_79, parameter_73, parameter_67, constant_1, constant_0, parameter_61, parameter_55, parameter_49, parameter_43, parameter_37, parameter_31, parameter_25, parameter_19, parameter_13, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_17, parameter_14, parameter_16, parameter_15, parameter_18, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_137, parameter_134, parameter_136, parameter_135, parameter_138, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_167, parameter_164, parameter_166, parameter_165, parameter_168, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_185, parameter_182, parameter_184, parameter_183, parameter_186, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_257, parameter_254, parameter_256, parameter_255, parameter_258, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_269, parameter_266, parameter_268, parameter_267, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_282, parameter_287, parameter_284, parameter_286, parameter_285, parameter_288, parameter_290, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_301, parameter_298, parameter_300, parameter_299, parameter_302, parameter_304, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_321, parameter_318, parameter_320, parameter_319, parameter_322, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1595_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_7
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_6
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # parameter_323
            paddle.uniform([1, 19, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_5
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_4
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            # constant_3
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # parameter_305
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_2
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # parameter_265
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([1, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([1, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([1, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_1
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_0
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            # parameter_61
            paddle.uniform([1, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([1, 8, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([1, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([64, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([16, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([8, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([16, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([16, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([32, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([16, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([32, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([32, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([64, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([64, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([128, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([128, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([128, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([19, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_323
            paddle.static.InputSpec(shape=[1, 19, 1, 1], dtype='float32'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_305
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # constant_2
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_265
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_0
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_61
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[1, 8, 1, 1], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[64, 3, 3, 3], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[16, 3, 3, 3], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[8, 16, 1, 1], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[16, 8, 3, 3], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[16, 32, 3, 3], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[32, 96, 1, 1], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[16, 1, 3, 3], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[32, 16, 1, 1], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[32, 192, 1, 1], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[64, 192, 1, 1], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[64, 32, 1, 1], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[64, 384, 1, 1], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[128, 384, 1, 1], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[128, 768, 1, 1], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[128, 768, 1, 1], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[128, 768, 1, 1], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[19, 128, 1, 1], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
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