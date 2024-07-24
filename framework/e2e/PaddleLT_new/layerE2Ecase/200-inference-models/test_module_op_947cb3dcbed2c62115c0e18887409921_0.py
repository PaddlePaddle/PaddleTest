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
    return [1320][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2370_0_0(self, constant_30, constant_29, parameter_229, parameter_220, parameter_219, constant_28, constant_27, constant_25, constant_24, constant_23, constant_22, constant_21, parameter_54, constant_19, parameter_53, parameter_52, constant_18, parameter_51, parameter_50, constant_17, constant_16, parameter_49, constant_15, parameter_48, parameter_47, constant_20, constant_14, parameter_45, constant_26, parameter_46, constant_13, constant_12, constant_11, constant_10, parameter_44, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_55, parameter_56, parameter_58, parameter_57, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_148, parameter_145, parameter_147, parameter_146, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_247, parameter_248, parameter_249, parameter_250, parameter_251, feed_0):

        # pd_op.conv2d: (-1x32x32x100xf32) <- (-1x1x32x100xf32, 32x1x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x32x100xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x32x100xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x32x100xf32) <- (-1x32x32x100xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.pool2d: (-1x32x16x50xf32) <- (-1x32x32x100xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__0, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x16x50xf32) <- (-1x32x16x50xf32, 64x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(pool2d_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x50xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x50xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x16x50xf32) <- (-1x64x16x50xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.pool2d: (-1x64x8x25xf32) <- (-1x64x16x50xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__1, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x8x25xf32) <- (-1x64x8x25xf32, 128x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x8x25xf32) <- (-1x128x8x25xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.pool2d: (-1x128x4x12xf32) <- (-1x128x8x25xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu__2, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x4x12xf32) <- (-1x128x4x12xf32, 256x128x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_2, parameter_15, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x12xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x4x12xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x4x12xf32) <- (-1x256x4x12xf32)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.pool2d: (-1x256x2x6xf32) <- (-1x256x4x12xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu__3, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x2x6xf32) <- (-1x256x2x6xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x2x6xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x2x6xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x2x6xf32) <- (-1x256x2x6xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.pool2d: (-1x256x1x3xf32) <- (-1x256x2x6xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu__4, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x1x3xf32) <- (-1x256x1x3xf32, 512x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(pool2d_4, parameter_25, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x1x3xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x1x3xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x1x3xf32) <- (-1x512x1x3xf32)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x1x3xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(relu__5, constant_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.flatten_: (-1x512xf32, None) <- (-1x512x1x1xf32)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(pool2d_5, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256xf32) <- (-1x512xf32, 512x256xf32)
        matmul_0 = paddle.matmul(flatten__0, parameter_30, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__0 = paddle._C_ops.add_(matmul_0, parameter_31)

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_32, parameter_33, parameter_34, parameter_35, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256xf32) <- (-1x256xf32)
        relu__6 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.matmul: (-1x54xf32) <- (-1x256xf32, 256x54xf32)
        matmul_1 = paddle.matmul(relu__6, parameter_36, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x54xf32) <- (-1x54xf32, 54xf32)
        add__1 = paddle._C_ops.add_(matmul_1, parameter_37)

        # pd_op.shape: (4xi32) <- (-1x1x32x100xf32)
        shape_0 = paddle._C_ops.shape(feed_0)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, constant_4, constant_5]

        # pd_op.reshape_: (-1x54x1xf32, 0x-1x54xf32) <- (-1x54xf32, [xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__1, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.slice: (-1xf32) <- (-1x54x1xf32, 2xi64, 2xi64)
        slice_1 = paddle._C_ops.slice(reshape__0, [1, 2], constant_6, constant_7, [1, 1], [1, 2])

        # pd_op.sigmoid: (-1xf32) <- (-1xf32)
        sigmoid_0 = paddle.nn.functional.sigmoid(slice_1)

        # pd_op.unsqueeze_: (-1x1xf32, None) <- (-1xf32, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid_0, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unsqueeze_: (-1x1x1xf32, None) <- (-1x1xf32, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze__0, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unsqueeze_: (-1x1x1x1xf32, None) <- (-1x1x1xf32, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze__2, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.slice: (-1x13x1xf32) <- (-1x54x1xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(reshape__0, [1], constant_2, constant_9, [1], [])

        # pd_op.conv2d: (-1x16x4x12xf32) <- (-1x128x4x12xf32, 16x128x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(pool2d_2, parameter_38, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x4x12xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x4x12xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_39, parameter_40, parameter_41, parameter_42, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x16x4x12xf32) <- (-1x16x4x12xf32)
        relu__7 = paddle._C_ops.relu_(batch_norm__42)

        # pd_op.conv2d: (-1x1x4x12xf32) <- (-1x16x4x12xf32, 1x16x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu__7, parameter_43, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1x4x12xf32) <- (-1x1x4x12xf32, 1x1x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_7, parameter_44)

        # pd_op.pool2d: (-1x1x2x6xf32) <- (-1x1x4x12xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(add__2, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.sigmoid_: (-1x1x2x6xf32) <- (-1x1x2x6xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(pool2d_6)

        # pd_op.bilinear_interp: (-1x1x32x100xf32) <- (-1x1x2x6xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(sigmoid__0, None, None, None, 'NCHW', -1, 32, 100, [], 'bilinear', False, 0)

        # pd_op.slice: (-1x40x1xf32) <- (-1x54x1xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(reshape__0, [1], constant_10, constant_11, [1], [])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_1 = [slice_0, constant_12, constant_13]

        # pd_op.reshape_: (-1x20x2xf32, 0x-1x40x1xf32) <- (-1x40x1xf32, [xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_3, combine_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full([1], float('-1'), paddle.float64, paddle.core.CPUPlace())

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.float64, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.linspace: (10xf64) <- (1xf64, 1xf64, 1xi32)
        linspace_0 = paddle._C_ops.linspace(full_0, full_1, full_2, paddle.float64, paddle.framework._current_expected_place())

        # builtin.combine: ([10xf64, 10xf64]) <- (10xf64, 10xf64)
        combine_2 = [linspace_0, parameter_45]

        # pd_op.stack: (10x2xf64) <- ([10xf64, 10xf64])
        stack_0 = paddle._C_ops.stack(combine_2, 1)

        # builtin.combine: ([10xf64, 10xf64]) <- (10xf64, 10xf64)
        combine_3 = [linspace_0, parameter_46]

        # pd_op.stack: (10x2xf64) <- ([10xf64, 10xf64])
        stack_1 = paddle._C_ops.stack(combine_3, 1)

        # builtin.combine: ([10x2xf64, 10x2xf64]) <- (10x2xf64, 10x2xf64)
        combine_4 = [stack_0, stack_1]

        # pd_op.concat: (20x2xf64) <- ([10x2xf64, 10x2xf64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_4, constant_14)

        # builtin.combine: ([100xf64, 32xf64]) <- (100xf64, 32xf64)
        combine_5 = [parameter_47, parameter_48]

        # pd_op.meshgrid: ([100x32xf64, 100x32xf64]) <- ([100xf64, 32xf64])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_5)

        # builtin.slice: (100x32xf64) <- ([100x32xf64, 100x32xf64])
        slice_4 = meshgrid_0[0]

        # builtin.slice: (100x32xf64) <- ([100x32xf64, 100x32xf64])
        slice_5 = meshgrid_0[1]

        # builtin.combine: ([100x32xf64, 100x32xf64]) <- (100x32xf64, 100x32xf64)
        combine_6 = [slice_4, slice_5]

        # pd_op.stack: (100x32x2xf64) <- ([100x32xf64, 100x32xf64])
        stack_2 = paddle._C_ops.stack(combine_6, 2)

        # pd_op.transpose: (32x100x2xf64) <- (100x32x2xf64)
        transpose_0 = paddle._C_ops.transpose(stack_2, [1, 0, 2])

        # pd_op.reshape_: (3200x2xf64, 0x32x100x2xf64) <- (32x100x2xf64, 2xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.reshape: (1x20x2xf64, 0x20x2xf64) <- (20x2xf64, 3xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(concat_0, constant_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.reshape: (20x1x2xf64, 0x20x2xf64) <- (20x2xf64, 3xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(concat_0, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.subtract: (20x20x2xf64) <- (1x20x2xf64, 20x1x2xf64)
        subtract_0 = reshape_0 - reshape_2

        # pd_op.p_norm: (20x20xf64) <- (20x20x2xf64)
        p_norm_0 = paddle._C_ops.p_norm(subtract_0, float('2'), 2, float('1e-12'), False, False)

        # pd_op.add_: (20x20xf64) <- (20x20xf64, 20x20xf64)
        add__3 = paddle._C_ops.add_(p_norm_0, parameter_49)

        # pd_op.elementwise_pow: (20x20xf64) <- (20x20xf64, xf64)
        elementwise_pow_0 = paddle.pow(add__3, parameter_50)

        # pd_op.log_: (20x20xf64) <- (20x20xf64)
        log__0 = paddle._C_ops.log_(add__3)

        # pd_op.multiply_: (20x20xf64) <- (20x20xf64, 20x20xf64)
        multiply__0 = paddle._C_ops.multiply_(elementwise_pow_0, log__0)

        # builtin.combine: ([20x1xf64, 20x2xf64, 20x20xf64]) <- (20x1xf64, 20x2xf64, 20x20xf64)
        combine_7 = [parameter_51, concat_0, multiply__0]

        # pd_op.concat: (20x23xf64) <- ([20x1xf64, 20x2xf64, 20x20xf64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_7, constant_18)

        # pd_op.transpose: (2x20xf64) <- (20x2xf64)
        transpose_1 = paddle._C_ops.transpose(concat_0, [1, 0])

        # builtin.combine: ([2x3xf64, 2x20xf64]) <- (2x3xf64, 2x20xf64)
        combine_8 = [parameter_52, transpose_1]

        # pd_op.concat: (2x23xf64) <- ([2x3xf64, 2x20xf64], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_8, constant_18)

        # builtin.combine: ([20x23xf64, 2x23xf64, 1x23xf64]) <- (20x23xf64, 2x23xf64, 1x23xf64)
        combine_9 = [concat_1, concat_2, parameter_53]

        # pd_op.concat: (23x23xf64) <- ([20x23xf64, 2x23xf64, 1x23xf64], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_9, constant_14)

        # pd_op.inverse: (23x23xf64) <- (23x23xf64)
        inverse_0 = paddle._C_ops.inverse(concat_3)

        # pd_op.cast: (23x23xf32) <- (23x23xf64)
        cast_0 = paddle._C_ops.cast(inverse_0, paddle.float32)

        # pd_op.unsqueeze: (3200x1x2xf64, None) <- (3200x2xf64, 1xi64)
        unsqueeze_0, unsqueeze_1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(reshape__4, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.tile: (3200x20x2xf64) <- (3200x1x2xf64, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, constant_19)

        # pd_op.unsqueeze_: (1x20x2xf64, None) <- (20x2xf64, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(concat_0, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.subtract_: (3200x20x2xf64) <- (3200x20x2xf64, 1x20x2xf64)
        subtract__0 = paddle._C_ops.subtract_(tile_0, unsqueeze__6)

        # pd_op.p_norm: (3200x20xf64) <- (3200x20x2xf64)
        p_norm_1 = paddle._C_ops.p_norm(subtract__0, float('2'), 2, float('1e-12'), False, False)

        # pd_op.square: (3200x20xf64) <- (3200x20xf64)
        square_0 = paddle._C_ops.square(p_norm_1)

        # pd_op.scale_: (3200x20xf64) <- (3200x20xf64, 1xf32)
        scale__0 = paddle._C_ops.scale_(p_norm_1, constant_20, float('1e-06'), True)

        # pd_op.log_: (3200x20xf64) <- (3200x20xf64)
        log__1 = paddle._C_ops.log_(scale__0)

        # pd_op.multiply_: (3200x20xf64) <- (3200x20xf64, 3200x20xf64)
        multiply__1 = paddle._C_ops.multiply_(square_0, log__1)

        # builtin.combine: ([3200x1xf64, 3200x2xf64, 3200x20xf64]) <- (3200x1xf64, 3200x2xf64, 3200x20xf64)
        combine_10 = [parameter_54, reshape__4, multiply__1]

        # pd_op.concat: (3200x23xf64) <- ([3200x1xf64, 3200x2xf64, 3200x20xf64], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_10, constant_18)

        # pd_op.cast: (3200x23xf32) <- (3200x23xf64)
        cast_1 = paddle._C_ops.cast(concat_4, paddle.float32)

        # pd_op.shape: (3xi32) <- (-1x20x2xf32)
        shape_1 = paddle._C_ops.shape(reshape__2)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_1, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32]) <- (xi32, 1xi32)
        combine_11 = [slice_6, constant_21]

        # pd_op.reshape: (-1x40xf32, 0x-1x20x2xf32) <- (-1x20x2xf32, [xi32, 1xi32])
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__2, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x6xf32) <- (-1x40xf32, 40x6xf32)
        matmul_2 = paddle.matmul(reshape_4, parameter_55, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x6xf32) <- (-1x6xf32, 6xf32)
        add__4 = paddle._C_ops.add_(matmul_2, parameter_56)

        # pd_op.reshape_: (-1x3x2xf32, 0x-1x6xf32) <- (-1x6xf32, 3xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__4, constant_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x20x2xf32, -1x3x2xf32]) <- (-1x20x2xf32, -1x3x2xf32)
        combine_12 = [reshape__2, reshape__6]

        # pd_op.concat: (-1x23x2xf32) <- ([-1x20x2xf32, -1x3x2xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_12, constant_18)

        # pd_op.matmul: (-1x23x2xf32) <- (23x23xf32, -1x23x2xf32)
        matmul_3 = paddle.matmul(cast_0, concat_5, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x3200x2xf32) <- (3200x23xf32, -1x23x2xf32)
        matmul_4 = paddle.matmul(cast_1, matmul_3, transpose_x=False, transpose_y=False)

        # pd_op.shape: (3xi32) <- (-1x3200x2xf32)
        shape_2 = paddle._C_ops.shape(matmul_4)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_2, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_7, constant_23, constant_24, constant_13]

        # pd_op.reshape_: (-1x32x100x2xf32, 0x-1x3200x2xf32) <- (-1x3200x2xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_4, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale: (-1x1x32x100xf32) <- (-1x1x32x100xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(feed_0, constant_20, float('1'), True)

        # pd_op.scale_: (-1x1x32x100xf32) <- (-1x1x32x100xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(scale_0, constant_25, float('0'), True)

        # pd_op.scale: (-1x1x1x1xf32) <- (-1x1x1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(unsqueeze__4, constant_26, float('1'), True)

        # pd_op.multiply_: (-1x1x32x100xf32) <- (-1x1x32x100xf32, -1x1x1x1xf32)
        multiply__2 = paddle._C_ops.multiply_(scale__1, scale_1)

        # pd_op.multiply_: (-1x1x32x100xf32) <- (-1x1x32x100xf32, -1x1x1x1xf32)
        multiply__3 = paddle._C_ops.multiply_(bilinear_interp_0, unsqueeze__4)

        # pd_op.add_: (-1x1x32x100xf32) <- (-1x1x32x100xf32, -1x1x32x100xf32)
        add__5 = paddle._C_ops.add_(multiply__2, multiply__3)

        # pd_op.unsqueeze_: (-1x13x1x1xf32, None) <- (-1x13x1xf32, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(slice_2, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unsqueeze_: (-1x13x1x1x1xf32, None) <- (-1x13x1x1xf32, 1xi64)
        unsqueeze__10, unsqueeze__11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze__8, constant_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_0 = paddle._C_ops.pow(add__5, float('0.03'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_1 = paddle._C_ops.pow(add__5, float('33.33'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_2 = paddle._C_ops.pow(add__5, float('0.08'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_3 = paddle._C_ops.pow(add__5, float('12.5'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_4 = paddle._C_ops.pow(add__5, float('0.16'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_5 = paddle._C_ops.pow(add__5, float('6.25'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_6 = paddle._C_ops.pow(add__5, float('0.27'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_7 = paddle._C_ops.pow(add__5, float('3.7'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_8 = paddle._C_ops.pow(add__5, float('0.43'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_9 = paddle._C_ops.pow(add__5, float('2.33'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_10 = paddle._C_ops.pow(add__5, float('0.66'))

        # pd_op.pow: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow_11 = paddle._C_ops.pow(add__5, float('1.52'))

        # pd_op.pow_: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        pow__0 = paddle._C_ops.pow_(add__5, float('1'))

        # builtin.combine: ([-1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32]) <- (-1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32)
        combine_14 = [pow_0, pow_1, pow_2, pow_3, pow_4, pow_5, pow_6, pow_7, pow_8, pow_9, pow_10, pow_11, pow__0]

        # pd_op.stack: (-1x13x1x32x100xf32) <- ([-1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32, -1x1x32x100xf32])
        stack_3 = paddle._C_ops.stack(combine_14, 1)

        # pd_op.multiply_: (-1x13x1x32x100xf32) <- (-1x13x1x32x100xf32, -1x13x1x1x1xf32)
        multiply__4 = paddle._C_ops.multiply_(stack_3, unsqueeze__10)

        # pd_op.sum: (-1x1x32x100xf32) <- (-1x13x1x32x100xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply__4, constant_3, None, False)

        # pd_op.instance_norm: (-1x1x32x100xf32, None, None) <- (-1x1x32x100xf32, 1xf32, 1xf32)
        instance_norm_0, instance_norm_1, instance_norm_2 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(sum_0, parameter_57, parameter_58, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.sigmoid_: (-1x1x32x100xf32) <- (-1x1x32x100xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(instance_norm_0)

        # pd_op.scale_: (-1x1x32x100xf32) <- (-1x1x32x100xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(sigmoid__1, constant_27, float('0'), True)

        # pd_op.scale_: (-1x1x32x100xf32) <- (-1x1x32x100xf32, 1xf32)
        scale__3 = paddle._C_ops.scale_(scale__2, constant_20, float('-1'), True)

        # pd_op.grid_sample: (-1x1x32x100xf32) <- (-1x1x32x100xf32, -1x32x100x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(scale__3, reshape__8, 'bilinear', 'border', True)

        # pd_op.conv2d: (-1x32x32x100xf32) <- (-1x1x32x100xf32, 32x1x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(grid_sample_0, parameter_59, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x32x100xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x32x100xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x32x100xf32) <- (-1x32x32x100xf32)
        relu__8 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.conv2d: (-1x64x32x100xf32) <- (-1x32x32x100xf32, 64x32x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__8, parameter_64, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x32x100xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x32x100xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_65, parameter_66, parameter_67, parameter_68, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x32x100xf32) <- (-1x64x32x100xf32)
        relu__9 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.pool2d: (-1x64x16x50xf32) <- (-1x64x32x100xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(relu__9, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x16x50xf32) <- (-1x64x16x50xf32, 128x64x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(pool2d_7, parameter_69, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x50xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x16x50xf32) <- (-1x128x16x50xf32)
        relu__10 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.conv2d: (-1x128x16x50xf32) <- (-1x128x16x50xf32, 128x128x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu__10, parameter_74, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x50xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_75, parameter_76, parameter_77, parameter_78, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x16x50xf32) <- (-1x64x16x50xf32, 128x64x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(pool2d_7, parameter_79, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x50xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_80, parameter_81, parameter_82, parameter_83, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x16x50xf32) <- (-1x128x16x50xf32, -1x128x16x50xf32)
        add__6 = paddle._C_ops.add_(batch_norm__66, batch_norm__72)

        # pd_op.relu_: (-1x128x16x50xf32) <- (-1x128x16x50xf32)
        relu__11 = paddle._C_ops.relu_(add__6)

        # pd_op.conv2d: (-1x128x16x50xf32) <- (-1x128x16x50xf32, 128x128x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu__11, parameter_84, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x50xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_85, parameter_86, parameter_87, parameter_88, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x16x50xf32) <- (-1x128x16x50xf32)
        relu__12 = paddle._C_ops.relu_(batch_norm__78)

        # pd_op.pool2d: (-1x128x8x25xf32) <- (-1x128x16x50xf32, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(relu__12, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x128x8x25xf32, 256x128x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(pool2d_8, parameter_89, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_90, parameter_91, parameter_92, parameter_93, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__13 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x256x8x25xf32, 256x256x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__13, parameter_94, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_95, parameter_96, parameter_97, parameter_98, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x128x8x25xf32, 256x128x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(pool2d_8, parameter_99, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_100, parameter_101, parameter_102, parameter_103, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x25xf32) <- (-1x256x8x25xf32, -1x256x8x25xf32)
        add__7 = paddle._C_ops.add_(batch_norm__90, batch_norm__96)

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__14 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x256x8x25xf32, 256x256x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu__14, parameter_104, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_105, parameter_106, parameter_107, parameter_108, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__15 = paddle._C_ops.relu_(batch_norm__102)

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x256x8x25xf32, 256x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu__15, parameter_109, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x25xf32) <- (-1x256x8x25xf32, -1x256x8x25xf32)
        add__8 = paddle._C_ops.add_(batch_norm__108, relu__14)

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__16 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x256x8x25xf32, 256x256x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu__16, parameter_114, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_115, parameter_116, parameter_117, parameter_118, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__17 = paddle._C_ops.relu_(batch_norm__114)

        # pd_op.pool2d: (-1x256x4x26xf32) <- (-1x256x8x25xf32, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(relu__17, constant_0, [2, 1], [0, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x256x4x26xf32, 512x256x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(pool2d_9, parameter_119, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_120, parameter_121, parameter_122, parameter_123, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__18 = paddle._C_ops.relu_(batch_norm__120)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu__18, parameter_124, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_125, parameter_126, parameter_127, parameter_128, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x256x4x26xf32, 512x256x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(pool2d_9, parameter_129, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_130, parameter_131, parameter_132, parameter_133, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__9 = paddle._C_ops.add_(batch_norm__126, batch_norm__132)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__19 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu__19, parameter_134, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_135, parameter_136, parameter_137, parameter_138, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__20 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__20, parameter_139, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_140, parameter_141, parameter_142, parameter_143, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__10 = paddle._C_ops.add_(batch_norm__144, relu__19)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__21 = paddle._C_ops.relu_(add__10)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu__21, parameter_144, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_145, parameter_146, parameter_147, parameter_148, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__22 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu__22, parameter_149, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_150, parameter_151, parameter_152, parameter_153, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__11 = paddle._C_ops.add_(batch_norm__156, relu__21)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__23 = paddle._C_ops.relu_(add__11)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu__23, parameter_154, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_155, parameter_156, parameter_157, parameter_158, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__24 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__24, parameter_159, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_160, parameter_161, parameter_162, parameter_163, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__12 = paddle._C_ops.add_(batch_norm__168, relu__23)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__25 = paddle._C_ops.relu_(add__12)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu__25, parameter_164, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_165, parameter_166, parameter_167, parameter_168, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__26 = paddle._C_ops.relu_(batch_norm__174)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__26, parameter_169, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_170, parameter_171, parameter_172, parameter_173, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__13 = paddle._C_ops.add_(batch_norm__180, relu__25)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__27 = paddle._C_ops.relu_(add__13)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu__27, parameter_174, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_175, parameter_176, parameter_177, parameter_178, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__28 = paddle._C_ops.relu_(batch_norm__186)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu__28, parameter_179, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_180, parameter_181, parameter_182, parameter_183, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__29 = paddle._C_ops.relu_(batch_norm__192)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(relu__29, parameter_184, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_185, parameter_186, parameter_187, parameter_188, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__14 = paddle._C_ops.add_(batch_norm__198, relu__28)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__30 = paddle._C_ops.relu_(add__14)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu__30, parameter_189, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_190, parameter_191, parameter_192, parameter_193, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__31 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(relu__31, parameter_194, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_195, parameter_196, parameter_197, parameter_198, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__15 = paddle._C_ops.add_(batch_norm__210, relu__30)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__32 = paddle._C_ops.relu_(add__15)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(relu__32, parameter_199, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_200, parameter_201, parameter_202, parameter_203, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__33 = paddle._C_ops.relu_(batch_norm__216)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(relu__33, parameter_204, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_205, parameter_206, parameter_207, parameter_208, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__16 = paddle._C_ops.add_(batch_norm__222, relu__32)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__34 = paddle._C_ops.relu_(add__16)

        # pd_op.conv2d: (-1x512x2x27xf32) <- (-1x512x4x26xf32, 512x512x2x2xf32)
        conv2d_38 = paddle._C_ops.conv2d(relu__34, parameter_209, [2, 1], [0, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x27xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x2x27xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x2x27xf32) <- (-1x512x2x27xf32)
        relu__35 = paddle._C_ops.relu_(batch_norm__228)

        # pd_op.conv2d: (-1x512x1x26xf32) <- (-1x512x2x27xf32, 512x512x2x2xf32)
        conv2d_39 = paddle._C_ops.conv2d(relu__35, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x1x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x1x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_215, parameter_216, parameter_217, parameter_218, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x1x26xf32) <- (-1x512x1x26xf32)
        relu__36 = paddle._C_ops.relu_(batch_norm__234)

        # pd_op.squeeze_: (-1x512x26xf32, None) <- (-1x512x1x26xf32, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(relu__36, constant_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x26x512xf32) <- (-1x512x26xf32)
        transpose_2 = paddle._C_ops.transpose(squeeze__0, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x26x512xf32)
        shape_3 = paddle._C_ops.shape(transpose_2)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_3, [0], constant_2, constant_3, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_15 = [parameter_219, slice_8, parameter_220]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(slice_8, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_16 = [parameter_219, memcpy_h2d_0, parameter_220]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_4 = paddle._C_ops.stack(combine_16, 0)

        # pd_op.full_with_tensor: (2x-1x256xf32) <- (1xf32, 3xi32)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(full_3, stack_4, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_17 = [parameter_219, slice_8, parameter_220]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_1 = paddle._C_ops.memcpy_h2d(slice_8, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_18 = [parameter_219, memcpy_h2d_1, parameter_220]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_5 = paddle._C_ops.stack(combine_18, 0)

        # pd_op.full_with_tensor: (2x-1x256xf32) <- (1xf32, 3xi32)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_3, stack_5, paddle.float32)

        # pd_op.transpose: (26x-1x512xf32) <- (-1x26x512xf32)
        transpose_3 = paddle._C_ops.transpose(transpose_2, [1, 0, 2])

        # builtin.combine: ([2x-1x256xf32, 2x-1x256xf32]) <- (2x-1x256xf32, 2x-1x256xf32)
        combine_19 = [full_with_tensor_0, full_with_tensor_1]

        # builtin.combine: ([1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32]) <- (1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        combine_20 = [parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228]

        # pd_op.rnn_: (26x-1x512xf32, xui8, [2x-1x256xf32, 2x-1x256xf32], xui8) <- (26x-1x512xf32, [2x-1x256xf32, 2x-1x256xf32], [1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32], None, xui8)
        rnn__0, rnn__1, rnn__2, rnn__3 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_3, combine_19, combine_20, None, parameter_229, float('0'), True, 512, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x26x512xf32) <- (26x-1x512xf32)
        transpose_4 = paddle._C_ops.transpose(rnn__0, [1, 0, 2])

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_5 = paddle.matmul(transpose_4, parameter_230, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, 256xf32)
        add__17 = paddle._C_ops.add_(matmul_5, parameter_231)

        # pd_op.shape: (3xi32) <- (-1x26x256xf32)
        shape_4 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_4, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_21 = [parameter_219, slice_9, parameter_220]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_2 = paddle._C_ops.memcpy_h2d(slice_9, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_22 = [parameter_219, memcpy_h2d_2, parameter_220]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_6 = paddle._C_ops.stack(combine_22, 0)

        # pd_op.full_with_tensor: (2x-1x256xf32) <- (1xf32, 3xi32)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(full_3, stack_6, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_23 = [parameter_219, slice_9, parameter_220]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_3 = paddle._C_ops.memcpy_h2d(slice_9, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_24 = [parameter_219, memcpy_h2d_3, parameter_220]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_7 = paddle._C_ops.stack(combine_24, 0)

        # pd_op.full_with_tensor: (2x-1x256xf32) <- (1xf32, 3xi32)
        full_with_tensor_3 = paddle._C_ops.full_with_tensor(full_3, stack_7, paddle.float32)

        # pd_op.transpose: (26x-1x256xf32) <- (-1x26x256xf32)
        transpose_5 = paddle._C_ops.transpose(add__17, [1, 0, 2])

        # builtin.combine: ([2x-1x256xf32, 2x-1x256xf32]) <- (2x-1x256xf32, 2x-1x256xf32)
        combine_25 = [full_with_tensor_2, full_with_tensor_3]

        # builtin.combine: ([1024x256xf32, 1024x256xf32, 1024x256xf32, 1024x256xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32]) <- (1024x256xf32, 1024x256xf32, 1024x256xf32, 1024x256xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        combine_26 = [parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239]

        # pd_op.rnn_: (26x-1x512xf32, xui8, [2x-1x256xf32, 2x-1x256xf32], xui8) <- (26x-1x256xf32, [2x-1x256xf32, 2x-1x256xf32], [1024x256xf32, 1024x256xf32, 1024x256xf32, 1024x256xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32], None, xui8)
        rnn__4, rnn__5, rnn__6, rnn__7 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_5, combine_25, combine_26, None, parameter_229, float('0'), True, 256, 256, 1, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x26x512xf32) <- (26x-1x512xf32)
        transpose_6 = paddle._C_ops.transpose(rnn__4, [1, 0, 2])

        # pd_op.matmul: (-1x26x512xf32) <- (-1x26x512xf32, 512x512xf32)
        matmul_6 = paddle.matmul(transpose_6, parameter_240, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x26x512xf32) <- (-1x26x512xf32, 512xf32)
        add__18 = paddle._C_ops.add_(matmul_6, parameter_241)

        # pd_op.shape: (3xi32) <- (-1x26x512xf32)
        shape_5 = paddle._C_ops.shape(add__18)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(shape_5, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_27 = [slice_10, parameter_220]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_4 = paddle._C_ops.memcpy_h2d(slice_10, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_28 = [memcpy_h2d_4, parameter_220]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_8 = paddle._C_ops.stack(combine_28, 0)

        # pd_op.full_with_tensor: (-1x256xf32) <- (1xf32, 2xi32)
        full_with_tensor_4 = paddle._C_ops.full_with_tensor(full_3, stack_8, paddle.float32)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_29 = [slice_10, parameter_220]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_5 = paddle._C_ops.memcpy_h2d(slice_10, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_30 = [memcpy_h2d_5, parameter_220]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_9 = paddle._C_ops.stack(combine_30, 0)

        # pd_op.full_with_tensor: (-1x256xf32) <- (1xf32, 2xi32)
        full_with_tensor_5 = paddle._C_ops.full_with_tensor(full_3, stack_9, paddle.float32)

        # builtin.combine: ([xi32]) <- (xi32)
        combine_31 = [slice_10]

        # pd_op.stack: (1xi32) <- ([xi32])
        stack_10 = paddle._C_ops.stack(combine_31, 0)

        # pd_op.full_with_tensor: (-1xi32) <- (1xf32, 1xi32)
        full_with_tensor_6 = paddle._C_ops.full_with_tensor(full_3, stack_10, paddle.int32)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(full_with_tensor_6 % paddle.cast(constant_29, full_with_tensor_6.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_7 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_8 = paddle.matmul(full_with_tensor_4, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__19 = paddle._C_ops.add_(matmul_8, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__12, unsqueeze__13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__19, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__20 = paddle._C_ops.add_(matmul_7, unsqueeze__12)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__0 = paddle._C_ops.tanh_(add__20)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_9 = paddle.matmul(tanh__0, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__0 = paddle._C_ops.softmax_(matmul_9, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_7 = paddle._C_ops.transpose(softmax__0, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_10 = paddle.matmul(transpose_7, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__2, squeeze__3 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_10, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_32 = [squeeze__2, one_hot_0]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_32, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_11 = paddle.matmul(concat_6, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__21 = paddle._C_ops.add_(matmul_11, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_12 = paddle.matmul(full_with_tensor_4, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__22 = paddle._C_ops.add_(add__21, matmul_12)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__23 = paddle._C_ops.add_(add__22, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add__23, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_11 = split_with_num_0[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__2 = paddle._C_ops.sigmoid_(slice_11)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_12 = split_with_num_0[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__3 = paddle._C_ops.sigmoid_(slice_12)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_13 = split_with_num_0[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__4 = paddle._C_ops.sigmoid_(slice_13)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__5 = paddle._C_ops.multiply_(sigmoid__3, full_with_tensor_5)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_14 = split_with_num_0[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__1 = paddle._C_ops.tanh_(slice_14)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__6 = paddle._C_ops.multiply_(sigmoid__2, tanh__1)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__24 = paddle._C_ops.add_(multiply__5, multiply__6)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_0 = paddle._C_ops.tanh(add__24)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__7 = paddle._C_ops.multiply_(sigmoid__4, tanh_0)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_13 = paddle.matmul(multiply__7, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__25 = paddle._C_ops.add_(matmul_13, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_2, unsqueeze_3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__25, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(add__25, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_1 = paddle._C_ops.one_hot(argmax_0 % paddle.cast(constant_29, argmax_0.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_14 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_15 = paddle.matmul(multiply__7, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__26 = paddle._C_ops.add_(matmul_15, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__14, unsqueeze__15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__26, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__27 = paddle._C_ops.add_(matmul_14, unsqueeze__14)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__2 = paddle._C_ops.tanh_(add__27)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_16 = paddle.matmul(tanh__2, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__1 = paddle._C_ops.softmax_(matmul_16, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_8 = paddle._C_ops.transpose(softmax__1, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_17 = paddle.matmul(transpose_8, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__4, squeeze__5 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_17, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_33 = [squeeze__4, one_hot_1]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_33, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_18 = paddle.matmul(concat_7, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__28 = paddle._C_ops.add_(matmul_18, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_19 = paddle.matmul(multiply__7, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__29 = paddle._C_ops.add_(add__28, matmul_19)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__30 = paddle._C_ops.add_(add__29, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(add__30, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_15 = split_with_num_1[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__5 = paddle._C_ops.sigmoid_(slice_15)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_16 = split_with_num_1[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__6 = paddle._C_ops.sigmoid_(slice_16)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_17 = split_with_num_1[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__7 = paddle._C_ops.sigmoid_(slice_17)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__8 = paddle._C_ops.multiply_(sigmoid__6, add__24)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_18 = split_with_num_1[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__3 = paddle._C_ops.tanh_(slice_18)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__9 = paddle._C_ops.multiply_(sigmoid__5, tanh__3)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__31 = paddle._C_ops.add_(multiply__8, multiply__9)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_1 = paddle._C_ops.tanh(add__31)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__10 = paddle._C_ops.multiply_(sigmoid__7, tanh_1)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_20 = paddle.matmul(multiply__10, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__32 = paddle._C_ops.add_(matmul_20, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_4, unsqueeze_5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__32, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x1x70xf32, -1x1x70xf32]) <- (-1x1x70xf32, -1x1x70xf32)
        combine_34 = [unsqueeze_2, unsqueeze_4]

        # pd_op.concat: (-1x2x70xf32) <- ([-1x1x70xf32, -1x1x70xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_34, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_1 = paddle._C_ops.argmax(add__32, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_2 = paddle._C_ops.one_hot(argmax_1 % paddle.cast(constant_29, argmax_1.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_21 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_22 = paddle.matmul(multiply__10, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__33 = paddle._C_ops.add_(matmul_22, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__16, unsqueeze__17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__33, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__34 = paddle._C_ops.add_(matmul_21, unsqueeze__16)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__4 = paddle._C_ops.tanh_(add__34)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_23 = paddle.matmul(tanh__4, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__2 = paddle._C_ops.softmax_(matmul_23, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_9 = paddle._C_ops.transpose(softmax__2, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_24 = paddle.matmul(transpose_9, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__6, squeeze__7 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_24, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_35 = [squeeze__6, one_hot_2]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_35, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_25 = paddle.matmul(concat_9, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__35 = paddle._C_ops.add_(matmul_25, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_26 = paddle.matmul(multiply__10, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__36 = paddle._C_ops.add_(add__35, matmul_26)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__37 = paddle._C_ops.add_(add__36, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(add__37, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_19 = split_with_num_2[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__8 = paddle._C_ops.sigmoid_(slice_19)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_20 = split_with_num_2[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__9 = paddle._C_ops.sigmoid_(slice_20)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_21 = split_with_num_2[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__10 = paddle._C_ops.sigmoid_(slice_21)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__11 = paddle._C_ops.multiply_(sigmoid__9, add__31)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_22 = split_with_num_2[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__5 = paddle._C_ops.tanh_(slice_22)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__12 = paddle._C_ops.multiply_(sigmoid__8, tanh__5)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__38 = paddle._C_ops.add_(multiply__11, multiply__12)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_2 = paddle._C_ops.tanh(add__38)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__13 = paddle._C_ops.multiply_(sigmoid__10, tanh_2)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_27 = paddle.matmul(multiply__13, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__39 = paddle._C_ops.add_(matmul_27, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_6, unsqueeze_7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__39, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x2x70xf32, -1x1x70xf32]) <- (-1x2x70xf32, -1x1x70xf32)
        combine_36 = [concat_8, unsqueeze_6]

        # pd_op.concat: (-1x3x70xf32) <- ([-1x2x70xf32, -1x1x70xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_36, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_2 = paddle._C_ops.argmax(add__39, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_3 = paddle._C_ops.one_hot(argmax_2 % paddle.cast(constant_29, argmax_2.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_28 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_29 = paddle.matmul(multiply__13, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__40 = paddle._C_ops.add_(matmul_29, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__18, unsqueeze__19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__40, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__41 = paddle._C_ops.add_(matmul_28, unsqueeze__18)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__6 = paddle._C_ops.tanh_(add__41)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_30 = paddle.matmul(tanh__6, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__3 = paddle._C_ops.softmax_(matmul_30, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_10 = paddle._C_ops.transpose(softmax__3, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_31 = paddle.matmul(transpose_10, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__8, squeeze__9 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_31, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_37 = [squeeze__8, one_hot_3]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_37, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_32 = paddle.matmul(concat_11, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__42 = paddle._C_ops.add_(matmul_32, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_33 = paddle.matmul(multiply__13, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__43 = paddle._C_ops.add_(add__42, matmul_33)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__44 = paddle._C_ops.add_(add__43, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(add__44, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_23 = split_with_num_3[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__11 = paddle._C_ops.sigmoid_(slice_23)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_24 = split_with_num_3[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__12 = paddle._C_ops.sigmoid_(slice_24)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_25 = split_with_num_3[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__13 = paddle._C_ops.sigmoid_(slice_25)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__14 = paddle._C_ops.multiply_(sigmoid__12, add__38)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_26 = split_with_num_3[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__7 = paddle._C_ops.tanh_(slice_26)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__15 = paddle._C_ops.multiply_(sigmoid__11, tanh__7)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__45 = paddle._C_ops.add_(multiply__14, multiply__15)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_3 = paddle._C_ops.tanh(add__45)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__16 = paddle._C_ops.multiply_(sigmoid__13, tanh_3)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_34 = paddle.matmul(multiply__16, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__46 = paddle._C_ops.add_(matmul_34, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_8, unsqueeze_9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__46, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x3x70xf32, -1x1x70xf32]) <- (-1x3x70xf32, -1x1x70xf32)
        combine_38 = [concat_10, unsqueeze_8]

        # pd_op.concat: (-1x4x70xf32) <- ([-1x3x70xf32, -1x1x70xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_38, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_3 = paddle._C_ops.argmax(add__46, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_4 = paddle._C_ops.one_hot(argmax_3 % paddle.cast(constant_29, argmax_3.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_35 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_36 = paddle.matmul(multiply__16, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__47 = paddle._C_ops.add_(matmul_36, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__20, unsqueeze__21 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__47, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__48 = paddle._C_ops.add_(matmul_35, unsqueeze__20)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__8 = paddle._C_ops.tanh_(add__48)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_37 = paddle.matmul(tanh__8, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__4 = paddle._C_ops.softmax_(matmul_37, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_11 = paddle._C_ops.transpose(softmax__4, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_38 = paddle.matmul(transpose_11, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__10, squeeze__11 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_38, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_39 = [squeeze__10, one_hot_4]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_39, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_39 = paddle.matmul(concat_13, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__49 = paddle._C_ops.add_(matmul_39, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_40 = paddle.matmul(multiply__16, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__50 = paddle._C_ops.add_(add__49, matmul_40)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__51 = paddle._C_ops.add_(add__50, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(add__51, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_27 = split_with_num_4[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__14 = paddle._C_ops.sigmoid_(slice_27)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_28 = split_with_num_4[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__15 = paddle._C_ops.sigmoid_(slice_28)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_29 = split_with_num_4[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__16 = paddle._C_ops.sigmoid_(slice_29)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__17 = paddle._C_ops.multiply_(sigmoid__15, add__45)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_30 = split_with_num_4[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__9 = paddle._C_ops.tanh_(slice_30)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__18 = paddle._C_ops.multiply_(sigmoid__14, tanh__9)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__52 = paddle._C_ops.add_(multiply__17, multiply__18)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_4 = paddle._C_ops.tanh(add__52)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__19 = paddle._C_ops.multiply_(sigmoid__16, tanh_4)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_41 = paddle.matmul(multiply__19, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__53 = paddle._C_ops.add_(matmul_41, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_10, unsqueeze_11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__53, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x4x70xf32, -1x1x70xf32]) <- (-1x4x70xf32, -1x1x70xf32)
        combine_40 = [concat_12, unsqueeze_10]

        # pd_op.concat: (-1x5x70xf32) <- ([-1x4x70xf32, -1x1x70xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_40, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_4 = paddle._C_ops.argmax(add__53, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_5 = paddle._C_ops.one_hot(argmax_4 % paddle.cast(constant_29, argmax_4.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_42 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_43 = paddle.matmul(multiply__19, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__54 = paddle._C_ops.add_(matmul_43, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__22, unsqueeze__23 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__54, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__55 = paddle._C_ops.add_(matmul_42, unsqueeze__22)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__10 = paddle._C_ops.tanh_(add__55)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_44 = paddle.matmul(tanh__10, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__5 = paddle._C_ops.softmax_(matmul_44, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_12 = paddle._C_ops.transpose(softmax__5, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_45 = paddle.matmul(transpose_12, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__12, squeeze__13 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_45, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_41 = [squeeze__12, one_hot_5]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_41, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_46 = paddle.matmul(concat_15, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__56 = paddle._C_ops.add_(matmul_46, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_47 = paddle.matmul(multiply__19, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__57 = paddle._C_ops.add_(add__56, matmul_47)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__58 = paddle._C_ops.add_(add__57, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(add__58, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_31 = split_with_num_5[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__17 = paddle._C_ops.sigmoid_(slice_31)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_32 = split_with_num_5[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__18 = paddle._C_ops.sigmoid_(slice_32)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_33 = split_with_num_5[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__19 = paddle._C_ops.sigmoid_(slice_33)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__20 = paddle._C_ops.multiply_(sigmoid__18, add__52)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_34 = split_with_num_5[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__11 = paddle._C_ops.tanh_(slice_34)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__21 = paddle._C_ops.multiply_(sigmoid__17, tanh__11)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__59 = paddle._C_ops.add_(multiply__20, multiply__21)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_5 = paddle._C_ops.tanh(add__59)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__22 = paddle._C_ops.multiply_(sigmoid__19, tanh_5)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_48 = paddle.matmul(multiply__22, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__60 = paddle._C_ops.add_(matmul_48, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_12, unsqueeze_13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__60, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x5x70xf32, -1x1x70xf32]) <- (-1x5x70xf32, -1x1x70xf32)
        combine_42 = [concat_14, unsqueeze_12]

        # pd_op.concat: (-1x6x70xf32) <- ([-1x5x70xf32, -1x1x70xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_42, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_5 = paddle._C_ops.argmax(add__60, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_6 = paddle._C_ops.one_hot(argmax_5 % paddle.cast(constant_29, argmax_5.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_49 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_50 = paddle.matmul(multiply__22, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__61 = paddle._C_ops.add_(matmul_50, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__24, unsqueeze__25 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__61, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__62 = paddle._C_ops.add_(matmul_49, unsqueeze__24)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__12 = paddle._C_ops.tanh_(add__62)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_51 = paddle.matmul(tanh__12, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__6 = paddle._C_ops.softmax_(matmul_51, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_13 = paddle._C_ops.transpose(softmax__6, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_52 = paddle.matmul(transpose_13, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__14, squeeze__15 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_52, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_43 = [squeeze__14, one_hot_6]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_43, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_53 = paddle.matmul(concat_17, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__63 = paddle._C_ops.add_(matmul_53, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_54 = paddle.matmul(multiply__22, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__64 = paddle._C_ops.add_(add__63, matmul_54)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__65 = paddle._C_ops.add_(add__64, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(add__65, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_35 = split_with_num_6[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__20 = paddle._C_ops.sigmoid_(slice_35)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_36 = split_with_num_6[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__21 = paddle._C_ops.sigmoid_(slice_36)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_37 = split_with_num_6[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__22 = paddle._C_ops.sigmoid_(slice_37)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__23 = paddle._C_ops.multiply_(sigmoid__21, add__59)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_38 = split_with_num_6[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__13 = paddle._C_ops.tanh_(slice_38)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__24 = paddle._C_ops.multiply_(sigmoid__20, tanh__13)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__66 = paddle._C_ops.add_(multiply__23, multiply__24)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_6 = paddle._C_ops.tanh(add__66)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__25 = paddle._C_ops.multiply_(sigmoid__22, tanh_6)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_55 = paddle.matmul(multiply__25, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__67 = paddle._C_ops.add_(matmul_55, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_14, unsqueeze_15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__67, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x6x70xf32, -1x1x70xf32]) <- (-1x6x70xf32, -1x1x70xf32)
        combine_44 = [concat_16, unsqueeze_14]

        # pd_op.concat: (-1x7x70xf32) <- ([-1x6x70xf32, -1x1x70xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_44, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_6 = paddle._C_ops.argmax(add__67, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_7 = paddle._C_ops.one_hot(argmax_6 % paddle.cast(constant_29, argmax_6.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_56 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_57 = paddle.matmul(multiply__25, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__68 = paddle._C_ops.add_(matmul_57, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__26, unsqueeze__27 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__68, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__69 = paddle._C_ops.add_(matmul_56, unsqueeze__26)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__14 = paddle._C_ops.tanh_(add__69)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_58 = paddle.matmul(tanh__14, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__7 = paddle._C_ops.softmax_(matmul_58, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_14 = paddle._C_ops.transpose(softmax__7, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_59 = paddle.matmul(transpose_14, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__16, squeeze__17 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_59, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_45 = [squeeze__16, one_hot_7]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_45, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_60 = paddle.matmul(concat_19, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__70 = paddle._C_ops.add_(matmul_60, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_61 = paddle.matmul(multiply__25, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__71 = paddle._C_ops.add_(add__70, matmul_61)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__72 = paddle._C_ops.add_(add__71, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(add__72, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_39 = split_with_num_7[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__23 = paddle._C_ops.sigmoid_(slice_39)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_40 = split_with_num_7[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__24 = paddle._C_ops.sigmoid_(slice_40)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_41 = split_with_num_7[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__25 = paddle._C_ops.sigmoid_(slice_41)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__26 = paddle._C_ops.multiply_(sigmoid__24, add__66)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_42 = split_with_num_7[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__15 = paddle._C_ops.tanh_(slice_42)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__27 = paddle._C_ops.multiply_(sigmoid__23, tanh__15)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__73 = paddle._C_ops.add_(multiply__26, multiply__27)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_7 = paddle._C_ops.tanh(add__73)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__28 = paddle._C_ops.multiply_(sigmoid__25, tanh_7)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_62 = paddle.matmul(multiply__28, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__74 = paddle._C_ops.add_(matmul_62, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_16, unsqueeze_17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__74, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x7x70xf32, -1x1x70xf32]) <- (-1x7x70xf32, -1x1x70xf32)
        combine_46 = [concat_18, unsqueeze_16]

        # pd_op.concat: (-1x8x70xf32) <- ([-1x7x70xf32, -1x1x70xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_46, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_7 = paddle._C_ops.argmax(add__74, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_8 = paddle._C_ops.one_hot(argmax_7 % paddle.cast(constant_29, argmax_7.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_63 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_64 = paddle.matmul(multiply__28, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__75 = paddle._C_ops.add_(matmul_64, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__28, unsqueeze__29 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__75, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__76 = paddle._C_ops.add_(matmul_63, unsqueeze__28)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__16 = paddle._C_ops.tanh_(add__76)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_65 = paddle.matmul(tanh__16, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__8 = paddle._C_ops.softmax_(matmul_65, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_15 = paddle._C_ops.transpose(softmax__8, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_66 = paddle.matmul(transpose_15, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__18, squeeze__19 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_66, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_47 = [squeeze__18, one_hot_8]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_47, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_67 = paddle.matmul(concat_21, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__77 = paddle._C_ops.add_(matmul_67, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_68 = paddle.matmul(multiply__28, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__78 = paddle._C_ops.add_(add__77, matmul_68)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__79 = paddle._C_ops.add_(add__78, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(add__79, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_43 = split_with_num_8[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__26 = paddle._C_ops.sigmoid_(slice_43)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_44 = split_with_num_8[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__27 = paddle._C_ops.sigmoid_(slice_44)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_45 = split_with_num_8[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__28 = paddle._C_ops.sigmoid_(slice_45)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__29 = paddle._C_ops.multiply_(sigmoid__27, add__73)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_46 = split_with_num_8[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__17 = paddle._C_ops.tanh_(slice_46)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__30 = paddle._C_ops.multiply_(sigmoid__26, tanh__17)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__80 = paddle._C_ops.add_(multiply__29, multiply__30)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_8 = paddle._C_ops.tanh(add__80)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__31 = paddle._C_ops.multiply_(sigmoid__28, tanh_8)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_69 = paddle.matmul(multiply__31, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__81 = paddle._C_ops.add_(matmul_69, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_18, unsqueeze_19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__81, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x8x70xf32, -1x1x70xf32]) <- (-1x8x70xf32, -1x1x70xf32)
        combine_48 = [concat_20, unsqueeze_18]

        # pd_op.concat: (-1x9x70xf32) <- ([-1x8x70xf32, -1x1x70xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_48, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_8 = paddle._C_ops.argmax(add__81, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_9 = paddle._C_ops.one_hot(argmax_8 % paddle.cast(constant_29, argmax_8.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_70 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_71 = paddle.matmul(multiply__31, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__82 = paddle._C_ops.add_(matmul_71, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__30, unsqueeze__31 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__82, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__83 = paddle._C_ops.add_(matmul_70, unsqueeze__30)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__18 = paddle._C_ops.tanh_(add__83)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_72 = paddle.matmul(tanh__18, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__9 = paddle._C_ops.softmax_(matmul_72, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_16 = paddle._C_ops.transpose(softmax__9, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_73 = paddle.matmul(transpose_16, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__20, squeeze__21 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_73, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_49 = [squeeze__20, one_hot_9]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_49, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_74 = paddle.matmul(concat_23, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__84 = paddle._C_ops.add_(matmul_74, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_75 = paddle.matmul(multiply__31, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__85 = paddle._C_ops.add_(add__84, matmul_75)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__86 = paddle._C_ops.add_(add__85, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(add__86, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_47 = split_with_num_9[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__29 = paddle._C_ops.sigmoid_(slice_47)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_48 = split_with_num_9[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__30 = paddle._C_ops.sigmoid_(slice_48)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_49 = split_with_num_9[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__31 = paddle._C_ops.sigmoid_(slice_49)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__32 = paddle._C_ops.multiply_(sigmoid__30, add__80)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_50 = split_with_num_9[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__19 = paddle._C_ops.tanh_(slice_50)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__33 = paddle._C_ops.multiply_(sigmoid__29, tanh__19)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__87 = paddle._C_ops.add_(multiply__32, multiply__33)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_9 = paddle._C_ops.tanh(add__87)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__34 = paddle._C_ops.multiply_(sigmoid__31, tanh_9)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_76 = paddle.matmul(multiply__34, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__88 = paddle._C_ops.add_(matmul_76, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_20, unsqueeze_21 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__88, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x9x70xf32, -1x1x70xf32]) <- (-1x9x70xf32, -1x1x70xf32)
        combine_50 = [concat_22, unsqueeze_20]

        # pd_op.concat: (-1x10x70xf32) <- ([-1x9x70xf32, -1x1x70xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_50, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_9 = paddle._C_ops.argmax(add__88, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_10 = paddle._C_ops.one_hot(argmax_9 % paddle.cast(constant_29, argmax_9.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_77 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_78 = paddle.matmul(multiply__34, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__89 = paddle._C_ops.add_(matmul_78, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__32, unsqueeze__33 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__89, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__90 = paddle._C_ops.add_(matmul_77, unsqueeze__32)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__20 = paddle._C_ops.tanh_(add__90)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_79 = paddle.matmul(tanh__20, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__10 = paddle._C_ops.softmax_(matmul_79, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_17 = paddle._C_ops.transpose(softmax__10, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_80 = paddle.matmul(transpose_17, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__22, squeeze__23 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_80, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_51 = [squeeze__22, one_hot_10]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_51, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_81 = paddle.matmul(concat_25, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__91 = paddle._C_ops.add_(matmul_81, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_82 = paddle.matmul(multiply__34, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__92 = paddle._C_ops.add_(add__91, matmul_82)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__93 = paddle._C_ops.add_(add__92, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(add__93, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_51 = split_with_num_10[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__32 = paddle._C_ops.sigmoid_(slice_51)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_52 = split_with_num_10[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__33 = paddle._C_ops.sigmoid_(slice_52)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_53 = split_with_num_10[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__34 = paddle._C_ops.sigmoid_(slice_53)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__35 = paddle._C_ops.multiply_(sigmoid__33, add__87)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_54 = split_with_num_10[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__21 = paddle._C_ops.tanh_(slice_54)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__36 = paddle._C_ops.multiply_(sigmoid__32, tanh__21)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__94 = paddle._C_ops.add_(multiply__35, multiply__36)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_10 = paddle._C_ops.tanh(add__94)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__37 = paddle._C_ops.multiply_(sigmoid__34, tanh_10)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_83 = paddle.matmul(multiply__37, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__95 = paddle._C_ops.add_(matmul_83, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_22, unsqueeze_23 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__95, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x10x70xf32, -1x1x70xf32]) <- (-1x10x70xf32, -1x1x70xf32)
        combine_52 = [concat_24, unsqueeze_22]

        # pd_op.concat: (-1x11x70xf32) <- ([-1x10x70xf32, -1x1x70xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_52, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_10 = paddle._C_ops.argmax(add__95, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_11 = paddle._C_ops.one_hot(argmax_10 % paddle.cast(constant_29, argmax_10.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_84 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_85 = paddle.matmul(multiply__37, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__96 = paddle._C_ops.add_(matmul_85, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__34, unsqueeze__35 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__96, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__97 = paddle._C_ops.add_(matmul_84, unsqueeze__34)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__22 = paddle._C_ops.tanh_(add__97)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_86 = paddle.matmul(tanh__22, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__11 = paddle._C_ops.softmax_(matmul_86, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_18 = paddle._C_ops.transpose(softmax__11, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_87 = paddle.matmul(transpose_18, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__24, squeeze__25 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_87, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_53 = [squeeze__24, one_hot_11]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_53, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_88 = paddle.matmul(concat_27, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__98 = paddle._C_ops.add_(matmul_88, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_89 = paddle.matmul(multiply__37, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__99 = paddle._C_ops.add_(add__98, matmul_89)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__100 = paddle._C_ops.add_(add__99, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(add__100, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_55 = split_with_num_11[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__35 = paddle._C_ops.sigmoid_(slice_55)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_56 = split_with_num_11[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__36 = paddle._C_ops.sigmoid_(slice_56)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_57 = split_with_num_11[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__37 = paddle._C_ops.sigmoid_(slice_57)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__38 = paddle._C_ops.multiply_(sigmoid__36, add__94)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_58 = split_with_num_11[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__23 = paddle._C_ops.tanh_(slice_58)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__39 = paddle._C_ops.multiply_(sigmoid__35, tanh__23)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__101 = paddle._C_ops.add_(multiply__38, multiply__39)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_11 = paddle._C_ops.tanh(add__101)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__40 = paddle._C_ops.multiply_(sigmoid__37, tanh_11)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_90 = paddle.matmul(multiply__40, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__102 = paddle._C_ops.add_(matmul_90, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_24, unsqueeze_25 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__102, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x11x70xf32, -1x1x70xf32]) <- (-1x11x70xf32, -1x1x70xf32)
        combine_54 = [concat_26, unsqueeze_24]

        # pd_op.concat: (-1x12x70xf32) <- ([-1x11x70xf32, -1x1x70xf32], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_54, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_11 = paddle._C_ops.argmax(add__102, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_12 = paddle._C_ops.one_hot(argmax_11 % paddle.cast(constant_29, argmax_11.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_91 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_92 = paddle.matmul(multiply__40, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__103 = paddle._C_ops.add_(matmul_92, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__36, unsqueeze__37 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__103, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__104 = paddle._C_ops.add_(matmul_91, unsqueeze__36)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__24 = paddle._C_ops.tanh_(add__104)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_93 = paddle.matmul(tanh__24, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__12 = paddle._C_ops.softmax_(matmul_93, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_19 = paddle._C_ops.transpose(softmax__12, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_94 = paddle.matmul(transpose_19, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__26, squeeze__27 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_94, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_55 = [squeeze__26, one_hot_12]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_55, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_95 = paddle.matmul(concat_29, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__105 = paddle._C_ops.add_(matmul_95, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_96 = paddle.matmul(multiply__40, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__106 = paddle._C_ops.add_(add__105, matmul_96)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__107 = paddle._C_ops.add_(add__106, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_12 = paddle._C_ops.split_with_num(add__107, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_59 = split_with_num_12[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__38 = paddle._C_ops.sigmoid_(slice_59)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_60 = split_with_num_12[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__39 = paddle._C_ops.sigmoid_(slice_60)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_61 = split_with_num_12[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__40 = paddle._C_ops.sigmoid_(slice_61)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__41 = paddle._C_ops.multiply_(sigmoid__39, add__101)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_62 = split_with_num_12[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__25 = paddle._C_ops.tanh_(slice_62)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__42 = paddle._C_ops.multiply_(sigmoid__38, tanh__25)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__108 = paddle._C_ops.add_(multiply__41, multiply__42)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_12 = paddle._C_ops.tanh(add__108)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__43 = paddle._C_ops.multiply_(sigmoid__40, tanh_12)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_97 = paddle.matmul(multiply__43, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__109 = paddle._C_ops.add_(matmul_97, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_26, unsqueeze_27 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__109, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x12x70xf32, -1x1x70xf32]) <- (-1x12x70xf32, -1x1x70xf32)
        combine_56 = [concat_28, unsqueeze_26]

        # pd_op.concat: (-1x13x70xf32) <- ([-1x12x70xf32, -1x1x70xf32], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_56, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_12 = paddle._C_ops.argmax(add__109, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_13 = paddle._C_ops.one_hot(argmax_12 % paddle.cast(constant_29, argmax_12.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_98 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_99 = paddle.matmul(multiply__43, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__110 = paddle._C_ops.add_(matmul_99, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__38, unsqueeze__39 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__110, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__111 = paddle._C_ops.add_(matmul_98, unsqueeze__38)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__26 = paddle._C_ops.tanh_(add__111)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_100 = paddle.matmul(tanh__26, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__13 = paddle._C_ops.softmax_(matmul_100, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_20 = paddle._C_ops.transpose(softmax__13, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_101 = paddle.matmul(transpose_20, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__28, squeeze__29 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_101, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_57 = [squeeze__28, one_hot_13]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_57, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_102 = paddle.matmul(concat_31, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__112 = paddle._C_ops.add_(matmul_102, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_103 = paddle.matmul(multiply__43, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__113 = paddle._C_ops.add_(add__112, matmul_103)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__114 = paddle._C_ops.add_(add__113, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_13 = paddle._C_ops.split_with_num(add__114, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_63 = split_with_num_13[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__41 = paddle._C_ops.sigmoid_(slice_63)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_64 = split_with_num_13[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__42 = paddle._C_ops.sigmoid_(slice_64)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_65 = split_with_num_13[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__43 = paddle._C_ops.sigmoid_(slice_65)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__44 = paddle._C_ops.multiply_(sigmoid__42, add__108)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_66 = split_with_num_13[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__27 = paddle._C_ops.tanh_(slice_66)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__45 = paddle._C_ops.multiply_(sigmoid__41, tanh__27)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__115 = paddle._C_ops.add_(multiply__44, multiply__45)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_13 = paddle._C_ops.tanh(add__115)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__46 = paddle._C_ops.multiply_(sigmoid__43, tanh_13)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_104 = paddle.matmul(multiply__46, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__116 = paddle._C_ops.add_(matmul_104, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_28, unsqueeze_29 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__116, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x13x70xf32, -1x1x70xf32]) <- (-1x13x70xf32, -1x1x70xf32)
        combine_58 = [concat_30, unsqueeze_28]

        # pd_op.concat: (-1x14x70xf32) <- ([-1x13x70xf32, -1x1x70xf32], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_58, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_13 = paddle._C_ops.argmax(add__116, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_14 = paddle._C_ops.one_hot(argmax_13 % paddle.cast(constant_29, argmax_13.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_105 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_106 = paddle.matmul(multiply__46, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__117 = paddle._C_ops.add_(matmul_106, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__40, unsqueeze__41 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__117, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__118 = paddle._C_ops.add_(matmul_105, unsqueeze__40)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__28 = paddle._C_ops.tanh_(add__118)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_107 = paddle.matmul(tanh__28, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__14 = paddle._C_ops.softmax_(matmul_107, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_21 = paddle._C_ops.transpose(softmax__14, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_108 = paddle.matmul(transpose_21, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__30, squeeze__31 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_108, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_59 = [squeeze__30, one_hot_14]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_59, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_109 = paddle.matmul(concat_33, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__119 = paddle._C_ops.add_(matmul_109, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_110 = paddle.matmul(multiply__46, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__120 = paddle._C_ops.add_(add__119, matmul_110)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__121 = paddle._C_ops.add_(add__120, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_14 = paddle._C_ops.split_with_num(add__121, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_67 = split_with_num_14[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__44 = paddle._C_ops.sigmoid_(slice_67)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_68 = split_with_num_14[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__45 = paddle._C_ops.sigmoid_(slice_68)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_69 = split_with_num_14[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__46 = paddle._C_ops.sigmoid_(slice_69)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__47 = paddle._C_ops.multiply_(sigmoid__45, add__115)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_70 = split_with_num_14[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__29 = paddle._C_ops.tanh_(slice_70)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__48 = paddle._C_ops.multiply_(sigmoid__44, tanh__29)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__122 = paddle._C_ops.add_(multiply__47, multiply__48)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_14 = paddle._C_ops.tanh(add__122)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__49 = paddle._C_ops.multiply_(sigmoid__46, tanh_14)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_111 = paddle.matmul(multiply__49, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__123 = paddle._C_ops.add_(matmul_111, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_30, unsqueeze_31 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__123, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x14x70xf32, -1x1x70xf32]) <- (-1x14x70xf32, -1x1x70xf32)
        combine_60 = [concat_32, unsqueeze_30]

        # pd_op.concat: (-1x15x70xf32) <- ([-1x14x70xf32, -1x1x70xf32], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_60, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_14 = paddle._C_ops.argmax(add__123, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_15 = paddle._C_ops.one_hot(argmax_14 % paddle.cast(constant_29, argmax_14.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_112 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_113 = paddle.matmul(multiply__49, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__124 = paddle._C_ops.add_(matmul_113, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__42, unsqueeze__43 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__124, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__125 = paddle._C_ops.add_(matmul_112, unsqueeze__42)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__30 = paddle._C_ops.tanh_(add__125)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_114 = paddle.matmul(tanh__30, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__15 = paddle._C_ops.softmax_(matmul_114, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_22 = paddle._C_ops.transpose(softmax__15, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_115 = paddle.matmul(transpose_22, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__32, squeeze__33 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_115, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_61 = [squeeze__32, one_hot_15]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_61, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_116 = paddle.matmul(concat_35, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__126 = paddle._C_ops.add_(matmul_116, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_117 = paddle.matmul(multiply__49, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__127 = paddle._C_ops.add_(add__126, matmul_117)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__128 = paddle._C_ops.add_(add__127, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_15 = paddle._C_ops.split_with_num(add__128, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_71 = split_with_num_15[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__47 = paddle._C_ops.sigmoid_(slice_71)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_72 = split_with_num_15[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__48 = paddle._C_ops.sigmoid_(slice_72)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_73 = split_with_num_15[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__49 = paddle._C_ops.sigmoid_(slice_73)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__50 = paddle._C_ops.multiply_(sigmoid__48, add__122)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_74 = split_with_num_15[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__31 = paddle._C_ops.tanh_(slice_74)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__51 = paddle._C_ops.multiply_(sigmoid__47, tanh__31)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__129 = paddle._C_ops.add_(multiply__50, multiply__51)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_15 = paddle._C_ops.tanh(add__129)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__52 = paddle._C_ops.multiply_(sigmoid__49, tanh_15)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_118 = paddle.matmul(multiply__52, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__130 = paddle._C_ops.add_(matmul_118, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_32, unsqueeze_33 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__130, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x15x70xf32, -1x1x70xf32]) <- (-1x15x70xf32, -1x1x70xf32)
        combine_62 = [concat_34, unsqueeze_32]

        # pd_op.concat: (-1x16x70xf32) <- ([-1x15x70xf32, -1x1x70xf32], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_62, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_15 = paddle._C_ops.argmax(add__130, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_16 = paddle._C_ops.one_hot(argmax_15 % paddle.cast(constant_29, argmax_15.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_119 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_120 = paddle.matmul(multiply__52, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__131 = paddle._C_ops.add_(matmul_120, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__44, unsqueeze__45 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__131, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__132 = paddle._C_ops.add_(matmul_119, unsqueeze__44)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__32 = paddle._C_ops.tanh_(add__132)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_121 = paddle.matmul(tanh__32, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__16 = paddle._C_ops.softmax_(matmul_121, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_23 = paddle._C_ops.transpose(softmax__16, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_122 = paddle.matmul(transpose_23, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__34, squeeze__35 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_122, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_63 = [squeeze__34, one_hot_16]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_63, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_123 = paddle.matmul(concat_37, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__133 = paddle._C_ops.add_(matmul_123, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_124 = paddle.matmul(multiply__52, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__134 = paddle._C_ops.add_(add__133, matmul_124)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__135 = paddle._C_ops.add_(add__134, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_16 = paddle._C_ops.split_with_num(add__135, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_75 = split_with_num_16[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__50 = paddle._C_ops.sigmoid_(slice_75)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_76 = split_with_num_16[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__51 = paddle._C_ops.sigmoid_(slice_76)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_77 = split_with_num_16[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__52 = paddle._C_ops.sigmoid_(slice_77)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__53 = paddle._C_ops.multiply_(sigmoid__51, add__129)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_78 = split_with_num_16[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__33 = paddle._C_ops.tanh_(slice_78)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__54 = paddle._C_ops.multiply_(sigmoid__50, tanh__33)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__136 = paddle._C_ops.add_(multiply__53, multiply__54)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_16 = paddle._C_ops.tanh(add__136)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__55 = paddle._C_ops.multiply_(sigmoid__52, tanh_16)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_125 = paddle.matmul(multiply__55, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__137 = paddle._C_ops.add_(matmul_125, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_34, unsqueeze_35 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__137, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x16x70xf32, -1x1x70xf32]) <- (-1x16x70xf32, -1x1x70xf32)
        combine_64 = [concat_36, unsqueeze_34]

        # pd_op.concat: (-1x17x70xf32) <- ([-1x16x70xf32, -1x1x70xf32], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_64, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_16 = paddle._C_ops.argmax(add__137, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_17 = paddle._C_ops.one_hot(argmax_16 % paddle.cast(constant_29, argmax_16.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_126 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_127 = paddle.matmul(multiply__55, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__138 = paddle._C_ops.add_(matmul_127, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__46, unsqueeze__47 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__138, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__139 = paddle._C_ops.add_(matmul_126, unsqueeze__46)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__34 = paddle._C_ops.tanh_(add__139)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_128 = paddle.matmul(tanh__34, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__17 = paddle._C_ops.softmax_(matmul_128, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_24 = paddle._C_ops.transpose(softmax__17, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_129 = paddle.matmul(transpose_24, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__36, squeeze__37 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_129, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_65 = [squeeze__36, one_hot_17]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_39 = paddle._C_ops.concat(combine_65, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_130 = paddle.matmul(concat_39, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__140 = paddle._C_ops.add_(matmul_130, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_131 = paddle.matmul(multiply__55, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__141 = paddle._C_ops.add_(add__140, matmul_131)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__142 = paddle._C_ops.add_(add__141, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_17 = paddle._C_ops.split_with_num(add__142, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_79 = split_with_num_17[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__53 = paddle._C_ops.sigmoid_(slice_79)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_80 = split_with_num_17[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__54 = paddle._C_ops.sigmoid_(slice_80)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_81 = split_with_num_17[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__55 = paddle._C_ops.sigmoid_(slice_81)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__56 = paddle._C_ops.multiply_(sigmoid__54, add__136)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_82 = split_with_num_17[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__35 = paddle._C_ops.tanh_(slice_82)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__57 = paddle._C_ops.multiply_(sigmoid__53, tanh__35)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__143 = paddle._C_ops.add_(multiply__56, multiply__57)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_17 = paddle._C_ops.tanh(add__143)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__58 = paddle._C_ops.multiply_(sigmoid__55, tanh_17)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_132 = paddle.matmul(multiply__58, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__144 = paddle._C_ops.add_(matmul_132, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_36, unsqueeze_37 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__144, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x17x70xf32, -1x1x70xf32]) <- (-1x17x70xf32, -1x1x70xf32)
        combine_66 = [concat_38, unsqueeze_36]

        # pd_op.concat: (-1x18x70xf32) <- ([-1x17x70xf32, -1x1x70xf32], 1xi32)
        concat_40 = paddle._C_ops.concat(combine_66, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_17 = paddle._C_ops.argmax(add__144, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_18 = paddle._C_ops.one_hot(argmax_17 % paddle.cast(constant_29, argmax_17.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_133 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_134 = paddle.matmul(multiply__58, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__145 = paddle._C_ops.add_(matmul_134, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__48, unsqueeze__49 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__145, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__146 = paddle._C_ops.add_(matmul_133, unsqueeze__48)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__36 = paddle._C_ops.tanh_(add__146)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_135 = paddle.matmul(tanh__36, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__18 = paddle._C_ops.softmax_(matmul_135, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_25 = paddle._C_ops.transpose(softmax__18, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_136 = paddle.matmul(transpose_25, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__38, squeeze__39 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_136, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_67 = [squeeze__38, one_hot_18]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_41 = paddle._C_ops.concat(combine_67, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_137 = paddle.matmul(concat_41, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__147 = paddle._C_ops.add_(matmul_137, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_138 = paddle.matmul(multiply__58, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__148 = paddle._C_ops.add_(add__147, matmul_138)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__149 = paddle._C_ops.add_(add__148, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_18 = paddle._C_ops.split_with_num(add__149, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_83 = split_with_num_18[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__56 = paddle._C_ops.sigmoid_(slice_83)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_84 = split_with_num_18[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__57 = paddle._C_ops.sigmoid_(slice_84)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_85 = split_with_num_18[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__58 = paddle._C_ops.sigmoid_(slice_85)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__59 = paddle._C_ops.multiply_(sigmoid__57, add__143)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_86 = split_with_num_18[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__37 = paddle._C_ops.tanh_(slice_86)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__60 = paddle._C_ops.multiply_(sigmoid__56, tanh__37)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__150 = paddle._C_ops.add_(multiply__59, multiply__60)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_18 = paddle._C_ops.tanh(add__150)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__61 = paddle._C_ops.multiply_(sigmoid__58, tanh_18)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_139 = paddle.matmul(multiply__61, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__151 = paddle._C_ops.add_(matmul_139, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_38, unsqueeze_39 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__151, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x18x70xf32, -1x1x70xf32]) <- (-1x18x70xf32, -1x1x70xf32)
        combine_68 = [concat_40, unsqueeze_38]

        # pd_op.concat: (-1x19x70xf32) <- ([-1x18x70xf32, -1x1x70xf32], 1xi32)
        concat_42 = paddle._C_ops.concat(combine_68, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_18 = paddle._C_ops.argmax(add__151, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_19 = paddle._C_ops.one_hot(argmax_18 % paddle.cast(constant_29, argmax_18.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_140 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_141 = paddle.matmul(multiply__61, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__152 = paddle._C_ops.add_(matmul_141, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__50, unsqueeze__51 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__152, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__153 = paddle._C_ops.add_(matmul_140, unsqueeze__50)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__38 = paddle._C_ops.tanh_(add__153)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_142 = paddle.matmul(tanh__38, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__19 = paddle._C_ops.softmax_(matmul_142, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_26 = paddle._C_ops.transpose(softmax__19, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_143 = paddle.matmul(transpose_26, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__40, squeeze__41 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_143, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_69 = [squeeze__40, one_hot_19]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_43 = paddle._C_ops.concat(combine_69, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_144 = paddle.matmul(concat_43, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__154 = paddle._C_ops.add_(matmul_144, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_145 = paddle.matmul(multiply__61, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__155 = paddle._C_ops.add_(add__154, matmul_145)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__156 = paddle._C_ops.add_(add__155, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_19 = paddle._C_ops.split_with_num(add__156, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_87 = split_with_num_19[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__59 = paddle._C_ops.sigmoid_(slice_87)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_88 = split_with_num_19[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__60 = paddle._C_ops.sigmoid_(slice_88)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_89 = split_with_num_19[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__61 = paddle._C_ops.sigmoid_(slice_89)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__62 = paddle._C_ops.multiply_(sigmoid__60, add__150)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_90 = split_with_num_19[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__39 = paddle._C_ops.tanh_(slice_90)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__63 = paddle._C_ops.multiply_(sigmoid__59, tanh__39)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__157 = paddle._C_ops.add_(multiply__62, multiply__63)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_19 = paddle._C_ops.tanh(add__157)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__64 = paddle._C_ops.multiply_(sigmoid__61, tanh_19)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_146 = paddle.matmul(multiply__64, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__158 = paddle._C_ops.add_(matmul_146, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_40, unsqueeze_41 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__158, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x19x70xf32, -1x1x70xf32]) <- (-1x19x70xf32, -1x1x70xf32)
        combine_70 = [concat_42, unsqueeze_40]

        # pd_op.concat: (-1x20x70xf32) <- ([-1x19x70xf32, -1x1x70xf32], 1xi32)
        concat_44 = paddle._C_ops.concat(combine_70, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_19 = paddle._C_ops.argmax(add__158, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_20 = paddle._C_ops.one_hot(argmax_19 % paddle.cast(constant_29, argmax_19.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_147 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_148 = paddle.matmul(multiply__64, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__159 = paddle._C_ops.add_(matmul_148, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__52, unsqueeze__53 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__159, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__160 = paddle._C_ops.add_(matmul_147, unsqueeze__52)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__40 = paddle._C_ops.tanh_(add__160)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_149 = paddle.matmul(tanh__40, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__20 = paddle._C_ops.softmax_(matmul_149, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_27 = paddle._C_ops.transpose(softmax__20, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_150 = paddle.matmul(transpose_27, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__42, squeeze__43 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_150, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_71 = [squeeze__42, one_hot_20]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_45 = paddle._C_ops.concat(combine_71, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_151 = paddle.matmul(concat_45, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__161 = paddle._C_ops.add_(matmul_151, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_152 = paddle.matmul(multiply__64, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__162 = paddle._C_ops.add_(add__161, matmul_152)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__163 = paddle._C_ops.add_(add__162, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_20 = paddle._C_ops.split_with_num(add__163, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_91 = split_with_num_20[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__62 = paddle._C_ops.sigmoid_(slice_91)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_92 = split_with_num_20[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__63 = paddle._C_ops.sigmoid_(slice_92)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_93 = split_with_num_20[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__64 = paddle._C_ops.sigmoid_(slice_93)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__65 = paddle._C_ops.multiply_(sigmoid__63, add__157)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_94 = split_with_num_20[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__41 = paddle._C_ops.tanh_(slice_94)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__66 = paddle._C_ops.multiply_(sigmoid__62, tanh__41)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__164 = paddle._C_ops.add_(multiply__65, multiply__66)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_20 = paddle._C_ops.tanh(add__164)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__67 = paddle._C_ops.multiply_(sigmoid__64, tanh_20)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_153 = paddle.matmul(multiply__67, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__165 = paddle._C_ops.add_(matmul_153, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_42, unsqueeze_43 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__165, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x20x70xf32, -1x1x70xf32]) <- (-1x20x70xf32, -1x1x70xf32)
        combine_72 = [concat_44, unsqueeze_42]

        # pd_op.concat: (-1x21x70xf32) <- ([-1x20x70xf32, -1x1x70xf32], 1xi32)
        concat_46 = paddle._C_ops.concat(combine_72, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_20 = paddle._C_ops.argmax(add__165, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_21 = paddle._C_ops.one_hot(argmax_20 % paddle.cast(constant_29, argmax_20.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_154 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_155 = paddle.matmul(multiply__67, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__166 = paddle._C_ops.add_(matmul_155, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__54, unsqueeze__55 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__166, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__167 = paddle._C_ops.add_(matmul_154, unsqueeze__54)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__42 = paddle._C_ops.tanh_(add__167)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_156 = paddle.matmul(tanh__42, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__21 = paddle._C_ops.softmax_(matmul_156, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_28 = paddle._C_ops.transpose(softmax__21, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_157 = paddle.matmul(transpose_28, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__44, squeeze__45 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_157, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_73 = [squeeze__44, one_hot_21]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_47 = paddle._C_ops.concat(combine_73, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_158 = paddle.matmul(concat_47, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__168 = paddle._C_ops.add_(matmul_158, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_159 = paddle.matmul(multiply__67, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__169 = paddle._C_ops.add_(add__168, matmul_159)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__170 = paddle._C_ops.add_(add__169, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_21 = paddle._C_ops.split_with_num(add__170, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_95 = split_with_num_21[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__65 = paddle._C_ops.sigmoid_(slice_95)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_96 = split_with_num_21[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__66 = paddle._C_ops.sigmoid_(slice_96)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_97 = split_with_num_21[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__67 = paddle._C_ops.sigmoid_(slice_97)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__68 = paddle._C_ops.multiply_(sigmoid__66, add__164)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_98 = split_with_num_21[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__43 = paddle._C_ops.tanh_(slice_98)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__69 = paddle._C_ops.multiply_(sigmoid__65, tanh__43)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__171 = paddle._C_ops.add_(multiply__68, multiply__69)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_21 = paddle._C_ops.tanh(add__171)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__70 = paddle._C_ops.multiply_(sigmoid__67, tanh_21)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_160 = paddle.matmul(multiply__70, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__172 = paddle._C_ops.add_(matmul_160, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_44, unsqueeze_45 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__172, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x21x70xf32, -1x1x70xf32]) <- (-1x21x70xf32, -1x1x70xf32)
        combine_74 = [concat_46, unsqueeze_44]

        # pd_op.concat: (-1x22x70xf32) <- ([-1x21x70xf32, -1x1x70xf32], 1xi32)
        concat_48 = paddle._C_ops.concat(combine_74, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_21 = paddle._C_ops.argmax(add__172, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_22 = paddle._C_ops.one_hot(argmax_21 % paddle.cast(constant_29, argmax_21.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_161 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_162 = paddle.matmul(multiply__70, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__173 = paddle._C_ops.add_(matmul_162, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__56, unsqueeze__57 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__173, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__174 = paddle._C_ops.add_(matmul_161, unsqueeze__56)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__44 = paddle._C_ops.tanh_(add__174)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_163 = paddle.matmul(tanh__44, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__22 = paddle._C_ops.softmax_(matmul_163, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_29 = paddle._C_ops.transpose(softmax__22, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_164 = paddle.matmul(transpose_29, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__46, squeeze__47 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_164, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_75 = [squeeze__46, one_hot_22]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_49 = paddle._C_ops.concat(combine_75, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_165 = paddle.matmul(concat_49, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__175 = paddle._C_ops.add_(matmul_165, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_166 = paddle.matmul(multiply__70, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__176 = paddle._C_ops.add_(add__175, matmul_166)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__177 = paddle._C_ops.add_(add__176, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_22 = paddle._C_ops.split_with_num(add__177, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_99 = split_with_num_22[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__68 = paddle._C_ops.sigmoid_(slice_99)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_100 = split_with_num_22[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__69 = paddle._C_ops.sigmoid_(slice_100)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_101 = split_with_num_22[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__70 = paddle._C_ops.sigmoid_(slice_101)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__71 = paddle._C_ops.multiply_(sigmoid__69, add__171)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_102 = split_with_num_22[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__45 = paddle._C_ops.tanh_(slice_102)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__72 = paddle._C_ops.multiply_(sigmoid__68, tanh__45)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__178 = paddle._C_ops.add_(multiply__71, multiply__72)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_22 = paddle._C_ops.tanh(add__178)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__73 = paddle._C_ops.multiply_(sigmoid__70, tanh_22)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_167 = paddle.matmul(multiply__73, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__179 = paddle._C_ops.add_(matmul_167, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_46, unsqueeze_47 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__179, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x22x70xf32, -1x1x70xf32]) <- (-1x22x70xf32, -1x1x70xf32)
        combine_76 = [concat_48, unsqueeze_46]

        # pd_op.concat: (-1x23x70xf32) <- ([-1x22x70xf32, -1x1x70xf32], 1xi32)
        concat_50 = paddle._C_ops.concat(combine_76, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_22 = paddle._C_ops.argmax(add__179, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_23 = paddle._C_ops.one_hot(argmax_22 % paddle.cast(constant_29, argmax_22.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_168 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_169 = paddle.matmul(multiply__73, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__180 = paddle._C_ops.add_(matmul_169, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__58, unsqueeze__59 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__180, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__181 = paddle._C_ops.add_(matmul_168, unsqueeze__58)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__46 = paddle._C_ops.tanh_(add__181)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_170 = paddle.matmul(tanh__46, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__23 = paddle._C_ops.softmax_(matmul_170, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_30 = paddle._C_ops.transpose(softmax__23, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_171 = paddle.matmul(transpose_30, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__48, squeeze__49 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_171, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_77 = [squeeze__48, one_hot_23]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_51 = paddle._C_ops.concat(combine_77, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_172 = paddle.matmul(concat_51, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__182 = paddle._C_ops.add_(matmul_172, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_173 = paddle.matmul(multiply__73, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__183 = paddle._C_ops.add_(add__182, matmul_173)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__184 = paddle._C_ops.add_(add__183, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_23 = paddle._C_ops.split_with_num(add__184, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_103 = split_with_num_23[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__71 = paddle._C_ops.sigmoid_(slice_103)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_104 = split_with_num_23[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__72 = paddle._C_ops.sigmoid_(slice_104)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_105 = split_with_num_23[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__73 = paddle._C_ops.sigmoid_(slice_105)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__74 = paddle._C_ops.multiply_(sigmoid__72, add__178)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_106 = split_with_num_23[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__47 = paddle._C_ops.tanh_(slice_106)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__75 = paddle._C_ops.multiply_(sigmoid__71, tanh__47)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__185 = paddle._C_ops.add_(multiply__74, multiply__75)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_23 = paddle._C_ops.tanh(add__185)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__76 = paddle._C_ops.multiply_(sigmoid__73, tanh_23)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_174 = paddle.matmul(multiply__76, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__186 = paddle._C_ops.add_(matmul_174, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_48, unsqueeze_49 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__186, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x23x70xf32, -1x1x70xf32]) <- (-1x23x70xf32, -1x1x70xf32)
        combine_78 = [concat_50, unsqueeze_48]

        # pd_op.concat: (-1x24x70xf32) <- ([-1x23x70xf32, -1x1x70xf32], 1xi32)
        concat_52 = paddle._C_ops.concat(combine_78, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_23 = paddle._C_ops.argmax(add__186, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_24 = paddle._C_ops.one_hot(argmax_23 % paddle.cast(constant_29, argmax_23.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_175 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_176 = paddle.matmul(multiply__76, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__187 = paddle._C_ops.add_(matmul_176, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__60, unsqueeze__61 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__187, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__188 = paddle._C_ops.add_(matmul_175, unsqueeze__60)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__48 = paddle._C_ops.tanh_(add__188)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_177 = paddle.matmul(tanh__48, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__24 = paddle._C_ops.softmax_(matmul_177, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_31 = paddle._C_ops.transpose(softmax__24, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_178 = paddle.matmul(transpose_31, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__50, squeeze__51 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_178, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_79 = [squeeze__50, one_hot_24]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_53 = paddle._C_ops.concat(combine_79, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_179 = paddle.matmul(concat_53, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__189 = paddle._C_ops.add_(matmul_179, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_180 = paddle.matmul(multiply__76, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__190 = paddle._C_ops.add_(add__189, matmul_180)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__191 = paddle._C_ops.add_(add__190, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_24 = paddle._C_ops.split_with_num(add__191, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_107 = split_with_num_24[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__74 = paddle._C_ops.sigmoid_(slice_107)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_108 = split_with_num_24[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__75 = paddle._C_ops.sigmoid_(slice_108)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_109 = split_with_num_24[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__76 = paddle._C_ops.sigmoid_(slice_109)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__77 = paddle._C_ops.multiply_(sigmoid__75, add__185)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_110 = split_with_num_24[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__49 = paddle._C_ops.tanh_(slice_110)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__78 = paddle._C_ops.multiply_(sigmoid__74, tanh__49)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__192 = paddle._C_ops.add_(multiply__77, multiply__78)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_24 = paddle._C_ops.tanh(add__192)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__79 = paddle._C_ops.multiply_(sigmoid__76, tanh_24)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_181 = paddle.matmul(multiply__79, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__193 = paddle._C_ops.add_(matmul_181, parameter_251)

        # pd_op.unsqueeze: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze_50, unsqueeze_51 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__193, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x24x70xf32, -1x1x70xf32]) <- (-1x24x70xf32, -1x1x70xf32)
        combine_80 = [concat_52, unsqueeze_50]

        # pd_op.concat: (-1x25x70xf32) <- ([-1x24x70xf32, -1x1x70xf32], 1xi32)
        concat_54 = paddle._C_ops.concat(combine_80, constant_18)

        # pd_op.argmax: (-1xi64) <- (-1x70xf32, 1xi64)
        argmax_24 = paddle._C_ops.argmax(add__193, constant_30, False, False, paddle.int64)

        # pd_op.one_hot: (-1x70xf32) <- (-1xi64, 1xi32)
        one_hot_25 = paddle._C_ops.one_hot(argmax_24 % paddle.cast(constant_29, argmax_24.dtype), constant_29)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_182 = paddle.matmul(add__18, parameter_242, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_183 = paddle.matmul(multiply__79, parameter_243, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__194 = paddle._C_ops.add_(matmul_183, parameter_244)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__62, unsqueeze__63 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__194, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__195 = paddle._C_ops.add_(matmul_182, unsqueeze__62)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__50 = paddle._C_ops.tanh_(add__195)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_184 = paddle.matmul(tanh__50, parameter_245, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__25 = paddle._C_ops.softmax_(matmul_184, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_32 = paddle._C_ops.transpose(softmax__25, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_185 = paddle.matmul(transpose_32, add__18, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__52, squeeze__53 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_185, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x70xf32]) <- (-1x512xf32, -1x70xf32)
        combine_81 = [squeeze__52, one_hot_25]

        # pd_op.concat: (-1x582xf32) <- ([-1x512xf32, -1x70xf32], 1xi32)
        concat_55 = paddle._C_ops.concat(combine_81, constant_18)

        # pd_op.matmul: (-1x1024xf32) <- (-1x582xf32, 1024x582xf32)
        matmul_186 = paddle.matmul(concat_55, parameter_246, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__196 = paddle._C_ops.add_(matmul_186, parameter_247)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_187 = paddle.matmul(multiply__79, parameter_248, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__197 = paddle._C_ops.add_(add__196, matmul_187)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__198 = paddle._C_ops.add_(add__197, parameter_249)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_25 = paddle._C_ops.split_with_num(add__198, 4, constant_18)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_111 = split_with_num_25[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__77 = paddle._C_ops.sigmoid_(slice_111)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_112 = split_with_num_25[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__78 = paddle._C_ops.sigmoid_(slice_112)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_113 = split_with_num_25[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__79 = paddle._C_ops.sigmoid_(slice_113)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__80 = paddle._C_ops.multiply_(sigmoid__78, add__192)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_114 = split_with_num_25[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__51 = paddle._C_ops.tanh_(slice_114)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__81 = paddle._C_ops.multiply_(sigmoid__77, tanh__51)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__199 = paddle._C_ops.add_(multiply__80, multiply__81)

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__52 = paddle._C_ops.tanh_(add__199)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__82 = paddle._C_ops.multiply_(sigmoid__79, tanh__52)

        # pd_op.matmul: (-1x70xf32) <- (-1x256xf32, 256x70xf32)
        matmul_188 = paddle.matmul(multiply__82, parameter_250, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x70xf32) <- (-1x70xf32, 70xf32)
        add__200 = paddle._C_ops.add_(matmul_188, parameter_251)

        # pd_op.unsqueeze_: (-1x1x70xf32, None) <- (-1x70xf32, 1xi64)
        unsqueeze__64, unsqueeze__65 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__200, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x25x70xf32, -1x1x70xf32]) <- (-1x25x70xf32, -1x1x70xf32)
        combine_82 = [concat_54, unsqueeze__64]

        # pd_op.concat: (-1x26x70xf32) <- ([-1x25x70xf32, -1x1x70xf32], 1xi32)
        concat_56 = paddle._C_ops.concat(combine_82, constant_18)

        # pd_op.softmax_: (-1x26x70xf32) <- (-1x26x70xf32)
        softmax__26 = paddle._C_ops.softmax_(concat_56, 2)

        # pd_op.scale_: (-1x26x70xf32) <- (-1x26x70xf32, 1xf32)
        scale__4 = paddle._C_ops.scale_(softmax__26, constant_20, float('0'), True)
        return scale__4



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

    def forward(self, constant_30, constant_29, parameter_229, parameter_220, parameter_219, constant_28, constant_27, constant_25, constant_24, constant_23, constant_22, constant_21, parameter_54, constant_19, parameter_53, parameter_52, constant_18, parameter_51, parameter_50, constant_17, constant_16, parameter_49, constant_15, parameter_48, parameter_47, constant_20, constant_14, parameter_45, constant_26, parameter_46, constant_13, constant_12, constant_11, constant_10, parameter_44, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_55, parameter_56, parameter_58, parameter_57, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_148, parameter_145, parameter_147, parameter_146, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_247, parameter_248, parameter_249, parameter_250, parameter_251, feed_0):
        return self.builtin_module_2370_0_0(constant_30, constant_29, parameter_229, parameter_220, parameter_219, constant_28, constant_27, constant_25, constant_24, constant_23, constant_22, constant_21, parameter_54, constant_19, parameter_53, parameter_52, constant_18, parameter_51, parameter_50, constant_17, constant_16, parameter_49, constant_15, parameter_48, parameter_47, constant_20, constant_14, parameter_45, constant_26, parameter_46, constant_13, constant_12, constant_11, constant_10, parameter_44, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_55, parameter_56, parameter_58, parameter_57, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_148, parameter_145, parameter_147, parameter_146, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_247, parameter_248, parameter_249, parameter_250, parameter_251, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2370_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_30
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_29
            paddle.to_tensor([70], dtype='int32').reshape([1]),
            # parameter_229
            paddle.cast(paddle.randint(low=0, high=3, shape=[], dtype='int64'), 'uint8'),
            # parameter_220
            paddle.to_tensor([256], dtype='int32').reshape([]),
            # parameter_219
            paddle.to_tensor([2], dtype='int32').reshape([]),
            # constant_28
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_27
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
            # constant_25
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
            # constant_24
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            # constant_23
            paddle.to_tensor([32], dtype='int32').reshape([1]),
            # constant_22
            paddle.to_tensor([-1, 3, 2], dtype='int64').reshape([3]),
            # constant_21
            paddle.to_tensor([40], dtype='int32').reshape([1]),
            # parameter_54
            paddle.uniform([3200, 1], dtype='float64', min=0, max=0.5),
            # constant_19
            paddle.to_tensor([1, 20, 1], dtype='int64').reshape([3]),
            # parameter_53
            paddle.uniform([1, 23], dtype='float64', min=0, max=0.5),
            # parameter_52
            paddle.uniform([2, 3], dtype='float64', min=0, max=0.5),
            # constant_18
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # parameter_51
            paddle.uniform([20, 1], dtype='float64', min=0, max=0.5),
            # parameter_50
            paddle.uniform([], dtype='float64', min=0, max=0.5),
            # constant_17
            paddle.to_tensor([20, 1, 2], dtype='int64').reshape([3]),
            # constant_16
            paddle.to_tensor([1, 20, 2], dtype='int64').reshape([3]),
            # parameter_49
            paddle.uniform([20, 20], dtype='float64', min=0, max=0.5),
            # constant_15
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
            # parameter_48
            paddle.uniform([32], dtype='float64', min=0, max=0.5),
            # parameter_47
            paddle.uniform([100], dtype='float64', min=0, max=0.5),
            # constant_20
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            # constant_14
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            # parameter_45
            paddle.uniform([10], dtype='float64', min=0, max=0.5),
            # constant_26
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
            # parameter_46
            paddle.uniform([10], dtype='float64', min=0, max=0.5),
            # constant_13
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            # constant_12
            paddle.to_tensor([20], dtype='int32').reshape([1]),
            # constant_11
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            # constant_10
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            # parameter_44
            paddle.uniform([1, 1, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_9
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            # constant_8
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            # constant_7
            paddle.to_tensor([14, 1], dtype='int64').reshape([2]),
            # constant_6
            paddle.to_tensor([13, 0], dtype='int64').reshape([2]),
            # constant_5
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([54], dtype='int32').reshape([1]),
            # constant_3
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_2
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # constant_1
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_0
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            # parameter_0
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256, 54], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([54], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([16, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([1, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([40, 6], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([6], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([512, 512, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([512, 512, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([256, 1], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([1024, 582], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([256, 70], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_30
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_29
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_229
            paddle.static.InputSpec(shape=[], dtype='uint8'),
            # parameter_220
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # parameter_219
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_28
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_27
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_25
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_24
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_23
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_22
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_21
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_54
            paddle.static.InputSpec(shape=[3200, 1], dtype='float64'),
            # constant_19
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_53
            paddle.static.InputSpec(shape=[1, 23], dtype='float64'),
            # parameter_52
            paddle.static.InputSpec(shape=[2, 3], dtype='float64'),
            # constant_18
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_51
            paddle.static.InputSpec(shape=[20, 1], dtype='float64'),
            # parameter_50
            paddle.static.InputSpec(shape=[], dtype='float64'),
            # constant_17
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_16
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_49
            paddle.static.InputSpec(shape=[20, 20], dtype='float64'),
            # constant_15
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_48
            paddle.static.InputSpec(shape=[32], dtype='float64'),
            # parameter_47
            paddle.static.InputSpec(shape=[100], dtype='float64'),
            # constant_20
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_45
            paddle.static.InputSpec(shape=[10], dtype='float64'),
            # constant_26
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[10], dtype='float64'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_44
            paddle.static.InputSpec(shape=[1, 1, 1, 1], dtype='float32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_7
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_6
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_1
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256, 54], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[54], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[16, 128, 3, 3], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[1, 16, 3, 3], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[40, 6], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[512, 512, 2, 2], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[512, 512, 2, 2], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[256, 1], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[1024, 582], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[256, 70], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
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