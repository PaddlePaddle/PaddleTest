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
    return [664][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_25519_0_0(self, constant_30, parameter_245, parameter_224, parameter_203, parameter_182, constant_29, constant_28, parameter_161, constant_27, constant_26, constant_25, parameter_135, parameter_114, constant_24, parameter_93, constant_23, constant_22, constant_21, parameter_72, constant_20, constant_19, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, parameter_46, constant_9, constant_8, parameter_25, constant_18, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_26, parameter_30, parameter_27, parameter_29, parameter_28, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_40, parameter_37, parameter_39, parameter_38, parameter_41, parameter_45, parameter_42, parameter_44, parameter_43, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_73, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_82, parameter_79, parameter_81, parameter_80, parameter_83, parameter_87, parameter_84, parameter_86, parameter_85, parameter_88, parameter_92, parameter_89, parameter_91, parameter_90, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_136, parameter_140, parameter_137, parameter_139, parameter_138, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_160, parameter_157, parameter_159, parameter_158, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_223, parameter_220, parameter_222, parameter_221, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_260, parameter_257, parameter_259, parameter_258, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_266, feed_0):

        # pd_op.conv2d: (-1x16x112x112xf32) <- (-1x3x224x224xf32, 16x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x112x112xf32) <- (-1x16x112x112xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x16x112x112xf32, 32x16x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(hardswish_0, parameter_5, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__6)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x32x56x56xf32, 64x32x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(hardswish_1, parameter_10, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__12)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x64x28x28xf32, 128x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(hardswish_2, parameter_15, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.flatten_: (-1x128x196xf32, None) <- (-1x128x14x14xf32)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten(batch_norm__18, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x128xf32) <- (-1x128x196xf32)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_0 = paddle._C_ops.shape(transpose_0)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x256xf32) <- (-1x196x128xf32, 128x256xf32)
        matmul_0 = paddle.matmul(transpose_0, parameter_20, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x196x256xf32)
        flatten_0, flatten_1 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_0, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_0, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x256xf32)
        shape_1 = paddle._C_ops.shape(matmul_0)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_0 = [slice_1, constant_2, constant_3]

        # pd_op.reshape_: (-1x196x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__24, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_0, constant_2, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x4x64xf32, 0x-1x196x256xf32) <- (-1x196x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__0, combine_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32]) <- (-1x196x4x64xf32, 3xi64, 1xi32)
        split_0 = paddle._C_ops.split(reshape__2, constant_6, constant_7)

        # builtin.slice: (-1x196x4x16xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_2 = split_0[0]

        # pd_op.transpose: (-1x4x196x16xf32) <- (-1x196x4x16xf32)
        transpose_1 = paddle._C_ops.transpose(slice_2, [0, 2, 1, 3])

        # builtin.slice: (-1x196x4x16xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_3 = split_0[1]

        # pd_op.transpose: (-1x4x196x16xf32) <- (-1x196x4x16xf32)
        transpose_2 = paddle._C_ops.transpose(slice_3, [0, 2, 1, 3])

        # builtin.slice: (-1x196x4x32xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_4 = split_0[2]

        # pd_op.transpose: (-1x4x196x32xf32) <- (-1x196x4x32xf32)
        transpose_3 = paddle._C_ops.transpose(slice_4, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x16x196xf32) <- (-1x4x196x16xf32)
        transpose_4 = paddle._C_ops.transpose(transpose_2, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x196x196xf32) <- (-1x4x196x16xf32, -1x4x16x196xf32)
        matmul_1 = paddle.matmul(transpose_1, transpose_4, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x196x196xf32) <- (-1x4x196x196xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(matmul_1, constant_8, float('0'), True)

        # pd_op.add_: (-1x4x196x196xf32) <- (-1x4x196x196xf32, 4x196x196xf32)
        add__0 = paddle._C_ops.add(scale__0, parameter_25)

        # pd_op.softmax_: (-1x4x196x196xf32) <- (-1x4x196x196xf32)
        softmax__0 = paddle._C_ops.softmax(add__0, -1)

        # pd_op.matmul: (-1x4x196x32xf32) <- (-1x4x196x196xf32, -1x4x196x32xf32)
        matmul_2 = paddle.matmul(softmax__0, transpose_3, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x4x32xf32) <- (-1x4x196x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_2 = [slice_0, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x196x4x32xf32) <- (-1x196x4x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_5, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x196x128xf32) <- (-1x196x128xf32)
        hardswish_3 = paddle._C_ops.hardswish(reshape__4)

        # pd_op.matmul: (-1x196x128xf32) <- (-1x196x128xf32, 128x128xf32)
        matmul_3 = paddle.matmul(hardswish_3, parameter_26, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x196x128xf32)
        flatten_2, flatten_3 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_3, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_2, parameter_27, parameter_28, parameter_29, parameter_30, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_2 = paddle._C_ops.shape(matmul_3)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_2, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_3 = [slice_5, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__30, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x128xf32) <- (-1x196x128xf32, -1x196x128xf32)
        add__1 = paddle._C_ops.add(transpose_0, reshape__6)

        # pd_op.matmul: (-1x196x256xf32) <- (-1x196x128xf32, 128x256xf32)
        matmul_4 = paddle.matmul(add__1, parameter_31, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x196x256xf32)
        flatten_4, flatten_5 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_4, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_4, parameter_32, parameter_33, parameter_34, parameter_35, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x256xf32)
        shape_3 = paddle._C_ops.shape(matmul_4)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_3, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_6, constant_2, constant_3]

        # pd_op.reshape_: (-1x196x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__36, combine_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x196x256xf32) <- (-1x196x256xf32)
        hardswish_4 = paddle._C_ops.hardswish(reshape__8)

        # pd_op.matmul: (-1x196x128xf32) <- (-1x196x256xf32, 256x128xf32)
        matmul_5 = paddle.matmul(hardswish_4, parameter_36, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x196x128xf32)
        flatten_6, flatten_7 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_5, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_6, parameter_37, parameter_38, parameter_39, parameter_40, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_4 = paddle._C_ops.shape(matmul_5)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_4, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_5 = [slice_7, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__42, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x128xf32) <- (-1x196x128xf32, -1x196x128xf32)
        add__2 = paddle._C_ops.add(add__1, reshape__10)

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_5 = paddle._C_ops.shape(add__2)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_5, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x256xf32) <- (-1x196x128xf32, 128x256xf32)
        matmul_6 = paddle.matmul(add__2, parameter_41, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x196x256xf32)
        flatten_8, flatten_9 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_6, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_8, parameter_42, parameter_43, parameter_44, parameter_45, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x256xf32)
        shape_6 = paddle._C_ops.shape(matmul_6)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_6, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_6 = [slice_9, constant_2, constant_3]

        # pd_op.reshape_: (-1x196x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__48, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_8, constant_2, constant_4, constant_5]

        # pd_op.reshape_: (-1x196x4x64xf32, 0x-1x196x256xf32) <- (-1x196x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__12, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32]) <- (-1x196x4x64xf32, 3xi64, 1xi32)
        split_1 = paddle._C_ops.split(reshape__14, constant_6, constant_7)

        # builtin.slice: (-1x196x4x16xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_10 = split_1[0]

        # pd_op.transpose: (-1x4x196x16xf32) <- (-1x196x4x16xf32)
        transpose_6 = paddle._C_ops.transpose(slice_10, [0, 2, 1, 3])

        # builtin.slice: (-1x196x4x16xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_11 = split_1[1]

        # pd_op.transpose: (-1x4x196x16xf32) <- (-1x196x4x16xf32)
        transpose_7 = paddle._C_ops.transpose(slice_11, [0, 2, 1, 3])

        # builtin.slice: (-1x196x4x32xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_12 = split_1[2]

        # pd_op.transpose: (-1x4x196x32xf32) <- (-1x196x4x32xf32)
        transpose_8 = paddle._C_ops.transpose(slice_12, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x16x196xf32) <- (-1x4x196x16xf32)
        transpose_9 = paddle._C_ops.transpose(transpose_7, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x196x196xf32) <- (-1x4x196x16xf32, -1x4x16x196xf32)
        matmul_7 = paddle.matmul(transpose_6, transpose_9, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x4x196x196xf32) <- (-1x4x196x196xf32, 1xf32)
        scale__1 = paddle._C_ops.scale(matmul_7, constant_8, float('0'), True)

        # pd_op.add_: (-1x4x196x196xf32) <- (-1x4x196x196xf32, 4x196x196xf32)
        add__3 = paddle._C_ops.add(scale__1, parameter_46)

        # pd_op.softmax_: (-1x4x196x196xf32) <- (-1x4x196x196xf32)
        softmax__1 = paddle._C_ops.softmax(add__3, -1)

        # pd_op.matmul: (-1x4x196x32xf32) <- (-1x4x196x196xf32, -1x4x196x32xf32)
        matmul_8 = paddle.matmul(softmax__1, transpose_8, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x4x32xf32) <- (-1x4x196x32xf32)
        transpose_10 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_8 = [slice_8, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x196x4x32xf32) <- (-1x196x4x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_10, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x196x128xf32) <- (-1x196x128xf32)
        hardswish_5 = paddle._C_ops.hardswish(reshape__16)

        # pd_op.matmul: (-1x196x128xf32) <- (-1x196x128xf32, 128x128xf32)
        matmul_9 = paddle.matmul(hardswish_5, parameter_47, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x196x128xf32)
        flatten_10, flatten_11 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_9, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_10, parameter_48, parameter_49, parameter_50, parameter_51, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_7 = paddle._C_ops.shape(matmul_9)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_7, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_9 = [slice_13, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__54, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x128xf32) <- (-1x196x128xf32, -1x196x128xf32)
        add__4 = paddle._C_ops.add(add__2, reshape__18)

        # pd_op.matmul: (-1x196x256xf32) <- (-1x196x128xf32, 128x256xf32)
        matmul_10 = paddle.matmul(add__4, parameter_52, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x196x256xf32)
        flatten_12, flatten_13 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_10, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_12, parameter_53, parameter_54, parameter_55, parameter_56, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x256xf32)
        shape_8 = paddle._C_ops.shape(matmul_10)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_8, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_14, constant_2, constant_3]

        # pd_op.reshape_: (-1x196x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__60, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x196x256xf32) <- (-1x196x256xf32)
        hardswish_6 = paddle._C_ops.hardswish(reshape__20)

        # pd_op.matmul: (-1x196x128xf32) <- (-1x196x256xf32, 256x128xf32)
        matmul_11 = paddle.matmul(hardswish_6, parameter_57, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x196x128xf32)
        flatten_14, flatten_15 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_11, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_14, parameter_58, parameter_59, parameter_60, parameter_61, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_9 = paddle._C_ops.shape(matmul_11)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_9, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_11 = [slice_15, constant_2, constant_9]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__66, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x128xf32) <- (-1x196x128xf32, -1x196x128xf32)
        add__5 = paddle._C_ops.add(add__4, reshape__22)

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_10 = paddle._C_ops.shape(add__5)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_10, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x640xf32) <- (-1x196x128xf32, 128x640xf32)
        matmul_12 = paddle.matmul(add__5, parameter_62, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x640xf32, None) <- (-1x196x640xf32)
        flatten_16, flatten_17 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_12, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x640xf32, 640xf32, 640xf32, xf32, xf32, None) <- (-1x640xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_16, parameter_63, parameter_64, parameter_65, parameter_66, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x640xf32)
        shape_11 = paddle._C_ops.shape(matmul_12)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_11, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_12 = [slice_17, constant_2, constant_10]

        # pd_op.reshape_: (-1x196x640xf32, 0x-1x640xf32) <- (-1x640xf32, [1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__72, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_16, constant_2, constant_11, constant_12]

        # pd_op.reshape_: (-1x196x8x-1xf32, 0x-1x196x640xf32) <- (-1x196x640xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__24, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x196x8x-1xf32, -1x196x8x-1xf32]) <- (-1x196x8x-1xf32, 2xi64, 1xi32)
        split_2 = paddle._C_ops.split(reshape__26, constant_13, constant_7)

        # builtin.slice: (-1x196x8x-1xf32) <- ([-1x196x8x-1xf32, -1x196x8x-1xf32])
        slice_18 = split_2[0]

        # pd_op.transpose: (-1x8x196x-1xf32) <- (-1x196x8x-1xf32)
        transpose_11 = paddle._C_ops.transpose(slice_18, [0, 2, 1, 3])

        # builtin.slice: (-1x196x8x-1xf32) <- ([-1x196x8x-1xf32, -1x196x8x-1xf32])
        slice_19 = split_2[1]

        # pd_op.transpose: (-1x8x196x-1xf32) <- (-1x196x8x-1xf32)
        transpose_12 = paddle._C_ops.transpose(slice_19, [0, 2, 1, 3])

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_12 = paddle._C_ops.shape(add__5)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_12, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_20, constant_14, constant_14, constant_9]

        # pd_op.reshape_: (-1x14x14x128xf32, 0x-1x196x128xf32) <- (-1x196x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__5, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.strided_slice: (-1x7x7x128xf32) <- (-1x14x14x128xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(reshape__28, [1, 2], constant_15, constant_16, constant_17)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_15 = [slice_20, constant_12, constant_9]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x7x7x128xf32) <- (-1x7x7x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape(strided_slice_0, combine_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_13 = paddle.matmul(reshape__30, parameter_67, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x-1x128xf32)
        flatten_18, flatten_19 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_13, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_18, parameter_68, parameter_69, parameter_70, parameter_71, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x128xf32)
        shape_13 = paddle._C_ops.shape(matmul_13)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(shape_13, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(shape_13, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_16 = [slice_21, slice_22, constant_9]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__78, combine_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_16, constant_19, constant_11, constant_20]

        # pd_op.reshape_: (-1x49x8x16xf32, 0x-1x-1x128xf32) <- (-1x-1x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__32, combine_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x49x16xf32) <- (-1x49x8x16xf32)
        transpose_13 = paddle._C_ops.transpose(reshape__34, [0, 2, 1, 3])

        # pd_op.transpose: (-1x8x-1x196xf32) <- (-1x8x196x-1xf32)
        transpose_14 = paddle._C_ops.transpose(transpose_11, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x196xf32) <- (-1x8x49x16xf32, -1x8x-1x196xf32)
        matmul_14 = paddle.matmul(transpose_13, transpose_14, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x196xf32) <- (-1x8x49x196xf32, 1xf32)
        scale__2 = paddle._C_ops.scale(matmul_14, constant_8, float('0'), True)

        # pd_op.add_: (-1x8x49x196xf32) <- (-1x8x49x196xf32, 8x49x196xf32)
        add__6 = paddle._C_ops.add(scale__2, parameter_72)

        # pd_op.softmax_: (-1x8x49x196xf32) <- (-1x8x49x196xf32)
        softmax__2 = paddle._C_ops.softmax(add__6, -1)

        # pd_op.matmul: (-1x8x49x-1xf32) <- (-1x8x49x196xf32, -1x8x196x-1xf32)
        matmul_15 = paddle.matmul(softmax__2, transpose_12, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x-1xf32) <- (-1x8x49x-1xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_18 = [slice_16, constant_12, constant_21]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x49x8x-1xf32) <- (-1x49x8x-1xf32, [1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_15, combine_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_7 = paddle._C_ops.hardswish(reshape__36)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_16 = paddle.matmul(hardswish_7, parameter_73, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_20, flatten_21 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_16, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_20, parameter_74, parameter_75, parameter_76, parameter_77, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_14 = paddle._C_ops.shape(matmul_16)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(shape_14, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_14, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_19 = [slice_23, slice_24, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__84, combine_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x256xf32, 256x512xf32)
        matmul_17 = paddle.matmul(reshape__38, parameter_78, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_22, flatten_23 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_17, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_22, parameter_79, parameter_80, parameter_81, parameter_82, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_15 = paddle._C_ops.shape(matmul_17)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_15, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(shape_15, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_20 = [slice_25, slice_26, constant_21]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__90, combine_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_8 = paddle._C_ops.hardswish(reshape__40)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_18 = paddle.matmul(hardswish_8, parameter_83, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_24, flatten_25 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_18, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_24, parameter_84, parameter_85, parameter_86, parameter_87, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_16 = paddle._C_ops.shape(matmul_18)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(shape_16, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_16, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_21 = [slice_27, slice_28, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__96, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_0 = reshape__38 + reshape__42

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_17 = paddle._C_ops.shape(add_0)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_17, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(shape_17, [0], constant_1, constant_18, [1], [0])

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_19 = paddle.matmul(add_0, parameter_88, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_26, flatten_27 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_19, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_26, parameter_89, parameter_90, parameter_91, parameter_92, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_18 = paddle._C_ops.shape(matmul_19)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(shape_18, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_18, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_22 = [slice_31, slice_32, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__102, combine_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_29, slice_30, constant_23, constant_5]

        # pd_op.reshape_: (-1x-1x6x64xf32, 0x-1x-1x384xf32) <- (-1x-1x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__44, combine_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32]) <- (-1x-1x6x64xf32, 3xi64, 1xi32)
        split_3 = paddle._C_ops.split(reshape__46, constant_6, constant_7)

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_33 = split_3[0]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_16 = paddle._C_ops.transpose(slice_33, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_34 = split_3[1]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_17 = paddle._C_ops.transpose(slice_34, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x32xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_35 = split_3[2]

        # pd_op.transpose: (-1x6x-1x32xf32) <- (-1x-1x6x32xf32)
        transpose_18 = paddle._C_ops.transpose(slice_35, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x16x-1xf32) <- (-1x6x-1x16xf32)
        transpose_19 = paddle._C_ops.transpose(transpose_17, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x-1x-1xf32) <- (-1x6x-1x16xf32, -1x6x16x-1xf32)
        matmul_20 = paddle.matmul(transpose_16, transpose_19, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x6x-1x-1xf32) <- (-1x6x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_20, constant_8, float('0'), True)

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x-1x-1xf32, 6x49x49xf32)
        add_1 = scale_0 + parameter_93

        # pd_op.softmax_: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        softmax__3 = paddle._C_ops.softmax(add_1, -1)

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x-1x32xf32)
        matmul_21 = paddle.matmul(softmax__3, transpose_18, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        transpose_20 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_24 = [slice_29, slice_30, constant_24]

        # pd_op.reshape_: (-1x-1x192xf32, 0x-1x49x6x32xf32) <- (-1x49x6x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_20, combine_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x192xf32) <- (-1x-1x192xf32)
        hardswish_9 = paddle._C_ops.hardswish(reshape__48)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x192xf32, 192x256xf32)
        matmul_22 = paddle.matmul(hardswish_9, parameter_94, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_28, flatten_29 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_22, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_28, parameter_95, parameter_96, parameter_97, parameter_98, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_19 = paddle._C_ops.shape(matmul_22)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(shape_19, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(shape_19, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_36, slice_37, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__108, combine_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_2 = add_0 + reshape__50

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x256xf32, 256x512xf32)
        matmul_23 = paddle.matmul(add_2, parameter_99, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_30, flatten_31 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_23, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_30, parameter_100, parameter_101, parameter_102, parameter_103, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_20 = paddle._C_ops.shape(matmul_23)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(shape_20, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(shape_20, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_26 = [slice_38, slice_39, constant_21]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__114, combine_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_10 = paddle._C_ops.hardswish(reshape__52)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_24 = paddle.matmul(hardswish_10, parameter_104, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_32, flatten_33 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_24, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_32, parameter_105, parameter_106, parameter_107, parameter_108, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_21 = paddle._C_ops.shape(matmul_24)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(shape_21, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(shape_21, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_27 = [slice_40, slice_41, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__120, combine_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_3 = add_2 + reshape__54

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_22 = paddle._C_ops.shape(add_3)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(shape_22, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(shape_22, [0], constant_1, constant_18, [1], [0])

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_25 = paddle.matmul(add_3, parameter_109, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_34, flatten_35 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_25, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_34, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_23 = paddle._C_ops.shape(matmul_25)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(shape_23, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(shape_23, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_28 = [slice_44, slice_45, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__126, combine_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_42, slice_43, constant_23, constant_5]

        # pd_op.reshape_: (-1x-1x6x64xf32, 0x-1x-1x384xf32) <- (-1x-1x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__56, combine_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32]) <- (-1x-1x6x64xf32, 3xi64, 1xi32)
        split_4 = paddle._C_ops.split(reshape__58, constant_6, constant_7)

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_46 = split_4[0]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_21 = paddle._C_ops.transpose(slice_46, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_47 = split_4[1]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_22 = paddle._C_ops.transpose(slice_47, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x32xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_48 = split_4[2]

        # pd_op.transpose: (-1x6x-1x32xf32) <- (-1x-1x6x32xf32)
        transpose_23 = paddle._C_ops.transpose(slice_48, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x16x-1xf32) <- (-1x6x-1x16xf32)
        transpose_24 = paddle._C_ops.transpose(transpose_22, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x-1x-1xf32) <- (-1x6x-1x16xf32, -1x6x16x-1xf32)
        matmul_26 = paddle.matmul(transpose_21, transpose_24, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x6x-1x-1xf32) <- (-1x6x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_26, constant_8, float('0'), True)

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x-1x-1xf32, 6x49x49xf32)
        add_4 = scale_1 + parameter_114

        # pd_op.softmax_: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        softmax__4 = paddle._C_ops.softmax(add_4, -1)

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x-1x32xf32)
        matmul_27 = paddle.matmul(softmax__4, transpose_23, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        transpose_25 = paddle._C_ops.transpose(matmul_27, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_30 = [slice_42, slice_43, constant_24]

        # pd_op.reshape_: (-1x-1x192xf32, 0x-1x49x6x32xf32) <- (-1x49x6x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_25, combine_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x192xf32) <- (-1x-1x192xf32)
        hardswish_11 = paddle._C_ops.hardswish(reshape__60)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x192xf32, 192x256xf32)
        matmul_28 = paddle.matmul(hardswish_11, parameter_115, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_36, flatten_37 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_28, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_36, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_24 = paddle._C_ops.shape(matmul_28)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(shape_24, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(shape_24, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_31 = [slice_49, slice_50, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__132, combine_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_5 = add_3 + reshape__62

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x256xf32, 256x512xf32)
        matmul_29 = paddle.matmul(add_5, parameter_120, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_38, flatten_39 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_29, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_38, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_25 = paddle._C_ops.shape(matmul_29)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(shape_25, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(shape_25, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_32 = [slice_51, slice_52, constant_21]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__138, combine_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_12 = paddle._C_ops.hardswish(reshape__64)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_30 = paddle.matmul(hardswish_12, parameter_125, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_40, flatten_41 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_30, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_40, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_26 = paddle._C_ops.shape(matmul_30)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(shape_26, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(shape_26, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_33 = [slice_53, slice_54, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__144, combine_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_6 = add_5 + reshape__66

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_27 = paddle._C_ops.shape(add_6)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(shape_27, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(shape_27, [0], constant_1, constant_18, [1], [0])

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_31 = paddle.matmul(add_6, parameter_130, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_42, flatten_43 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_31, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_42, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_28 = paddle._C_ops.shape(matmul_31)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(shape_28, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(shape_28, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_34 = [slice_57, slice_58, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__150, combine_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_55, slice_56, constant_23, constant_5]

        # pd_op.reshape_: (-1x-1x6x64xf32, 0x-1x-1x384xf32) <- (-1x-1x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__68, combine_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32]) <- (-1x-1x6x64xf32, 3xi64, 1xi32)
        split_5 = paddle._C_ops.split(reshape__70, constant_6, constant_7)

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_59 = split_5[0]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_26 = paddle._C_ops.transpose(slice_59, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_60 = split_5[1]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_27 = paddle._C_ops.transpose(slice_60, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x32xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_61 = split_5[2]

        # pd_op.transpose: (-1x6x-1x32xf32) <- (-1x-1x6x32xf32)
        transpose_28 = paddle._C_ops.transpose(slice_61, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x16x-1xf32) <- (-1x6x-1x16xf32)
        transpose_29 = paddle._C_ops.transpose(transpose_27, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x-1x-1xf32) <- (-1x6x-1x16xf32, -1x6x16x-1xf32)
        matmul_32 = paddle.matmul(transpose_26, transpose_29, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x6x-1x-1xf32) <- (-1x6x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_32, constant_8, float('0'), True)

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x-1x-1xf32, 6x49x49xf32)
        add_7 = scale_2 + parameter_135

        # pd_op.softmax_: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        softmax__5 = paddle._C_ops.softmax(add_7, -1)

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x-1x32xf32)
        matmul_33 = paddle.matmul(softmax__5, transpose_28, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_33, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_36 = [slice_55, slice_56, constant_24]

        # pd_op.reshape_: (-1x-1x192xf32, 0x-1x49x6x32xf32) <- (-1x49x6x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_30, combine_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x192xf32) <- (-1x-1x192xf32)
        hardswish_13 = paddle._C_ops.hardswish(reshape__72)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x192xf32, 192x256xf32)
        matmul_34 = paddle.matmul(hardswish_13, parameter_136, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_44, flatten_45 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_34, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_44, parameter_137, parameter_138, parameter_139, parameter_140, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_29 = paddle._C_ops.shape(matmul_34)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(shape_29, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(shape_29, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_37 = [slice_62, slice_63, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__156, combine_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_8 = add_6 + reshape__74

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x256xf32, 256x512xf32)
        matmul_35 = paddle.matmul(add_8, parameter_141, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_46, flatten_47 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_35, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_46, parameter_142, parameter_143, parameter_144, parameter_145, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_30 = paddle._C_ops.shape(matmul_35)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(shape_30, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(shape_30, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_38 = [slice_64, slice_65, constant_21]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__162, combine_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_14 = paddle._C_ops.hardswish(reshape__76)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_36 = paddle.matmul(hardswish_14, parameter_146, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_48, flatten_49 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_36, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_48, parameter_147, parameter_148, parameter_149, parameter_150, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_31 = paddle._C_ops.shape(matmul_36)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(shape_31, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(shape_31, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_39 = [slice_66, slice_67, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__168, combine_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_9 = add_8 + reshape__78

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_32 = paddle._C_ops.shape(add_9)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(shape_32, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(shape_32, [0], constant_1, constant_18, [1], [0])

        # pd_op.matmul: (-1x-1x1280xf32) <- (-1x-1x256xf32, 256x1280xf32)
        matmul_37 = paddle.matmul(add_9, parameter_151, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x1280xf32, None) <- (-1x-1x1280xf32)
        flatten_50, flatten_51 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_37, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x1280xf32, 1280xf32, 1280xf32, xf32, xf32, None) <- (-1x1280xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_50, parameter_152, parameter_153, parameter_154, parameter_155, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x1280xf32)
        shape_33 = paddle._C_ops.shape(matmul_37)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(shape_33, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(shape_33, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_40 = [slice_70, slice_71, constant_25]

        # pd_op.reshape_: (-1x-1x1280xf32, 0x-1x1280xf32) <- (-1x1280xf32, [1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__174, combine_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_68, slice_69, constant_20, constant_12]

        # pd_op.reshape_: (-1x-1x16x-1xf32, 0x-1x-1x1280xf32) <- (-1x-1x1280xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__80, combine_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x-1x16x-1xf32, -1x-1x16x-1xf32]) <- (-1x-1x16x-1xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(reshape__82, constant_13, constant_7)

        # builtin.slice: (-1x-1x16x-1xf32) <- ([-1x-1x16x-1xf32, -1x-1x16x-1xf32])
        slice_72 = split_6[0]

        # pd_op.transpose: (-1x16x-1x-1xf32) <- (-1x-1x16x-1xf32)
        transpose_31 = paddle._C_ops.transpose(slice_72, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x16x-1xf32) <- ([-1x-1x16x-1xf32, -1x-1x16x-1xf32])
        slice_73 = split_6[1]

        # pd_op.transpose: (-1x16x-1x-1xf32) <- (-1x-1x16x-1xf32)
        transpose_32 = paddle._C_ops.transpose(slice_73, [0, 2, 1, 3])

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_34 = paddle._C_ops.shape(add_9)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(shape_34, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_74, constant_26, constant_26, constant_3]

        # pd_op.reshape_: (-1x7x7x256xf32, 0x-1x-1x256xf32) <- (-1x-1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_9, combine_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.strided_slice: (-1x4x4x256xf32) <- (-1x7x7x256xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(reshape__84, [1, 2], constant_15, constant_27, constant_17)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_43 = [slice_74, constant_12, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x4x4x256xf32) <- (-1x4x4x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape(strided_slice_1, combine_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_38 = paddle.matmul(reshape__86, parameter_156, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_52, flatten_53 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_38, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_52, parameter_157, parameter_158, parameter_159, parameter_160, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_35 = paddle._C_ops.shape(matmul_38)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(shape_35, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(shape_35, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_44 = [slice_75, slice_76, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__180, combine_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [slice_68, constant_20, constant_20, constant_20]

        # pd_op.reshape_: (-1x16x16x16xf32, 0x-1x-1x256xf32) <- (-1x-1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__88, combine_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x16x16xf32) <- (-1x16x16x16xf32)
        transpose_33 = paddle._C_ops.transpose(reshape__90, [0, 2, 1, 3])

        # pd_op.transpose: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32)
        transpose_34 = paddle._C_ops.transpose(transpose_31, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x16x-1xf32) <- (-1x16x16x16xf32, -1x16x-1x-1xf32)
        matmul_39 = paddle.matmul(transpose_33, transpose_34, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x16x16x-1xf32) <- (-1x16x16x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_39, constant_8, float('0'), True)

        # pd_op.add: (-1x16x16x49xf32) <- (-1x16x16x-1xf32, 16x16x49xf32)
        add_10 = scale_3 + parameter_161

        # pd_op.softmax_: (-1x16x16x49xf32) <- (-1x16x16x49xf32)
        softmax__6 = paddle._C_ops.softmax(add_10, -1)

        # pd_op.matmul: (-1x16x16x-1xf32) <- (-1x16x16x49xf32, -1x16x-1x-1xf32)
        matmul_40 = paddle.matmul(softmax__6, transpose_32, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x16x-1xf32) <- (-1x16x16x-1xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_46 = [slice_68, constant_12, constant_28]

        # pd_op.reshape_: (-1x-1x1024xf32, 0x-1x16x16x-1xf32) <- (-1x16x16x-1xf32, [1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_35, combine_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        hardswish_15 = paddle._C_ops.hardswish(reshape__92)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1024xf32, 1024x384xf32)
        matmul_41 = paddle.matmul(hardswish_15, parameter_162, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_54, flatten_55 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_41, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_54, parameter_163, parameter_164, parameter_165, parameter_166, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_36 = paddle._C_ops.shape(matmul_41)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(shape_36, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(shape_36, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_47 = [slice_77, slice_78, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__186, combine_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_42 = paddle.matmul(reshape__94, parameter_167, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_56, flatten_57 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_42, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_56, parameter_168, parameter_169, parameter_170, parameter_171, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_37 = paddle._C_ops.shape(matmul_42)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(shape_37, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(shape_37, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_48 = [slice_79, slice_80, constant_29]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__192, combine_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_16 = paddle._C_ops.hardswish(reshape__96)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_43 = paddle.matmul(hardswish_16, parameter_172, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_58, flatten_59 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_43, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_58, parameter_173, parameter_174, parameter_175, parameter_176, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_38 = paddle._C_ops.shape(matmul_43)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(shape_38, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(shape_38, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_49 = [slice_81, slice_82, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__198, combine_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_11 = reshape__94 + reshape__98

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_39 = paddle._C_ops.shape(add_11)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(shape_39, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(shape_39, [0], constant_1, constant_18, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x384xf32, 384x512xf32)
        matmul_44 = paddle.matmul(add_11, parameter_177, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_60, flatten_61 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_44, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_60, parameter_178, parameter_179, parameter_180, parameter_181, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_40 = paddle._C_ops.shape(matmul_44)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(shape_40, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(shape_40, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_50 = [slice_85, slice_86, constant_21]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__204, combine_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_51 = [slice_83, slice_84, constant_11, constant_5]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__100, combine_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32]) <- (-1x-1x8x64xf32, 3xi64, 1xi32)
        split_7 = paddle._C_ops.split(reshape__102, constant_6, constant_7)

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_87 = split_7[0]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_36 = paddle._C_ops.transpose(slice_87, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_88 = split_7[1]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_37 = paddle._C_ops.transpose(slice_88, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x32xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_89 = split_7[2]

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_38 = paddle._C_ops.transpose(slice_89, [0, 2, 1, 3])

        # pd_op.transpose: (-1x8x16x-1xf32) <- (-1x8x-1x16xf32)
        transpose_39 = paddle._C_ops.transpose(transpose_37, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x16xf32, -1x8x16x-1xf32)
        matmul_45 = paddle.matmul(transpose_36, transpose_39, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_45, constant_8, float('0'), True)

        # pd_op.add: (-1x8x16x16xf32) <- (-1x8x-1x-1xf32, 8x16x16xf32)
        add_12 = scale_4 + parameter_182

        # pd_op.softmax_: (-1x8x16x16xf32) <- (-1x8x16x16xf32)
        softmax__7 = paddle._C_ops.softmax(add_12, -1)

        # pd_op.matmul: (-1x8x16x32xf32) <- (-1x8x16x16xf32, -1x8x-1x32xf32)
        matmul_46 = paddle.matmul(softmax__7, transpose_38, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x8x32xf32) <- (-1x8x16x32xf32)
        transpose_40 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_52 = [slice_83, slice_84, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x16x8x32xf32) <- (-1x16x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_40, combine_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x256xf32) <- (-1x-1x256xf32)
        hardswish_17 = paddle._C_ops.hardswish(reshape__104)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_47 = paddle.matmul(hardswish_17, parameter_183, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_62, flatten_63 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_47, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_62, parameter_184, parameter_185, parameter_186, parameter_187, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_41 = paddle._C_ops.shape(matmul_47)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(shape_41, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(shape_41, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_53 = [slice_90, slice_91, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__210, combine_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_13 = add_11 + reshape__106

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_48 = paddle.matmul(add_13, parameter_188, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_64, flatten_65 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_48, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_64, parameter_189, parameter_190, parameter_191, parameter_192, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_42 = paddle._C_ops.shape(matmul_48)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(shape_42, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(shape_42, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_54 = [slice_92, slice_93, constant_29]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__216, combine_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_18 = paddle._C_ops.hardswish(reshape__108)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_49 = paddle.matmul(hardswish_18, parameter_193, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_66, flatten_67 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_49, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_66, parameter_194, parameter_195, parameter_196, parameter_197, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_43 = paddle._C_ops.shape(matmul_49)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(shape_43, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(shape_43, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_55 = [slice_94, slice_95, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__222, combine_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_14 = add_13 + reshape__110

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_44 = paddle._C_ops.shape(add_14)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(shape_44, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(shape_44, [0], constant_1, constant_18, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x384xf32, 384x512xf32)
        matmul_50 = paddle.matmul(add_14, parameter_198, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_68, flatten_69 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_50, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_68, parameter_199, parameter_200, parameter_201, parameter_202, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_45 = paddle._C_ops.shape(matmul_50)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(shape_45, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(shape_45, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_56 = [slice_98, slice_99, constant_21]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__228, combine_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_57 = [slice_96, slice_97, constant_11, constant_5]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__112, combine_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32]) <- (-1x-1x8x64xf32, 3xi64, 1xi32)
        split_8 = paddle._C_ops.split(reshape__114, constant_6, constant_7)

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_100 = split_8[0]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_41 = paddle._C_ops.transpose(slice_100, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_101 = split_8[1]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_42 = paddle._C_ops.transpose(slice_101, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x32xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_102 = split_8[2]

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_43 = paddle._C_ops.transpose(slice_102, [0, 2, 1, 3])

        # pd_op.transpose: (-1x8x16x-1xf32) <- (-1x8x-1x16xf32)
        transpose_44 = paddle._C_ops.transpose(transpose_42, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x16xf32, -1x8x16x-1xf32)
        matmul_51 = paddle.matmul(transpose_41, transpose_44, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_51, constant_8, float('0'), True)

        # pd_op.add: (-1x8x16x16xf32) <- (-1x8x-1x-1xf32, 8x16x16xf32)
        add_15 = scale_5 + parameter_203

        # pd_op.softmax_: (-1x8x16x16xf32) <- (-1x8x16x16xf32)
        softmax__8 = paddle._C_ops.softmax(add_15, -1)

        # pd_op.matmul: (-1x8x16x32xf32) <- (-1x8x16x16xf32, -1x8x-1x32xf32)
        matmul_52 = paddle.matmul(softmax__8, transpose_43, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x8x32xf32) <- (-1x8x16x32xf32)
        transpose_45 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_58 = [slice_96, slice_97, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x16x8x32xf32) <- (-1x16x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_45, combine_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x256xf32) <- (-1x-1x256xf32)
        hardswish_19 = paddle._C_ops.hardswish(reshape__116)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_53 = paddle.matmul(hardswish_19, parameter_204, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_70, flatten_71 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_53, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_70, parameter_205, parameter_206, parameter_207, parameter_208, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_46 = paddle._C_ops.shape(matmul_53)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_103 = paddle._C_ops.slice(shape_46, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(shape_46, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_59 = [slice_103, slice_104, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__234, combine_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_16 = add_14 + reshape__118

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_54 = paddle.matmul(add_16, parameter_209, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_72, flatten_73 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_54, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_72, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_47 = paddle._C_ops.shape(matmul_54)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(shape_47, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(shape_47, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_60 = [slice_105, slice_106, constant_29]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__240, combine_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_20 = paddle._C_ops.hardswish(reshape__120)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_55 = paddle.matmul(hardswish_20, parameter_214, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_74, flatten_75 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_55, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_74, parameter_215, parameter_216, parameter_217, parameter_218, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_48 = paddle._C_ops.shape(matmul_55)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(shape_48, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_108 = paddle._C_ops.slice(shape_48, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_61 = [slice_107, slice_108, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__246, combine_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_17 = add_16 + reshape__122

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_49 = paddle._C_ops.shape(add_17)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_109 = paddle._C_ops.slice(shape_49, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_110 = paddle._C_ops.slice(shape_49, [0], constant_1, constant_18, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x384xf32, 384x512xf32)
        matmul_56 = paddle.matmul(add_17, parameter_219, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_76, flatten_77 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_56, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_76, parameter_220, parameter_221, parameter_222, parameter_223, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_50 = paddle._C_ops.shape(matmul_56)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_111 = paddle._C_ops.slice(shape_50, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(shape_50, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_62 = [slice_111, slice_112, constant_21]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__252, combine_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_63 = [slice_109, slice_110, constant_11, constant_5]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__124, combine_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32]) <- (-1x-1x8x64xf32, 3xi64, 1xi32)
        split_9 = paddle._C_ops.split(reshape__126, constant_6, constant_7)

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_113 = split_9[0]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_46 = paddle._C_ops.transpose(slice_113, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_114 = split_9[1]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_47 = paddle._C_ops.transpose(slice_114, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x32xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_115 = split_9[2]

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_48 = paddle._C_ops.transpose(slice_115, [0, 2, 1, 3])

        # pd_op.transpose: (-1x8x16x-1xf32) <- (-1x8x-1x16xf32)
        transpose_49 = paddle._C_ops.transpose(transpose_47, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x16xf32, -1x8x16x-1xf32)
        matmul_57 = paddle.matmul(transpose_46, transpose_49, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_57, constant_8, float('0'), True)

        # pd_op.add: (-1x8x16x16xf32) <- (-1x8x-1x-1xf32, 8x16x16xf32)
        add_18 = scale_6 + parameter_224

        # pd_op.softmax_: (-1x8x16x16xf32) <- (-1x8x16x16xf32)
        softmax__9 = paddle._C_ops.softmax(add_18, -1)

        # pd_op.matmul: (-1x8x16x32xf32) <- (-1x8x16x16xf32, -1x8x-1x32xf32)
        matmul_58 = paddle.matmul(softmax__9, transpose_48, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x8x32xf32) <- (-1x8x16x32xf32)
        transpose_50 = paddle._C_ops.transpose(matmul_58, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_64 = [slice_109, slice_110, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x16x8x32xf32) <- (-1x16x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_50, combine_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x256xf32) <- (-1x-1x256xf32)
        hardswish_21 = paddle._C_ops.hardswish(reshape__128)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_59 = paddle.matmul(hardswish_21, parameter_225, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_78, flatten_79 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_59, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_78, parameter_226, parameter_227, parameter_228, parameter_229, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_51 = paddle._C_ops.shape(matmul_59)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_116 = paddle._C_ops.slice(shape_51, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_117 = paddle._C_ops.slice(shape_51, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_65 = [slice_116, slice_117, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__258, combine_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_19 = add_17 + reshape__130

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_60 = paddle.matmul(add_19, parameter_230, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_80, flatten_81 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_60, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_80, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_52 = paddle._C_ops.shape(matmul_60)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_118 = paddle._C_ops.slice(shape_52, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(shape_52, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_66 = [slice_118, slice_119, constant_29]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__264, combine_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_22 = paddle._C_ops.hardswish(reshape__132)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_61 = paddle.matmul(hardswish_22, parameter_235, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_82, flatten_83 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_61, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_82, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_53 = paddle._C_ops.shape(matmul_61)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_120 = paddle._C_ops.slice(shape_53, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_121 = paddle._C_ops.slice(shape_53, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_67 = [slice_120, slice_121, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__270, combine_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_20 = add_19 + reshape__134

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_54 = paddle._C_ops.shape(add_20)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(shape_54, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_123 = paddle._C_ops.slice(shape_54, [0], constant_1, constant_18, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x384xf32, 384x512xf32)
        matmul_62 = paddle.matmul(add_20, parameter_240, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_84, flatten_85 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_62, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_84, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_55 = paddle._C_ops.shape(matmul_62)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_124 = paddle._C_ops.slice(shape_55, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_125 = paddle._C_ops.slice(shape_55, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_68 = [slice_124, slice_125, constant_21]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__276, combine_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_69 = [slice_122, slice_123, constant_11, constant_5]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__136, combine_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.split: ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32]) <- (-1x-1x8x64xf32, 3xi64, 1xi32)
        split_10 = paddle._C_ops.split(reshape__138, constant_6, constant_7)

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_126 = split_10[0]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_51 = paddle._C_ops.transpose(slice_126, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_127 = split_10[1]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_52 = paddle._C_ops.transpose(slice_127, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x32xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_128 = split_10[2]

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_53 = paddle._C_ops.transpose(slice_128, [0, 2, 1, 3])

        # pd_op.transpose: (-1x8x16x-1xf32) <- (-1x8x-1x16xf32)
        transpose_54 = paddle._C_ops.transpose(transpose_52, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x16xf32, -1x8x16x-1xf32)
        matmul_63 = paddle.matmul(transpose_51, transpose_54, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_63, constant_8, float('0'), True)

        # pd_op.add: (-1x8x16x16xf32) <- (-1x8x-1x-1xf32, 8x16x16xf32)
        add_21 = scale_7 + parameter_245

        # pd_op.softmax_: (-1x8x16x16xf32) <- (-1x8x16x16xf32)
        softmax__10 = paddle._C_ops.softmax(add_21, -1)

        # pd_op.matmul: (-1x8x16x32xf32) <- (-1x8x16x16xf32, -1x8x-1x32xf32)
        matmul_64 = paddle.matmul(softmax__10, transpose_53, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x16x8x32xf32) <- (-1x8x16x32xf32)
        transpose_55 = paddle._C_ops.transpose(matmul_64, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_70 = [slice_122, slice_123, constant_3]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x16x8x32xf32) <- (-1x16x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_55, combine_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x256xf32) <- (-1x-1x256xf32)
        hardswish_23 = paddle._C_ops.hardswish(reshape__140)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_65 = paddle.matmul(hardswish_23, parameter_246, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_86, flatten_87 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_65, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_86, parameter_247, parameter_248, parameter_249, parameter_250, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_56 = paddle._C_ops.shape(matmul_65)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_129 = paddle._C_ops.slice(shape_56, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_130 = paddle._C_ops.slice(shape_56, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_71 = [slice_129, slice_130, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__282, combine_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_22 = add_20 + reshape__142

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_66 = paddle.matmul(add_22, parameter_251, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_88, flatten_89 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_66, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_88, parameter_252, parameter_253, parameter_254, parameter_255, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_57 = paddle._C_ops.shape(matmul_66)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_131 = paddle._C_ops.slice(shape_57, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_132 = paddle._C_ops.slice(shape_57, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_72 = [slice_131, slice_132, constant_29]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__288, combine_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_24 = paddle._C_ops.hardswish(reshape__144)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_67 = paddle.matmul(hardswish_24, parameter_256, transpose_x=False, transpose_y=False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_90, flatten_91 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_67, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_90, parameter_257, parameter_258, parameter_259, parameter_260, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_58 = paddle._C_ops.shape(matmul_67)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_133 = paddle._C_ops.slice(shape_58, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(shape_58, [0], constant_1, constant_18, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_73 = [slice_133, slice_134, constant_22]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__294, combine_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_23 = add_22 + reshape__146

        # pd_op.mean: (-1x384xf32) <- (-1x-1x384xf32)
        mean_0 = paddle._C_ops.mean(add_23, [1], False)

        # pd_op.reshape_: (-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, 2xi64)
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape(mean_0, constant_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__148, parameter_261, parameter_262, parameter_263, parameter_264, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.matmul: (-1x1000xf32) <- (-1x384xf32, 384x1000xf32)
        matmul_68 = paddle.matmul(batch_norm__300, parameter_265, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__7 = paddle._C_ops.add(matmul_68, parameter_266)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__11 = paddle._C_ops.softmax(add__7, -1)
        return softmax__11



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

    def forward(self, constant_30, parameter_245, parameter_224, parameter_203, parameter_182, constant_29, constant_28, parameter_161, constant_27, constant_26, constant_25, parameter_135, parameter_114, constant_24, parameter_93, constant_23, constant_22, constant_21, parameter_72, constant_20, constant_19, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, parameter_46, constant_9, constant_8, parameter_25, constant_18, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_26, parameter_30, parameter_27, parameter_29, parameter_28, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_40, parameter_37, parameter_39, parameter_38, parameter_41, parameter_45, parameter_42, parameter_44, parameter_43, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_73, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_82, parameter_79, parameter_81, parameter_80, parameter_83, parameter_87, parameter_84, parameter_86, parameter_85, parameter_88, parameter_92, parameter_89, parameter_91, parameter_90, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_136, parameter_140, parameter_137, parameter_139, parameter_138, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_160, parameter_157, parameter_159, parameter_158, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_223, parameter_220, parameter_222, parameter_221, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_260, parameter_257, parameter_259, parameter_258, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_266, feed_0):
        return self.builtin_module_25519_0_0(constant_30, parameter_245, parameter_224, parameter_203, parameter_182, constant_29, constant_28, parameter_161, constant_27, constant_26, constant_25, parameter_135, parameter_114, constant_24, parameter_93, constant_23, constant_22, constant_21, parameter_72, constant_20, constant_19, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, parameter_46, constant_9, constant_8, parameter_25, constant_18, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_26, parameter_30, parameter_27, parameter_29, parameter_28, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_40, parameter_37, parameter_39, parameter_38, parameter_41, parameter_45, parameter_42, parameter_44, parameter_43, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_73, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_82, parameter_79, parameter_81, parameter_80, parameter_83, parameter_87, parameter_84, parameter_86, parameter_85, parameter_88, parameter_92, parameter_89, parameter_91, parameter_90, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_136, parameter_140, parameter_137, parameter_139, parameter_138, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_160, parameter_157, parameter_159, parameter_158, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_223, parameter_220, parameter_222, parameter_221, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_260, parameter_257, parameter_259, parameter_258, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_266, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_25519_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_30
            paddle.to_tensor([-1, 384], dtype='int64').reshape([2]),
            # parameter_245
            paddle.uniform([8, 16, 16], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([8, 16, 16], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([8, 16, 16], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([8, 16, 16], dtype='float32', min=0, max=0.5),
            # constant_29
            paddle.to_tensor([768], dtype='int32').reshape([1]),
            # constant_28
            paddle.to_tensor([1024], dtype='int32').reshape([1]),
            # parameter_161
            paddle.uniform([16, 16, 49], dtype='float32', min=0, max=0.5),
            # constant_27
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            # constant_26
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            # constant_25
            paddle.to_tensor([1280], dtype='int32').reshape([1]),
            # parameter_135
            paddle.uniform([6, 49, 49], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([6, 49, 49], dtype='float32', min=0, max=0.5),
            # constant_24
            paddle.to_tensor([192], dtype='int32').reshape([1]),
            # parameter_93
            paddle.uniform([6, 49, 49], dtype='float32', min=0, max=0.5),
            # constant_23
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            # constant_22
            paddle.to_tensor([384], dtype='int32').reshape([1]),
            # constant_21
            paddle.to_tensor([512], dtype='int32').reshape([1]),
            # parameter_72
            paddle.uniform([8, 49, 196], dtype='float32', min=0, max=0.5),
            # constant_20
            paddle.to_tensor([16], dtype='int32').reshape([1]),
            # constant_19
            paddle.to_tensor([49], dtype='int32').reshape([1]),
            # constant_17
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            # constant_16
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            # constant_15
            paddle.to_tensor([0, 0], dtype='int64').reshape([2]),
            # constant_14
            paddle.to_tensor([14], dtype='int32').reshape([1]),
            # constant_13
            paddle.to_tensor([16, 64], dtype='int64').reshape([2]),
            # constant_12
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            # constant_11
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            # constant_10
            paddle.to_tensor([640], dtype='int32').reshape([1]),
            # parameter_46
            paddle.uniform([4, 196, 196], dtype='float32', min=0, max=0.5),
            # constant_9
            paddle.to_tensor([128], dtype='int32').reshape([1]),
            # constant_8
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([4, 196, 196], dtype='float32', min=0, max=0.5),
            # constant_18
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_7
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            # constant_6
            paddle.to_tensor([16, 16, 32], dtype='int64').reshape([3]),
            # constant_5
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            # constant_3
            paddle.to_tensor([256], dtype='int32').reshape([1]),
            # constant_2
            paddle.to_tensor([196], dtype='int32').reshape([1]),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_0
            paddle.uniform([16, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256, 128], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([256, 128], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([128, 640], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([192, 256], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([192, 256], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([192, 256], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256, 1280], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([1024, 384], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([384, 512], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([384, 512], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([384, 512], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([384, 512], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([384, 1000], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_30
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_245
            paddle.static.InputSpec(shape=[8, 16, 16], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[8, 16, 16], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[8, 16, 16], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[8, 16, 16], dtype='float32'),
            # constant_29
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_28
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_161
            paddle.static.InputSpec(shape=[16, 16, 49], dtype='float32'),
            # constant_27
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_26
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_25
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_135
            paddle.static.InputSpec(shape=[6, 49, 49], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[6, 49, 49], dtype='float32'),
            # constant_24
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_93
            paddle.static.InputSpec(shape=[6, 49, 49], dtype='float32'),
            # constant_23
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_22
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_21
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_72
            paddle.static.InputSpec(shape=[8, 49, 196], dtype='float32'),
            # constant_20
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_19
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_17
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_16
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_15
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_13
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_46
            paddle.static.InputSpec(shape=[4, 196, 196], dtype='float32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[4, 196, 196], dtype='float32'),
            # constant_18
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_6
            paddle.static.InputSpec(shape=[3], dtype='int64'),
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
            # parameter_0
            paddle.static.InputSpec(shape=[16, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 16, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256, 128], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[256, 128], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[128, 640], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[192, 256], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[192, 256], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[192, 256], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[256, 1280], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[1024, 384], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[384, 512], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[384, 512], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[384, 512], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[384, 512], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[384, 1000], dtype='float32'),
            # parameter_266
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