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
    return [656][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1508_0_0(self, constant_31, constant_30, constant_29, constant_28, constant_27, constant_26, constant_25, constant_24, constant_23, constant_22, constant_21, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, feed_0):

        # pd_op.conv2d: (-1x2x112x224xf32) <- (-1x3x224x224xf32, 2x3x3x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 1], [1, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2x112x224xf32, 2xf32, 2xf32, xf32, xf32, None) <- (-1x2x112x224xf32, 2xf32, 2xf32, 2xf32, 2xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x4x112x112xf32) <- (-1x2x112x224xf32, 4x1x1x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(batch_norm__0, parameter_5, [1, 2], [0, 1], 'EXPLICIT', 2, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x4x112x112xf32, 4xf32, 4xf32, xf32, xf32, None) <- (-1x4x112x112xf32, 4xf32, 4xf32, 4xf32, 4xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x4x112x112xf32)
        shape_0 = paddle._C_ops.shape(batch_norm__6)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, constant_2, constant_2, constant_3, constant_3]

        # pd_op.reshape_: (-1x2x2x112x112xf32, 0x-1x4x112x112xf32) <- (-1x4x112x112xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(batch_norm__6, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x112x112xf32) <- (-1x2x2x112x112xf32)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_0, constant_4, constant_3, constant_3]

        # pd_op.reshape_: (-1x4x112x112xf32, 0x-1x2x2x112x112xf32) <- (-1x2x2x112x112xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_0, combine_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu6: (-1x4x112x112xf32) <- (-1x4x112x112xf32)
        relu6_0 = paddle._C_ops.relu6(reshape__2)

        # pd_op.depthwise_conv2d: (-1x8x56x112xf32) <- (-1x4x112x112xf32, 8x1x3x1xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(relu6_0, parameter_10, [2, 1], [1, 0], 'EXPLICIT', 4, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x8x56x112xf32, 8xf32, 8xf32, xf32, xf32, None) <- (-1x8x56x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x16x56x56xf32) <- (-1x8x56x112xf32, 16x1x1x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(batch_norm__12, parameter_15, [1, 2], [0, 1], 'EXPLICIT', 8, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x16x56x56xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x16x56x56xf32) <- (-1x16x56x56xf32)
        relu6_1 = paddle._C_ops.relu6(batch_norm__18)

        # pd_op.shape: (4xi32) <- (-1x16x56x56xf32)
        shape_1 = paddle._C_ops.shape(relu6_1)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_1, constant_4, constant_4, constant_5, constant_5]

        # pd_op.reshape_: (-1x4x4x56x56xf32, 0x-1x16x56x56xf32) <- (-1x16x56x56xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu6_1, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x4x56x56xf32) <- (-1x4x4x56x56xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__4, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_1, constant_6, constant_5, constant_5]

        # pd_op.reshape_: (-1x16x56x56xf32, 0x-1x4x4x56x56xf32) <- (-1x4x4x56x56xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_1, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x8x56x56xf32) <- (-1x16x56x56xf32, 8x8x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(reshape__6, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x8x56x56xf32, 8xf32, 8xf32, xf32, xf32, None) <- (-1x8x56x56xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x8x56x56xf32)
        shape_2 = paddle._C_ops.shape(batch_norm__24)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x8x1x1xf32) <- (-1x8x56x56xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(batch_norm__24, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_4 = [slice_2, constant_8]

        # pd_op.reshape_: (-1x8xf32, 0x-1x8x1x1xf32) <- (-1x8x1x1xf32, [1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_0, combine_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x4xf32) <- (-1x8xf32, 8x4xf32)
        matmul_0 = paddle.matmul(reshape__8, parameter_25, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x4xf32) <- (-1x4xf32, 4xf32)
        add__0 = paddle._C_ops.add(matmul_0, parameter_26)

        # pd_op.relu_: (-1x4xf32) <- (-1x4xf32)
        relu__0 = paddle._C_ops.relu(add__0)

        # pd_op.matmul: (-1x16xf32) <- (-1x4xf32, 4x16xf32)
        matmul_1 = paddle.matmul(relu__0, parameter_27, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add__1 = paddle._C_ops.add(matmul_1, parameter_28)

        # pd_op.hardsigmoid: (-1x16xf32) <- (-1x16xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(add__1, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_2, constant_6, constant_9, constant_9]

        # pd_op.reshape_: (-1x16x1x1xf32, 0x-1x16xf32) <- (-1x16xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_0, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x16x1x1xf32) <- (-1x16x1x1xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(reshape__10, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x16x1x1xf32) <- (-1x16x1x1xf32, 1xf32)
        scale__1 = paddle._C_ops.scale(scale__0, constant_11, float('0'), True)

        # pd_op.index_select: (-1x8x56x56xf32) <- (-1x8x56x56xf32, 8xi64)
        index_select_0 = paddle._C_ops.index_select(batch_norm__24, parameter_29, 1)

        # pd_op.split_with_num: ([-1x8x1x1xf32, -1x8x1x1xf32]) <- (-1x16x1x1xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(scale__1, 2, constant_12)

        # builtin.slice: (-1x8x1x1xf32) <- ([-1x8x1x1xf32, -1x8x1x1xf32])
        slice_3 = split_with_num_0[0]

        # pd_op.scale_: (-1x8x1x1xf32) <- (-1x8x1x1xf32, 1xf32)
        scale__2 = paddle._C_ops.scale(slice_3, constant_10, float('1'), True)

        # builtin.slice: (-1x8x1x1xf32) <- ([-1x8x1x1xf32, -1x8x1x1xf32])
        slice_4 = split_with_num_0[1]

        # pd_op.scale_: (-1x8x1x1xf32) <- (-1x8x1x1xf32, 1xf32)
        scale__3 = paddle._C_ops.scale(slice_4, constant_10, float('0'), True)

        # pd_op.multiply_: (-1x8x56x56xf32) <- (-1x8x56x56xf32, -1x8x1x1xf32)
        multiply__0 = paddle._C_ops.multiply(batch_norm__24, scale__2)

        # pd_op.multiply_: (-1x8x56x56xf32) <- (-1x8x56x56xf32, -1x8x1x1xf32)
        multiply__1 = paddle._C_ops.multiply(index_select_0, scale__3)

        # pd_op.add_: (-1x8x56x56xf32) <- (-1x8x56x56xf32, -1x8x56x56xf32)
        add__2 = paddle._C_ops.add(multiply__0, multiply__1)

        # pd_op.shape: (4xi32) <- (-1x8x56x56xf32)
        shape_3 = paddle._C_ops.shape(add__2)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_3, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_5, constant_2, constant_4, constant_5, constant_5]

        # pd_op.reshape_: (-1x2x4x56x56xf32, 0x-1x8x56x56xf32) <- (-1x8x56x56xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__2, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x2x56x56xf32) <- (-1x2x4x56x56xf32)
        transpose_2 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_5, constant_8, constant_5, constant_5]

        # pd_op.reshape_: (-1x8x56x56xf32, 0x-1x4x2x56x56xf32) <- (-1x4x2x56x56xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_2, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x8x56x56xf32)
        shape_4 = paddle._C_ops.shape(reshape__14)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_4, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_6, constant_4, constant_2, constant_5, constant_5]

        # pd_op.reshape_: (-1x4x2x56x56xf32, 0x-1x8x56x56xf32) <- (-1x8x56x56xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__14, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x4x56x56xf32) <- (-1x4x2x56x56xf32)
        transpose_3 = paddle._C_ops.transpose(reshape__16, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_6, constant_8, constant_5, constant_5]

        # pd_op.reshape_: (-1x8x56x56xf32, 0x-1x2x4x56x56xf32) <- (-1x2x4x56x56xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_3, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x16x28x56xf32) <- (-1x8x56x56xf32, 16x1x3x1xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(reshape__18, parameter_30, [2, 1], [1, 0], 'EXPLICIT', 8, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x56xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x28x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x32x28x28xf32) <- (-1x16x28x56xf32, 32x1x1x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(batch_norm__30, parameter_35, [1, 2], [0, 1], 'EXPLICIT', 16, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x32x28x28xf32)
        shape_5 = paddle._C_ops.shape(batch_norm__36)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_5, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x32x1x1xf32) <- (-1x32x28x28xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(batch_norm__36, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_10 = [slice_7, constant_13]

        # pd_op.reshape_: (-1x32xf32, 0x-1x32x1x1xf32) <- (-1x32x1x1xf32, [1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_1, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x4xf32) <- (-1x32xf32, 32x4xf32)
        matmul_2 = paddle.matmul(reshape__20, parameter_40, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x4xf32) <- (-1x4xf32, 4xf32)
        add__3 = paddle._C_ops.add(matmul_2, parameter_41)

        # pd_op.relu_: (-1x4xf32) <- (-1x4xf32)
        relu__1 = paddle._C_ops.relu(add__3)

        # pd_op.matmul: (-1x128xf32) <- (-1x4xf32, 4x128xf32)
        matmul_3 = paddle.matmul(relu__1, parameter_42, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x128xf32) <- (-1x128xf32, 128xf32)
        add__4 = paddle._C_ops.add(matmul_3, parameter_43)

        # pd_op.hardsigmoid: (-1x128xf32) <- (-1x128xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(add__4, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_7, constant_14, constant_9, constant_9]

        # pd_op.reshape_: (-1x128x1x1xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_1, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__4 = paddle._C_ops.scale(reshape__22, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__5 = paddle._C_ops.scale(scale__4, constant_11, float('0'), True)

        # pd_op.index_select: (-1x32x28x28xf32) <- (-1x32x28x28xf32, 32xi64)
        index_select_1 = paddle._C_ops.index_select(batch_norm__36, parameter_44, 1)

        # pd_op.split_with_num: ([-1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32]) <- (-1x128x1x1xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(scale__5, 4, constant_12)

        # builtin.slice: (-1x32x1x1xf32) <- ([-1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32])
        slice_8 = split_with_num_1[0]

        # pd_op.scale_: (-1x32x1x1xf32) <- (-1x32x1x1xf32, 1xf32)
        scale__6 = paddle._C_ops.scale(slice_8, constant_10, float('1'), True)

        # builtin.slice: (-1x32x1x1xf32) <- ([-1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32])
        slice_9 = split_with_num_1[2]

        # pd_op.scale_: (-1x32x1x1xf32) <- (-1x32x1x1xf32, 1xf32)
        scale__7 = paddle._C_ops.scale(slice_9, constant_10, float('1'), True)

        # builtin.slice: (-1x32x1x1xf32) <- ([-1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32])
        slice_10 = split_with_num_1[1]

        # pd_op.scale_: (-1x32x1x1xf32) <- (-1x32x1x1xf32, 1xf32)
        scale__8 = paddle._C_ops.scale(slice_10, constant_10, float('0'), True)

        # builtin.slice: (-1x32x1x1xf32) <- ([-1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32, -1x32x1x1xf32])
        slice_11 = split_with_num_1[3]

        # pd_op.scale_: (-1x32x1x1xf32) <- (-1x32x1x1xf32, 1xf32)
        scale__9 = paddle._C_ops.scale(slice_11, constant_10, float('0'), True)

        # pd_op.multiply: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x1x1xf32)
        multiply_0 = batch_norm__36 * scale__6

        # pd_op.multiply: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x1x1xf32)
        multiply_1 = index_select_1 * scale__8

        # pd_op.add_: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x28x28xf32)
        add__5 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.multiply_: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x1x1xf32)
        multiply__2 = paddle._C_ops.multiply(batch_norm__36, scale__7)

        # pd_op.multiply_: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x1x1xf32)
        multiply__3 = paddle._C_ops.multiply(index_select_1, scale__9)

        # pd_op.add_: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x28x28xf32)
        add__6 = paddle._C_ops.add(multiply__2, multiply__3)

        # pd_op.maximum: (-1x32x28x28xf32) <- (-1x32x28x28xf32, -1x32x28x28xf32)
        maximum_0 = paddle.maximum(add__5, add__6)

        # pd_op.shape: (4xi32) <- (-1x32x28x28xf32)
        shape_6 = paddle._C_ops.shape(maximum_0)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_6, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_12, constant_8, constant_4, constant_15, constant_15]

        # pd_op.reshape_: (-1x8x4x28x28xf32, 0x-1x32x28x28xf32) <- (-1x32x28x28xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape(maximum_0, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x8x28x28xf32) <- (-1x8x4x28x28xf32)
        transpose_4 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_12, constant_13, constant_15, constant_15]

        # pd_op.reshape_: (-1x32x28x28xf32, 0x-1x4x8x28x28xf32) <- (-1x4x8x28x28xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_4, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x32x28x28xf32)
        shape_7 = paddle._C_ops.shape(reshape__26)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_7, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_13, constant_6, constant_2, constant_15, constant_15]

        # pd_op.reshape_: (-1x16x2x28x28xf32, 0x-1x32x28x28xf32) <- (-1x32x28x28xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__26, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x16x28x28xf32) <- (-1x16x2x28x28xf32)
        transpose_5 = paddle._C_ops.transpose(reshape__28, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_15 = [slice_13, constant_13, constant_15, constant_15]

        # pd_op.reshape_: (-1x32x28x28xf32, 0x-1x2x16x28x28xf32) <- (-1x2x16x28x28xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_5, combine_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x12x28x28xf32) <- (-1x32x28x28xf32, 12x8x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(reshape__30, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.batch_norm_: (-1x12x28x28xf32, 12xf32, 12xf32, xf32, xf32, None) <- (-1x12x28x28xf32, 12xf32, 12xf32, 12xf32, 12xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x12x28x28xf32)
        shape_8 = paddle._C_ops.shape(batch_norm__42)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_8, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x12x1x1xf32) <- (-1x12x28x28xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(batch_norm__42, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_16 = [slice_14, constant_16]

        # pd_op.reshape_: (-1x12xf32, 0x-1x12x1x1xf32) <- (-1x12x1x1xf32, [1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_2, combine_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x4xf32) <- (-1x12xf32, 12x4xf32)
        matmul_4 = paddle.matmul(reshape__32, parameter_50, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x4xf32) <- (-1x4xf32, 4xf32)
        add__7 = paddle._C_ops.add(matmul_4, parameter_51)

        # pd_op.relu_: (-1x4xf32) <- (-1x4xf32)
        relu__2 = paddle._C_ops.relu(add__7)

        # pd_op.matmul: (-1x24xf32) <- (-1x4xf32, 4x24xf32)
        matmul_5 = paddle.matmul(relu__2, parameter_52, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x24xf32) <- (-1x24xf32, 24xf32)
        add__8 = paddle._C_ops.add(matmul_5, parameter_53)

        # pd_op.hardsigmoid: (-1x24xf32) <- (-1x24xf32)
        hardsigmoid_2 = paddle._C_ops.hardsigmoid(add__8, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_14, constant_17, constant_9, constant_9]

        # pd_op.reshape_: (-1x24x1x1xf32, 0x-1x24xf32) <- (-1x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_2, combine_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1xf32)
        scale__10 = paddle._C_ops.scale(reshape__34, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1xf32)
        scale__11 = paddle._C_ops.scale(scale__10, constant_11, float('0'), True)

        # pd_op.index_select: (-1x12x28x28xf32) <- (-1x12x28x28xf32, 12xi64)
        index_select_2 = paddle._C_ops.index_select(batch_norm__42, parameter_54, 1)

        # pd_op.split_with_num: ([-1x12x1x1xf32, -1x12x1x1xf32]) <- (-1x24x1x1xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(scale__11, 2, constant_12)

        # builtin.slice: (-1x12x1x1xf32) <- ([-1x12x1x1xf32, -1x12x1x1xf32])
        slice_15 = split_with_num_2[0]

        # pd_op.scale_: (-1x12x1x1xf32) <- (-1x12x1x1xf32, 1xf32)
        scale__12 = paddle._C_ops.scale(slice_15, constant_10, float('1'), True)

        # builtin.slice: (-1x12x1x1xf32) <- ([-1x12x1x1xf32, -1x12x1x1xf32])
        slice_16 = split_with_num_2[1]

        # pd_op.scale_: (-1x12x1x1xf32) <- (-1x12x1x1xf32, 1xf32)
        scale__13 = paddle._C_ops.scale(slice_16, constant_10, float('0'), True)

        # pd_op.multiply_: (-1x12x28x28xf32) <- (-1x12x28x28xf32, -1x12x1x1xf32)
        multiply__4 = paddle._C_ops.multiply(batch_norm__42, scale__12)

        # pd_op.multiply_: (-1x12x28x28xf32) <- (-1x12x28x28xf32, -1x12x1x1xf32)
        multiply__5 = paddle._C_ops.multiply(index_select_2, scale__13)

        # pd_op.add_: (-1x12x28x28xf32) <- (-1x12x28x28xf32, -1x12x28x28xf32)
        add__9 = paddle._C_ops.add(multiply__4, multiply__5)

        # pd_op.shape: (4xi32) <- (-1x12x28x28xf32)
        shape_9 = paddle._C_ops.shape(add__9)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_9, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_17, constant_4, constant_18, constant_15, constant_15]

        # pd_op.reshape_: (-1x4x3x28x28xf32, 0x-1x12x28x28xf32) <- (-1x12x28x28xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__9, combine_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3x4x28x28xf32) <- (-1x4x3x28x28xf32)
        transpose_6 = paddle._C_ops.transpose(reshape__36, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_17, constant_16, constant_15, constant_15]

        # pd_op.reshape_: (-1x12x28x28xf32, 0x-1x3x4x28x28xf32) <- (-1x3x4x28x28xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_6, combine_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x12x28x28xf32)
        shape_10 = paddle._C_ops.shape(reshape__38)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(shape_10, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_18, constant_19, constant_2, constant_15, constant_15]

        # pd_op.reshape_: (-1x6x2x28x28xf32, 0x-1x12x28x28xf32) <- (-1x12x28x28xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__38, combine_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x6x28x28xf32) <- (-1x6x2x28x28xf32)
        transpose_7 = paddle._C_ops.transpose(reshape__40, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_18, constant_16, constant_15, constant_15]

        # pd_op.reshape_: (-1x12x28x28xf32, 0x-1x2x6x28x28xf32) <- (-1x2x6x28x28xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_7, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x24x14x28xf32) <- (-1x12x28x28xf32, 24x1x5x1xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(reshape__42, parameter_55, [2, 1], [2, 0], 'EXPLICIT', 12, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x24x14x28xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x14x28xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x48x14x14xf32) <- (-1x24x14x28xf32, 48x1x1x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(batch_norm__48, parameter_60, [1, 2], [0, 2], 'EXPLICIT', 24, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x14x14xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x14x14xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x48x14x14xf32)
        shape_11 = paddle._C_ops.shape(batch_norm__54)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(shape_11, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x48x1x1xf32) <- (-1x48x14x14xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(batch_norm__54, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_22 = [slice_19, constant_20]

        # pd_op.reshape_: (-1x48xf32, 0x-1x48x1x1xf32) <- (-1x48x1x1xf32, [1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_3, combine_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8xf32) <- (-1x48xf32, 48x8xf32)
        matmul_6 = paddle.matmul(reshape__44, parameter_65, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x8xf32) <- (-1x8xf32, 8xf32)
        add__10 = paddle._C_ops.add(matmul_6, parameter_66)

        # pd_op.relu_: (-1x8xf32) <- (-1x8xf32)
        relu__3 = paddle._C_ops.relu(add__10)

        # pd_op.matmul: (-1x192xf32) <- (-1x8xf32, 8x192xf32)
        matmul_7 = paddle.matmul(relu__3, parameter_67, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x192xf32) <- (-1x192xf32, 192xf32)
        add__11 = paddle._C_ops.add(matmul_7, parameter_68)

        # pd_op.hardsigmoid: (-1x192xf32) <- (-1x192xf32)
        hardsigmoid_3 = paddle._C_ops.hardsigmoid(add__11, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_19, constant_21, constant_9, constant_9]

        # pd_op.reshape_: (-1x192x1x1xf32, 0x-1x192xf32) <- (-1x192xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_3, combine_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1xf32)
        scale__14 = paddle._C_ops.scale(reshape__46, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1xf32)
        scale__15 = paddle._C_ops.scale(scale__14, constant_11, float('0'), True)

        # pd_op.index_select: (-1x48x14x14xf32) <- (-1x48x14x14xf32, 48xi64)
        index_select_3 = paddle._C_ops.index_select(batch_norm__54, parameter_69, 1)

        # pd_op.split_with_num: ([-1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32]) <- (-1x192x1x1xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(scale__15, 4, constant_12)

        # builtin.slice: (-1x48x1x1xf32) <- ([-1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32])
        slice_20 = split_with_num_3[0]

        # pd_op.scale_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1xf32)
        scale__16 = paddle._C_ops.scale(slice_20, constant_10, float('1'), True)

        # builtin.slice: (-1x48x1x1xf32) <- ([-1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32])
        slice_21 = split_with_num_3[2]

        # pd_op.scale_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1xf32)
        scale__17 = paddle._C_ops.scale(slice_21, constant_10, float('1'), True)

        # builtin.slice: (-1x48x1x1xf32) <- ([-1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32])
        slice_22 = split_with_num_3[1]

        # pd_op.scale_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1xf32)
        scale__18 = paddle._C_ops.scale(slice_22, constant_10, float('0'), True)

        # builtin.slice: (-1x48x1x1xf32) <- ([-1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32, -1x48x1x1xf32])
        slice_23 = split_with_num_3[3]

        # pd_op.scale_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1xf32)
        scale__19 = paddle._C_ops.scale(slice_23, constant_10, float('0'), True)

        # pd_op.multiply: (-1x48x14x14xf32) <- (-1x48x14x14xf32, -1x48x1x1xf32)
        multiply_2 = batch_norm__54 * scale__16

        # pd_op.multiply: (-1x48x14x14xf32) <- (-1x48x14x14xf32, -1x48x1x1xf32)
        multiply_3 = index_select_3 * scale__18

        # pd_op.add_: (-1x48x14x14xf32) <- (-1x48x14x14xf32, -1x48x14x14xf32)
        add__12 = paddle._C_ops.add(multiply_2, multiply_3)

        # pd_op.multiply_: (-1x48x14x14xf32) <- (-1x48x14x14xf32, -1x48x1x1xf32)
        multiply__6 = paddle._C_ops.multiply(batch_norm__54, scale__17)

        # pd_op.multiply_: (-1x48x14x14xf32) <- (-1x48x14x14xf32, -1x48x1x1xf32)
        multiply__7 = paddle._C_ops.multiply(index_select_3, scale__19)

        # pd_op.add_: (-1x48x14x14xf32) <- (-1x48x14x14xf32, -1x48x14x14xf32)
        add__13 = paddle._C_ops.add(multiply__6, multiply__7)

        # pd_op.maximum: (-1x48x14x14xf32) <- (-1x48x14x14xf32, -1x48x14x14xf32)
        maximum_1 = paddle.maximum(add__12, add__13)

        # pd_op.shape: (4xi32) <- (-1x48x14x14xf32)
        shape_12 = paddle._C_ops.shape(maximum_1)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_12, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_24, constant_16, constant_4, constant_22, constant_22]

        # pd_op.reshape_: (-1x12x4x14x14xf32, 0x-1x48x14x14xf32) <- (-1x48x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape(maximum_1, combine_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x12x14x14xf32) <- (-1x12x4x14x14xf32)
        transpose_8 = paddle._C_ops.transpose(reshape__48, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_25 = [slice_24, constant_20, constant_22, constant_22]

        # pd_op.reshape_: (-1x48x14x14xf32, 0x-1x4x12x14x14xf32) <- (-1x4x12x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_8, combine_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x48x14x14xf32)
        shape_13 = paddle._C_ops.shape(reshape__50)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_13, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_25, constant_17, constant_2, constant_22, constant_22]

        # pd_op.reshape_: (-1x24x2x14x14xf32, 0x-1x48x14x14xf32) <- (-1x48x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__50, combine_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x24x14x14xf32) <- (-1x24x2x14x14xf32)
        transpose_9 = paddle._C_ops.transpose(reshape__52, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_27 = [slice_25, constant_20, constant_22, constant_22]

        # pd_op.reshape_: (-1x48x14x14xf32, 0x-1x2x24x14x14xf32) <- (-1x2x24x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_9, combine_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x16x14x14xf32) <- (-1x48x14x14xf32, 16x12x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(reshape__54, parameter_70, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.batch_norm_: (-1x16x14x14xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x14x14xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x16x14x14xf32)
        shape_14 = paddle._C_ops.shape(batch_norm__60)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(shape_14, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x16x1x1xf32) <- (-1x16x14x14xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(batch_norm__60, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_28 = [slice_26, constant_6]

        # pd_op.reshape_: (-1x16xf32, 0x-1x16x1x1xf32) <- (-1x16x1x1xf32, [1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_4, combine_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x4xf32) <- (-1x16xf32, 16x4xf32)
        matmul_8 = paddle.matmul(reshape__56, parameter_75, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x4xf32) <- (-1x4xf32, 4xf32)
        add__14 = paddle._C_ops.add(matmul_8, parameter_76)

        # pd_op.relu_: (-1x4xf32) <- (-1x4xf32)
        relu__4 = paddle._C_ops.relu(add__14)

        # pd_op.matmul: (-1x32xf32) <- (-1x4xf32, 4x32xf32)
        matmul_9 = paddle.matmul(relu__4, parameter_77, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add__15 = paddle._C_ops.add(matmul_9, parameter_78)

        # pd_op.hardsigmoid: (-1x32xf32) <- (-1x32xf32)
        hardsigmoid_4 = paddle._C_ops.hardsigmoid(add__15, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_26, constant_13, constant_9, constant_9]

        # pd_op.reshape_: (-1x32x1x1xf32, 0x-1x32xf32) <- (-1x32xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_4, combine_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x32x1x1xf32) <- (-1x32x1x1xf32, 1xf32)
        scale__20 = paddle._C_ops.scale(reshape__58, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x32x1x1xf32) <- (-1x32x1x1xf32, 1xf32)
        scale__21 = paddle._C_ops.scale(scale__20, constant_11, float('0'), True)

        # pd_op.index_select: (-1x16x14x14xf32) <- (-1x16x14x14xf32, 16xi64)
        index_select_4 = paddle._C_ops.index_select(batch_norm__60, parameter_79, 1)

        # pd_op.split_with_num: ([-1x16x1x1xf32, -1x16x1x1xf32]) <- (-1x32x1x1xf32, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(scale__21, 2, constant_12)

        # builtin.slice: (-1x16x1x1xf32) <- ([-1x16x1x1xf32, -1x16x1x1xf32])
        slice_27 = split_with_num_4[0]

        # pd_op.scale_: (-1x16x1x1xf32) <- (-1x16x1x1xf32, 1xf32)
        scale__22 = paddle._C_ops.scale(slice_27, constant_10, float('1'), True)

        # builtin.slice: (-1x16x1x1xf32) <- ([-1x16x1x1xf32, -1x16x1x1xf32])
        slice_28 = split_with_num_4[1]

        # pd_op.scale_: (-1x16x1x1xf32) <- (-1x16x1x1xf32, 1xf32)
        scale__23 = paddle._C_ops.scale(slice_28, constant_10, float('0'), True)

        # pd_op.multiply_: (-1x16x14x14xf32) <- (-1x16x14x14xf32, -1x16x1x1xf32)
        multiply__8 = paddle._C_ops.multiply(batch_norm__60, scale__22)

        # pd_op.multiply_: (-1x16x14x14xf32) <- (-1x16x14x14xf32, -1x16x1x1xf32)
        multiply__9 = paddle._C_ops.multiply(index_select_4, scale__23)

        # pd_op.add_: (-1x16x14x14xf32) <- (-1x16x14x14xf32, -1x16x14x14xf32)
        add__16 = paddle._C_ops.add(multiply__8, multiply__9)

        # pd_op.shape: (4xi32) <- (-1x16x14x14xf32)
        shape_15 = paddle._C_ops.shape(add__16)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_15, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_30 = [slice_29, constant_4, constant_4, constant_22, constant_22]

        # pd_op.reshape_: (-1x4x4x14x14xf32, 0x-1x16x14x14xf32) <- (-1x16x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__16, combine_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x4x14x14xf32) <- (-1x4x4x14x14xf32)
        transpose_10 = paddle._C_ops.transpose(reshape__60, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [slice_29, constant_6, constant_22, constant_22]

        # pd_op.reshape_: (-1x16x14x14xf32, 0x-1x4x4x14x14xf32) <- (-1x4x4x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_10, combine_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x16x14x14xf32)
        shape_16 = paddle._C_ops.shape(reshape__62)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(shape_16, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_30, constant_8, constant_2, constant_22, constant_22]

        # pd_op.reshape_: (-1x8x2x14x14xf32, 0x-1x16x14x14xf32) <- (-1x16x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__62, combine_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x8x14x14xf32) <- (-1x8x2x14x14xf32)
        transpose_11 = paddle._C_ops.transpose(reshape__64, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_30, constant_6, constant_22, constant_22]

        # pd_op.reshape_: (-1x16x14x14xf32, 0x-1x2x8x14x14xf32) <- (-1x2x8x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_11, combine_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x16x14x14xf32, 64x4x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(reshape__66, parameter_80, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x64x14x14xf32)
        shape_17 = paddle._C_ops.shape(batch_norm__66)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(shape_17, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x64x1x1xf32) <- (-1x64x14x14xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(batch_norm__66, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_34 = [slice_31, constant_23]

        # pd_op.reshape_: (-1x64xf32, 0x-1x64x1x1xf32) <- (-1x64x1x1xf32, [1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_5, combine_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8xf32) <- (-1x64xf32, 64x8xf32)
        matmul_10 = paddle.matmul(reshape__68, parameter_85, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x8xf32) <- (-1x8xf32, 8xf32)
        add__17 = paddle._C_ops.add(matmul_10, parameter_86)

        # pd_op.relu_: (-1x8xf32) <- (-1x8xf32)
        relu__5 = paddle._C_ops.relu(add__17)

        # pd_op.matmul: (-1x256xf32) <- (-1x8xf32, 8x256xf32)
        matmul_11 = paddle.matmul(relu__5, parameter_87, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__18 = paddle._C_ops.add(matmul_11, parameter_88)

        # pd_op.hardsigmoid: (-1x256xf32) <- (-1x256xf32)
        hardsigmoid_5 = paddle._C_ops.hardsigmoid(add__18, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_31, constant_24, constant_9, constant_9]

        # pd_op.reshape_: (-1x256x1x1xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_5, combine_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__24 = paddle._C_ops.scale(reshape__70, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__25 = paddle._C_ops.scale(scale__24, constant_11, float('0'), True)

        # pd_op.index_select: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64xi64)
        index_select_5 = paddle._C_ops.index_select(batch_norm__66, parameter_89, 1)

        # pd_op.split_with_num: ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32]) <- (-1x256x1x1xf32, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(scale__25, 4, constant_12)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32])
        slice_32 = split_with_num_5[0]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__26 = paddle._C_ops.scale(slice_32, constant_10, float('1'), True)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32])
        slice_33 = split_with_num_5[2]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__27 = paddle._C_ops.scale(slice_33, constant_10, float('1'), True)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32])
        slice_34 = split_with_num_5[1]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__28 = paddle._C_ops.scale(slice_34, constant_10, float('0'), True)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32])
        slice_35 = split_with_num_5[3]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__29 = paddle._C_ops.scale(slice_35, constant_10, float('0'), True)

        # pd_op.multiply: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x1x1xf32)
        multiply_4 = batch_norm__66 * scale__26

        # pd_op.multiply: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x1x1xf32)
        multiply_5 = index_select_5 * scale__28

        # pd_op.add_: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        add__19 = paddle._C_ops.add(multiply_4, multiply_5)

        # pd_op.multiply_: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x1x1xf32)
        multiply__10 = paddle._C_ops.multiply(batch_norm__66, scale__27)

        # pd_op.multiply_: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x1x1xf32)
        multiply__11 = paddle._C_ops.multiply(index_select_5, scale__29)

        # pd_op.add_: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        add__20 = paddle._C_ops.add(multiply__10, multiply__11)

        # pd_op.maximum: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        maximum_2 = paddle.maximum(add__19, add__20)

        # pd_op.shape: (4xi32) <- (-1x64x14x14xf32)
        shape_18 = paddle._C_ops.shape(maximum_2)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(shape_18, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_36 = [slice_36, constant_4, constant_6, constant_22, constant_22]

        # pd_op.reshape_: (-1x4x16x14x14xf32, 0x-1x64x14x14xf32) <- (-1x64x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape(maximum_2, combine_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x4x14x14xf32) <- (-1x4x16x14x14xf32)
        transpose_12 = paddle._C_ops.transpose(reshape__72, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_36, constant_23, constant_22, constant_22]

        # pd_op.reshape_: (-1x64x14x14xf32, 0x-1x16x4x14x14xf32) <- (-1x16x4x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_12, combine_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x1x5x1xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(reshape__74, parameter_90, [1, 1], [2, 0], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_7, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64x1x1x5xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(batch_norm__72, parameter_95, [1, 1], [0, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_8, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x64x14x14xf32)
        shape_19 = paddle._C_ops.shape(batch_norm__78)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(shape_19, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x64x1x1xf32) <- (-1x64x14x14xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(batch_norm__78, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_38 = [slice_37, constant_23]

        # pd_op.reshape_: (-1x64xf32, 0x-1x64x1x1xf32) <- (-1x64x1x1xf32, [1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_6, combine_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8xf32) <- (-1x64xf32, 64x8xf32)
        matmul_12 = paddle.matmul(reshape__76, parameter_100, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x8xf32) <- (-1x8xf32, 8xf32)
        add__21 = paddle._C_ops.add(matmul_12, parameter_101)

        # pd_op.relu_: (-1x8xf32) <- (-1x8xf32)
        relu__6 = paddle._C_ops.relu(add__21)

        # pd_op.matmul: (-1x256xf32) <- (-1x8xf32, 8x256xf32)
        matmul_13 = paddle.matmul(relu__6, parameter_102, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__22 = paddle._C_ops.add(matmul_13, parameter_103)

        # pd_op.hardsigmoid: (-1x256xf32) <- (-1x256xf32)
        hardsigmoid_6 = paddle._C_ops.hardsigmoid(add__22, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_39 = [slice_37, constant_24, constant_9, constant_9]

        # pd_op.reshape_: (-1x256x1x1xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_6, combine_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__30 = paddle._C_ops.scale(reshape__78, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__31 = paddle._C_ops.scale(scale__30, constant_11, float('0'), True)

        # pd_op.index_select: (-1x64x14x14xf32) <- (-1x64x14x14xf32, 64xi64)
        index_select_6 = paddle._C_ops.index_select(batch_norm__78, parameter_104, 1)

        # pd_op.split_with_num: ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32]) <- (-1x256x1x1xf32, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(scale__31, 4, constant_12)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32])
        slice_38 = split_with_num_6[0]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__32 = paddle._C_ops.scale(slice_38, constant_10, float('1'), True)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32])
        slice_39 = split_with_num_6[2]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__33 = paddle._C_ops.scale(slice_39, constant_10, float('1'), True)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32])
        slice_40 = split_with_num_6[1]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__34 = paddle._C_ops.scale(slice_40, constant_10, float('0'), True)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32, -1x64x1x1xf32])
        slice_41 = split_with_num_6[3]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__35 = paddle._C_ops.scale(slice_41, constant_10, float('0'), True)

        # pd_op.multiply: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x1x1xf32)
        multiply_6 = batch_norm__78 * scale__32

        # pd_op.multiply: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x1x1xf32)
        multiply_7 = index_select_6 * scale__34

        # pd_op.add_: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        add__23 = paddle._C_ops.add(multiply_6, multiply_7)

        # pd_op.multiply_: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x1x1xf32)
        multiply__12 = paddle._C_ops.multiply(batch_norm__78, scale__33)

        # pd_op.multiply_: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x1x1xf32)
        multiply__13 = paddle._C_ops.multiply(index_select_6, scale__35)

        # pd_op.add_: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        add__24 = paddle._C_ops.add(multiply__12, multiply__13)

        # pd_op.maximum: (-1x64x14x14xf32) <- (-1x64x14x14xf32, -1x64x14x14xf32)
        maximum_3 = paddle.maximum(add__23, add__24)

        # pd_op.shape: (4xi32) <- (-1x64x14x14xf32)
        shape_20 = paddle._C_ops.shape(maximum_3)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(shape_20, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_40 = [slice_42, constant_6, constant_4, constant_22, constant_22]

        # pd_op.reshape_: (-1x16x4x14x14xf32, 0x-1x64x14x14xf32) <- (-1x64x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape(maximum_3, combine_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x14x14xf32) <- (-1x16x4x14x14xf32)
        transpose_13 = paddle._C_ops.transpose(reshape__80, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_42, constant_23, constant_22, constant_22]

        # pd_op.reshape_: (-1x64x14x14xf32, 0x-1x4x16x14x14xf32) <- (-1x4x16x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_13, combine_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x32x14x14xf32) <- (-1x64x14x14xf32, 32x16x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(reshape__82, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.batch_norm_: (-1x32x14x14xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x14x14xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x32x14x14xf32)
        shape_21 = paddle._C_ops.shape(batch_norm__84)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(shape_21, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x32x1x1xf32) <- (-1x32x14x14xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(batch_norm__84, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_42 = [slice_43, constant_13]

        # pd_op.reshape_: (-1x32xf32, 0x-1x32x1x1xf32) <- (-1x32x1x1xf32, [1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_7, combine_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8xf32) <- (-1x32xf32, 32x8xf32)
        matmul_14 = paddle.matmul(reshape__84, parameter_110, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x8xf32) <- (-1x8xf32, 8xf32)
        add__25 = paddle._C_ops.add(matmul_14, parameter_111)

        # pd_op.relu_: (-1x8xf32) <- (-1x8xf32)
        relu__7 = paddle._C_ops.relu(add__25)

        # pd_op.matmul: (-1x64xf32) <- (-1x8xf32, 8x64xf32)
        matmul_15 = paddle.matmul(relu__7, parameter_112, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__26 = paddle._C_ops.add(matmul_15, parameter_113)

        # pd_op.hardsigmoid: (-1x64xf32) <- (-1x64xf32)
        hardsigmoid_7 = paddle._C_ops.hardsigmoid(add__26, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_43 = [slice_43, constant_23, constant_9, constant_9]

        # pd_op.reshape_: (-1x64x1x1xf32, 0x-1x64xf32) <- (-1x64xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_7, combine_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__36 = paddle._C_ops.scale(reshape__86, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__37 = paddle._C_ops.scale(scale__36, constant_11, float('0'), True)

        # pd_op.index_select: (-1x32x14x14xf32) <- (-1x32x14x14xf32, 32xi64)
        index_select_7 = paddle._C_ops.index_select(batch_norm__84, parameter_114, 1)

        # pd_op.split_with_num: ([-1x32x1x1xf32, -1x32x1x1xf32]) <- (-1x64x1x1xf32, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(scale__37, 2, constant_12)

        # builtin.slice: (-1x32x1x1xf32) <- ([-1x32x1x1xf32, -1x32x1x1xf32])
        slice_44 = split_with_num_7[0]

        # pd_op.scale_: (-1x32x1x1xf32) <- (-1x32x1x1xf32, 1xf32)
        scale__38 = paddle._C_ops.scale(slice_44, constant_10, float('1'), True)

        # builtin.slice: (-1x32x1x1xf32) <- ([-1x32x1x1xf32, -1x32x1x1xf32])
        slice_45 = split_with_num_7[1]

        # pd_op.scale_: (-1x32x1x1xf32) <- (-1x32x1x1xf32, 1xf32)
        scale__39 = paddle._C_ops.scale(slice_45, constant_10, float('0'), True)

        # pd_op.multiply_: (-1x32x14x14xf32) <- (-1x32x14x14xf32, -1x32x1x1xf32)
        multiply__14 = paddle._C_ops.multiply(batch_norm__84, scale__38)

        # pd_op.multiply_: (-1x32x14x14xf32) <- (-1x32x14x14xf32, -1x32x1x1xf32)
        multiply__15 = paddle._C_ops.multiply(index_select_7, scale__39)

        # pd_op.add_: (-1x32x14x14xf32) <- (-1x32x14x14xf32, -1x32x14x14xf32)
        add__27 = paddle._C_ops.add(multiply__14, multiply__15)

        # pd_op.shape: (4xi32) <- (-1x32x14x14xf32)
        shape_22 = paddle._C_ops.shape(add__27)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(shape_22, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_46, constant_4, constant_8, constant_22, constant_22]

        # pd_op.reshape_: (-1x4x8x14x14xf32, 0x-1x32x14x14xf32) <- (-1x32x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__27, combine_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x4x14x14xf32) <- (-1x4x8x14x14xf32)
        transpose_14 = paddle._C_ops.transpose(reshape__88, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [slice_46, constant_13, constant_22, constant_22]

        # pd_op.reshape_: (-1x32x14x14xf32, 0x-1x8x4x14x14xf32) <- (-1x8x4x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_14, combine_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x32x14x14xf32)
        shape_23 = paddle._C_ops.shape(reshape__90)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(shape_23, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_46 = [slice_47, constant_6, constant_2, constant_22, constant_22]

        # pd_op.reshape_: (-1x16x2x14x14xf32, 0x-1x32x14x14xf32) <- (-1x32x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__90, combine_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x16x14x14xf32) <- (-1x16x2x14x14xf32)
        transpose_15 = paddle._C_ops.transpose(reshape__92, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_47 = [slice_47, constant_13, constant_22, constant_22]

        # pd_op.reshape_: (-1x32x14x14xf32, 0x-1x2x16x14x14xf32) <- (-1x2x16x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_15, combine_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x32x14x14xf32, 128x4x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(reshape__94, parameter_115, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 8, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x128x14x14xf32)
        shape_24 = paddle._C_ops.shape(batch_norm__90)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(shape_24, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x128x1x1xf32) <- (-1x128x14x14xf32, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(batch_norm__90, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_48 = [slice_48, constant_14]

        # pd_op.reshape_: (-1x128xf32, 0x-1x128x1x1xf32) <- (-1x128x1x1xf32, [1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_8, combine_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16xf32) <- (-1x128xf32, 128x16xf32)
        matmul_16 = paddle.matmul(reshape__96, parameter_120, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add__28 = paddle._C_ops.add(matmul_16, parameter_121)

        # pd_op.relu_: (-1x16xf32) <- (-1x16xf32)
        relu__8 = paddle._C_ops.relu(add__28)

        # pd_op.matmul: (-1x512xf32) <- (-1x16xf32, 16x512xf32)
        matmul_17 = paddle.matmul(relu__8, parameter_122, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x512xf32) <- (-1x512xf32, 512xf32)
        add__29 = paddle._C_ops.add(matmul_17, parameter_123)

        # pd_op.hardsigmoid: (-1x512xf32) <- (-1x512xf32)
        hardsigmoid_8 = paddle._C_ops.hardsigmoid(add__29, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_49 = [slice_48, constant_25, constant_9, constant_9]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_8, combine_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1xf32)
        scale__40 = paddle._C_ops.scale(reshape__98, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1xf32)
        scale__41 = paddle._C_ops.scale(scale__40, constant_11, float('0'), True)

        # pd_op.index_select: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128xi64)
        index_select_8 = paddle._C_ops.index_select(batch_norm__90, parameter_124, 1)

        # pd_op.split_with_num: ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(scale__41, 4, constant_12)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32])
        slice_49 = split_with_num_8[0]

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__42 = paddle._C_ops.scale(slice_49, constant_10, float('1'), True)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32])
        slice_50 = split_with_num_8[2]

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__43 = paddle._C_ops.scale(slice_50, constant_10, float('1'), True)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32])
        slice_51 = split_with_num_8[1]

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__44 = paddle._C_ops.scale(slice_51, constant_10, float('0'), True)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32])
        slice_52 = split_with_num_8[3]

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__45 = paddle._C_ops.scale(slice_52, constant_10, float('0'), True)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x1x1xf32)
        multiply_8 = batch_norm__90 * scale__42

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x1x1xf32)
        multiply_9 = index_select_8 * scale__44

        # pd_op.add_: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add__30 = paddle._C_ops.add(multiply_8, multiply_9)

        # pd_op.multiply_: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x1x1xf32)
        multiply__16 = paddle._C_ops.multiply(batch_norm__90, scale__43)

        # pd_op.multiply_: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x1x1xf32)
        multiply__17 = paddle._C_ops.multiply(index_select_8, scale__45)

        # pd_op.add_: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add__31 = paddle._C_ops.add(multiply__16, multiply__17)

        # pd_op.maximum: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        maximum_4 = paddle.maximum(add__30, add__31)

        # pd_op.shape: (4xi32) <- (-1x128x14x14xf32)
        shape_25 = paddle._C_ops.shape(maximum_4)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(shape_25, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_50 = [slice_53, constant_8, constant_6, constant_22, constant_22]

        # pd_op.reshape_: (-1x8x16x14x14xf32, 0x-1x128x14x14xf32) <- (-1x128x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape(maximum_4, combine_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x8x14x14xf32) <- (-1x8x16x14x14xf32)
        transpose_16 = paddle._C_ops.transpose(reshape__100, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_51 = [slice_53, constant_14, constant_22, constant_22]

        # pd_op.reshape_: (-1x128x14x14xf32, 0x-1x16x8x14x14xf32) <- (-1x16x8x14x14xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_16, combine_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x128x7x14xf32) <- (-1x128x14x14xf32, 128x1x5x1xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(reshape__102, parameter_125, [2, 1], [2, 0], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x128x7x14xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x7x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_9, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x128x7x7xf32) <- (-1x128x7x14xf32, 128x1x1x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(batch_norm__96, parameter_130, [1, 2], [0, 2], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_10, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x128x7x7xf32)
        shape_26 = paddle._C_ops.shape(batch_norm__102)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(shape_26, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x128x1x1xf32) <- (-1x128x7x7xf32, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(batch_norm__102, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_52 = [slice_54, constant_14]

        # pd_op.reshape_: (-1x128xf32, 0x-1x128x1x1xf32) <- (-1x128x1x1xf32, [1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_9, combine_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16xf32) <- (-1x128xf32, 128x16xf32)
        matmul_18 = paddle.matmul(reshape__104, parameter_135, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add__32 = paddle._C_ops.add(matmul_18, parameter_136)

        # pd_op.relu_: (-1x16xf32) <- (-1x16xf32)
        relu__9 = paddle._C_ops.relu(add__32)

        # pd_op.matmul: (-1x512xf32) <- (-1x16xf32, 16x512xf32)
        matmul_19 = paddle.matmul(relu__9, parameter_137, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x512xf32) <- (-1x512xf32, 512xf32)
        add__33 = paddle._C_ops.add(matmul_19, parameter_138)

        # pd_op.hardsigmoid: (-1x512xf32) <- (-1x512xf32)
        hardsigmoid_9 = paddle._C_ops.hardsigmoid(add__33, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_53 = [slice_54, constant_25, constant_9, constant_9]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_9, combine_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1xf32)
        scale__46 = paddle._C_ops.scale(reshape__106, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1xf32)
        scale__47 = paddle._C_ops.scale(scale__46, constant_11, float('0'), True)

        # pd_op.index_select: (-1x128x7x7xf32) <- (-1x128x7x7xf32, 128xi64)
        index_select_9 = paddle._C_ops.index_select(batch_norm__102, parameter_139, 1)

        # pd_op.split_with_num: ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(scale__47, 4, constant_12)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32])
        slice_55 = split_with_num_9[0]

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__48 = paddle._C_ops.scale(slice_55, constant_10, float('1'), True)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32])
        slice_56 = split_with_num_9[2]

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__49 = paddle._C_ops.scale(slice_56, constant_10, float('1'), True)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32])
        slice_57 = split_with_num_9[1]

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__50 = paddle._C_ops.scale(slice_57, constant_10, float('0'), True)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32, -1x128x1x1xf32])
        slice_58 = split_with_num_9[3]

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__51 = paddle._C_ops.scale(slice_58, constant_10, float('0'), True)

        # pd_op.multiply: (-1x128x7x7xf32) <- (-1x128x7x7xf32, -1x128x1x1xf32)
        multiply_10 = batch_norm__102 * scale__48

        # pd_op.multiply: (-1x128x7x7xf32) <- (-1x128x7x7xf32, -1x128x1x1xf32)
        multiply_11 = index_select_9 * scale__50

        # pd_op.add_: (-1x128x7x7xf32) <- (-1x128x7x7xf32, -1x128x7x7xf32)
        add__34 = paddle._C_ops.add(multiply_10, multiply_11)

        # pd_op.multiply_: (-1x128x7x7xf32) <- (-1x128x7x7xf32, -1x128x1x1xf32)
        multiply__18 = paddle._C_ops.multiply(batch_norm__102, scale__49)

        # pd_op.multiply_: (-1x128x7x7xf32) <- (-1x128x7x7xf32, -1x128x1x1xf32)
        multiply__19 = paddle._C_ops.multiply(index_select_9, scale__51)

        # pd_op.add_: (-1x128x7x7xf32) <- (-1x128x7x7xf32, -1x128x7x7xf32)
        add__35 = paddle._C_ops.add(multiply__18, multiply__19)

        # pd_op.maximum: (-1x128x7x7xf32) <- (-1x128x7x7xf32, -1x128x7x7xf32)
        maximum_5 = paddle.maximum(add__34, add__35)

        # pd_op.shape: (4xi32) <- (-1x128x7x7xf32)
        shape_27 = paddle._C_ops.shape(maximum_5)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(shape_27, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_54 = [slice_59, constant_13, constant_4, constant_26, constant_26]

        # pd_op.reshape_: (-1x32x4x7x7xf32, 0x-1x128x7x7xf32) <- (-1x128x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape(maximum_5, combine_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x32x7x7xf32) <- (-1x32x4x7x7xf32)
        transpose_17 = paddle._C_ops.transpose(reshape__108, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_55 = [slice_59, constant_14, constant_26, constant_26]

        # pd_op.reshape_: (-1x128x7x7xf32, 0x-1x4x32x7x7xf32) <- (-1x4x32x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_17, combine_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x7x7xf32) <- (-1x128x7x7xf32, 64x16x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(reshape__110, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 8, 'NCHW')

        # pd_op.batch_norm_: (-1x64x7x7xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x7x7xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x64x7x7xf32)
        shape_28 = paddle._C_ops.shape(batch_norm__108)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(shape_28, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x64x1x1xf32) <- (-1x64x7x7xf32, 2xi64)
        pool2d_10 = paddle._C_ops.pool2d(batch_norm__108, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_56 = [slice_60, constant_23]

        # pd_op.reshape_: (-1x64xf32, 0x-1x64x1x1xf32) <- (-1x64x1x1xf32, [1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_10, combine_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16xf32) <- (-1x64xf32, 64x16xf32)
        matmul_20 = paddle.matmul(reshape__112, parameter_145, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add__36 = paddle._C_ops.add(matmul_20, parameter_146)

        # pd_op.relu_: (-1x16xf32) <- (-1x16xf32)
        relu__10 = paddle._C_ops.relu(add__36)

        # pd_op.matmul: (-1x128xf32) <- (-1x16xf32, 16x128xf32)
        matmul_21 = paddle.matmul(relu__10, parameter_147, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x128xf32) <- (-1x128xf32, 128xf32)
        add__37 = paddle._C_ops.add(matmul_21, parameter_148)

        # pd_op.hardsigmoid: (-1x128xf32) <- (-1x128xf32)
        hardsigmoid_10 = paddle._C_ops.hardsigmoid(add__37, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_57 = [slice_60, constant_14, constant_9, constant_9]

        # pd_op.reshape_: (-1x128x1x1xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_10, combine_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__52 = paddle._C_ops.scale(reshape__114, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1xf32)
        scale__53 = paddle._C_ops.scale(scale__52, constant_11, float('0'), True)

        # pd_op.index_select: (-1x64x7x7xf32) <- (-1x64x7x7xf32, 64xi64)
        index_select_10 = paddle._C_ops.index_select(batch_norm__108, parameter_149, 1)

        # pd_op.split_with_num: ([-1x64x1x1xf32, -1x64x1x1xf32]) <- (-1x128x1x1xf32, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(scale__53, 2, constant_12)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32])
        slice_61 = split_with_num_10[0]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__54 = paddle._C_ops.scale(slice_61, constant_10, float('1'), True)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32])
        slice_62 = split_with_num_10[1]

        # pd_op.scale_: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1xf32)
        scale__55 = paddle._C_ops.scale(slice_62, constant_10, float('0'), True)

        # pd_op.multiply_: (-1x64x7x7xf32) <- (-1x64x7x7xf32, -1x64x1x1xf32)
        multiply__20 = paddle._C_ops.multiply(batch_norm__108, scale__54)

        # pd_op.multiply_: (-1x64x7x7xf32) <- (-1x64x7x7xf32, -1x64x1x1xf32)
        multiply__21 = paddle._C_ops.multiply(index_select_10, scale__55)

        # pd_op.add_: (-1x64x7x7xf32) <- (-1x64x7x7xf32, -1x64x7x7xf32)
        add__38 = paddle._C_ops.add(multiply__20, multiply__21)

        # pd_op.shape: (4xi32) <- (-1x64x7x7xf32)
        shape_29 = paddle._C_ops.shape(add__38)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(shape_29, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_58 = [slice_63, constant_8, constant_8, constant_26, constant_26]

        # pd_op.reshape_: (-1x8x8x7x7xf32, 0x-1x64x7x7xf32) <- (-1x64x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__38, combine_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x8x7x7xf32) <- (-1x8x8x7x7xf32)
        transpose_18 = paddle._C_ops.transpose(reshape__116, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_59 = [slice_63, constant_23, constant_26, constant_26]

        # pd_op.reshape_: (-1x64x7x7xf32, 0x-1x8x8x7x7xf32) <- (-1x8x8x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_18, combine_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x64x7x7xf32)
        shape_30 = paddle._C_ops.shape(reshape__118)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(shape_30, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_60 = [slice_64, constant_13, constant_2, constant_26, constant_26]

        # pd_op.reshape_: (-1x32x2x7x7xf32, 0x-1x64x7x7xf32) <- (-1x64x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__118, combine_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x32x7x7xf32) <- (-1x32x2x7x7xf32)
        transpose_19 = paddle._C_ops.transpose(reshape__120, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_61 = [slice_64, constant_23, constant_26, constant_26]

        # pd_op.reshape_: (-1x64x7x7xf32, 0x-1x2x32x7x7xf32) <- (-1x2x32x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_19, combine_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x64x7x7xf32, 256x8x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(reshape__122, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 8, 'NCHW')

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x256x7x7xf32)
        shape_31 = paddle._C_ops.shape(batch_norm__114)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(shape_31, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x7x7xf32, 2xi64)
        pool2d_11 = paddle._C_ops.pool2d(batch_norm__114, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_62 = [slice_65, constant_24]

        # pd_op.reshape_: (-1x256xf32, 0x-1x256x1x1xf32) <- (-1x256x1x1xf32, [1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_11, combine_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16xf32) <- (-1x256xf32, 256x16xf32)
        matmul_22 = paddle.matmul(reshape__124, parameter_155, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add__39 = paddle._C_ops.add(matmul_22, parameter_156)

        # pd_op.relu_: (-1x16xf32) <- (-1x16xf32)
        relu__11 = paddle._C_ops.relu(add__39)

        # pd_op.matmul: (-1x1024xf32) <- (-1x16xf32, 16x1024xf32)
        matmul_23 = paddle.matmul(relu__11, parameter_157, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__40 = paddle._C_ops.add(matmul_23, parameter_158)

        # pd_op.hardsigmoid: (-1x1024xf32) <- (-1x1024xf32)
        hardsigmoid_11 = paddle._C_ops.hardsigmoid(add__40, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_63 = [slice_65, constant_27, constant_9, constant_9]

        # pd_op.reshape_: (-1x1024x1x1xf32, 0x-1x1024xf32) <- (-1x1024xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_11, combine_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x1024x1x1xf32) <- (-1x1024x1x1xf32, 1xf32)
        scale__56 = paddle._C_ops.scale(reshape__126, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x1024x1x1xf32) <- (-1x1024x1x1xf32, 1xf32)
        scale__57 = paddle._C_ops.scale(scale__56, constant_11, float('0'), True)

        # pd_op.index_select: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256xi64)
        index_select_11 = paddle._C_ops.index_select(batch_norm__114, parameter_159, 1)

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x1024x1x1xf32, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(scale__57, 4, constant_12)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32])
        slice_66 = split_with_num_11[0]

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__58 = paddle._C_ops.scale(slice_66, constant_10, float('1'), True)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32])
        slice_67 = split_with_num_11[2]

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__59 = paddle._C_ops.scale(slice_67, constant_10, float('1'), True)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32])
        slice_68 = split_with_num_11[1]

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__60 = paddle._C_ops.scale(slice_68, constant_10, float('0'), True)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32])
        slice_69 = split_with_num_11[3]

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__61 = paddle._C_ops.scale(slice_69, constant_10, float('0'), True)

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x1x1xf32)
        multiply_12 = batch_norm__114 * scale__58

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x1x1xf32)
        multiply_13 = index_select_11 * scale__60

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__41 = paddle._C_ops.add(multiply_12, multiply_13)

        # pd_op.multiply_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x1x1xf32)
        multiply__22 = paddle._C_ops.multiply(batch_norm__114, scale__59)

        # pd_op.multiply_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x1x1xf32)
        multiply__23 = paddle._C_ops.multiply(index_select_11, scale__61)

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__42 = paddle._C_ops.add(multiply__22, multiply__23)

        # pd_op.maximum: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        maximum_6 = paddle.maximum(add__41, add__42)

        # pd_op.shape: (4xi32) <- (-1x256x7x7xf32)
        shape_32 = paddle._C_ops.shape(maximum_6)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(shape_32, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_64 = [slice_70, constant_8, constant_13, constant_26, constant_26]

        # pd_op.reshape_: (-1x8x32x7x7xf32, 0x-1x256x7x7xf32) <- (-1x256x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape(maximum_6, combine_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x32x8x7x7xf32) <- (-1x8x32x7x7xf32)
        transpose_20 = paddle._C_ops.transpose(reshape__128, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_65 = [slice_70, constant_24, constant_26, constant_26]

        # pd_op.reshape_: (-1x256x7x7xf32, 0x-1x32x8x7x7xf32) <- (-1x32x8x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_20, combine_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x3x1xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(reshape__130, parameter_160, [1, 1], [1, 0], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_11, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x1x3xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(batch_norm__120, parameter_165, [1, 1], [0, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_12, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x256x7x7xf32)
        shape_33 = paddle._C_ops.shape(batch_norm__126)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(shape_33, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x7x7xf32, 2xi64)
        pool2d_12 = paddle._C_ops.pool2d(batch_norm__126, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_66 = [slice_71, constant_24]

        # pd_op.reshape_: (-1x256xf32, 0x-1x256x1x1xf32) <- (-1x256x1x1xf32, [1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_12, combine_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16xf32) <- (-1x256xf32, 256x16xf32)
        matmul_24 = paddle.matmul(reshape__132, parameter_170, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add__43 = paddle._C_ops.add(matmul_24, parameter_171)

        # pd_op.relu_: (-1x16xf32) <- (-1x16xf32)
        relu__12 = paddle._C_ops.relu(add__43)

        # pd_op.matmul: (-1x1024xf32) <- (-1x16xf32, 16x1024xf32)
        matmul_25 = paddle.matmul(relu__12, parameter_172, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__44 = paddle._C_ops.add(matmul_25, parameter_173)

        # pd_op.hardsigmoid: (-1x1024xf32) <- (-1x1024xf32)
        hardsigmoid_12 = paddle._C_ops.hardsigmoid(add__44, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_67 = [slice_71, constant_27, constant_9, constant_9]

        # pd_op.reshape_: (-1x1024x1x1xf32, 0x-1x1024xf32) <- (-1x1024xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_12, combine_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x1024x1x1xf32) <- (-1x1024x1x1xf32, 1xf32)
        scale__62 = paddle._C_ops.scale(reshape__134, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x1024x1x1xf32) <- (-1x1024x1x1xf32, 1xf32)
        scale__63 = paddle._C_ops.scale(scale__62, constant_11, float('0'), True)

        # pd_op.index_select: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256xi64)
        index_select_12 = paddle._C_ops.index_select(batch_norm__126, parameter_174, 1)

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x1024x1x1xf32, 1xi32)
        split_with_num_12 = paddle._C_ops.split_with_num(scale__63, 4, constant_12)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32])
        slice_72 = split_with_num_12[0]

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__64 = paddle._C_ops.scale(slice_72, constant_10, float('1'), True)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32])
        slice_73 = split_with_num_12[2]

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__65 = paddle._C_ops.scale(slice_73, constant_10, float('1'), True)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32])
        slice_74 = split_with_num_12[1]

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__66 = paddle._C_ops.scale(slice_74, constant_10, float('0'), True)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32, -1x256x1x1xf32])
        slice_75 = split_with_num_12[3]

        # pd_op.scale_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1xf32)
        scale__67 = paddle._C_ops.scale(slice_75, constant_10, float('0'), True)

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x1x1xf32)
        multiply_14 = batch_norm__126 * scale__64

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x1x1xf32)
        multiply_15 = index_select_12 * scale__66

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__45 = paddle._C_ops.add(multiply_14, multiply_15)

        # pd_op.multiply_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x1x1xf32)
        multiply__24 = paddle._C_ops.multiply(batch_norm__126, scale__65)

        # pd_op.multiply_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x1x1xf32)
        multiply__25 = paddle._C_ops.multiply(index_select_12, scale__67)

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__46 = paddle._C_ops.add(multiply__24, multiply__25)

        # pd_op.maximum: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        maximum_7 = paddle.maximum(add__45, add__46)

        # pd_op.shape: (4xi32) <- (-1x256x7x7xf32)
        shape_34 = paddle._C_ops.shape(maximum_7)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(shape_34, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_68 = [slice_76, constant_23, constant_4, constant_26, constant_26]

        # pd_op.reshape_: (-1x64x4x7x7xf32, 0x-1x256x7x7xf32) <- (-1x256x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape(maximum_7, combine_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x64x7x7xf32) <- (-1x64x4x7x7xf32)
        transpose_21 = paddle._C_ops.transpose(reshape__136, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_69 = [slice_76, constant_24, constant_26, constant_26]

        # pd_op.reshape_: (-1x256x7x7xf32, 0x-1x4x64x7x7xf32) <- (-1x4x64x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_21, combine_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x96x7x7xf32) <- (-1x256x7x7xf32, 96x32x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(reshape__138, parameter_175, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 8, 'NCHW')

        # pd_op.batch_norm_: (-1x96x7x7xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x7x7xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x96x7x7xf32)
        shape_35 = paddle._C_ops.shape(batch_norm__132)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(shape_35, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x7x7xf32, 2xi64)
        pool2d_13 = paddle._C_ops.pool2d(batch_norm__132, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_70 = [slice_77, constant_28]

        # pd_op.reshape_: (-1x96xf32, 0x-1x96x1x1xf32) <- (-1x96x1x1xf32, [1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_13, combine_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x12xf32) <- (-1x96xf32, 96x12xf32)
        matmul_26 = paddle.matmul(reshape__140, parameter_180, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x12xf32) <- (-1x12xf32, 12xf32)
        add__47 = paddle._C_ops.add(matmul_26, parameter_181)

        # pd_op.relu_: (-1x12xf32) <- (-1x12xf32)
        relu__13 = paddle._C_ops.relu(add__47)

        # pd_op.matmul: (-1x192xf32) <- (-1x12xf32, 12x192xf32)
        matmul_27 = paddle.matmul(relu__13, parameter_182, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x192xf32) <- (-1x192xf32, 192xf32)
        add__48 = paddle._C_ops.add(matmul_27, parameter_183)

        # pd_op.hardsigmoid: (-1x192xf32) <- (-1x192xf32)
        hardsigmoid_13 = paddle._C_ops.hardsigmoid(add__48, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_71 = [slice_77, constant_21, constant_9, constant_9]

        # pd_op.reshape_: (-1x192x1x1xf32, 0x-1x192xf32) <- (-1x192xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_13, combine_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1xf32)
        scale__68 = paddle._C_ops.scale(reshape__142, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1xf32)
        scale__69 = paddle._C_ops.scale(scale__68, constant_11, float('0'), True)

        # pd_op.index_select: (-1x96x7x7xf32) <- (-1x96x7x7xf32, 96xi64)
        index_select_13 = paddle._C_ops.index_select(batch_norm__132, parameter_184, 1)

        # pd_op.split_with_num: ([-1x96x1x1xf32, -1x96x1x1xf32]) <- (-1x192x1x1xf32, 1xi32)
        split_with_num_13 = paddle._C_ops.split_with_num(scale__69, 2, constant_12)

        # builtin.slice: (-1x96x1x1xf32) <- ([-1x96x1x1xf32, -1x96x1x1xf32])
        slice_78 = split_with_num_13[0]

        # pd_op.scale_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1xf32)
        scale__70 = paddle._C_ops.scale(slice_78, constant_10, float('1'), True)

        # builtin.slice: (-1x96x1x1xf32) <- ([-1x96x1x1xf32, -1x96x1x1xf32])
        slice_79 = split_with_num_13[1]

        # pd_op.scale_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1xf32)
        scale__71 = paddle._C_ops.scale(slice_79, constant_10, float('0'), True)

        # pd_op.multiply_: (-1x96x7x7xf32) <- (-1x96x7x7xf32, -1x96x1x1xf32)
        multiply__26 = paddle._C_ops.multiply(batch_norm__132, scale__70)

        # pd_op.multiply_: (-1x96x7x7xf32) <- (-1x96x7x7xf32, -1x96x1x1xf32)
        multiply__27 = paddle._C_ops.multiply(index_select_13, scale__71)

        # pd_op.add_: (-1x96x7x7xf32) <- (-1x96x7x7xf32, -1x96x7x7xf32)
        add__49 = paddle._C_ops.add(multiply__26, multiply__27)

        # pd_op.shape: (4xi32) <- (-1x96x7x7xf32)
        shape_36 = paddle._C_ops.shape(add__49)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(shape_36, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_72 = [slice_80, constant_8, constant_16, constant_26, constant_26]

        # pd_op.reshape_: (-1x8x12x7x7xf32, 0x-1x96x7x7xf32) <- (-1x96x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__49, combine_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x8x7x7xf32) <- (-1x8x12x7x7xf32)
        transpose_22 = paddle._C_ops.transpose(reshape__144, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_73 = [slice_80, constant_28, constant_26, constant_26]

        # pd_op.reshape_: (-1x96x7x7xf32, 0x-1x12x8x7x7xf32) <- (-1x12x8x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_22, combine_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x96x7x7xf32)
        shape_37 = paddle._C_ops.shape(reshape__146)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(shape_37, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_74 = [slice_81, constant_20, constant_2, constant_26, constant_26]

        # pd_op.reshape_: (-1x48x2x7x7xf32, 0x-1x96x7x7xf32) <- (-1x96x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__146, combine_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x48x7x7xf32) <- (-1x48x2x7x7xf32)
        transpose_23 = paddle._C_ops.transpose(reshape__148, [0, 2, 1, 3, 4])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_75 = [slice_81, constant_28, constant_26, constant_26]

        # pd_op.reshape_: (-1x96x7x7xf32, 0x-1x2x48x7x7xf32) <- (-1x2x48x7x7xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_23, combine_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x384x7x7xf32) <- (-1x96x7x7xf32, 384x8x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(reshape__150, parameter_185, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x7x7xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x384x7x7xf32)
        shape_38 = paddle._C_ops.shape(batch_norm__138)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(shape_38, [0], constant_0, constant_1, [1], [0])

        # pd_op.pool2d: (-1x384x1x1xf32) <- (-1x384x7x7xf32, 2xi64)
        pool2d_14 = paddle._C_ops.pool2d(batch_norm__138, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_76 = [slice_82, constant_29]

        # pd_op.reshape_: (-1x384xf32, 0x-1x384x1x1xf32) <- (-1x384x1x1xf32, [1xi32, 1xi32])
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_14, combine_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x24xf32) <- (-1x384xf32, 384x24xf32)
        matmul_28 = paddle.matmul(reshape__152, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x24xf32) <- (-1x24xf32, 24xf32)
        add__50 = paddle._C_ops.add(matmul_28, parameter_191)

        # pd_op.relu_: (-1x24xf32) <- (-1x24xf32)
        relu__14 = paddle._C_ops.relu(add__50)

        # pd_op.matmul: (-1x768xf32) <- (-1x24xf32, 24x768xf32)
        matmul_29 = paddle.matmul(relu__14, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x768xf32) <- (-1x768xf32, 768xf32)
        add__51 = paddle._C_ops.add(matmul_29, parameter_193)

        # pd_op.hardsigmoid: (-1x768xf32) <- (-1x768xf32)
        hardsigmoid_14 = paddle._C_ops.hardsigmoid(add__51, float('0.166667'), float('0.5'))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_77 = [slice_82, constant_30, constant_9, constant_9]

        # pd_op.reshape_: (-1x768x1x1xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape(hardsigmoid_14, combine_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale_: (-1x768x1x1xf32) <- (-1x768x1x1xf32, 1xf32)
        scale__72 = paddle._C_ops.scale(reshape__154, constant_10, float('-0.5'), True)

        # pd_op.scale_: (-1x768x1x1xf32) <- (-1x768x1x1xf32, 1xf32)
        scale__73 = paddle._C_ops.scale(scale__72, constant_11, float('0'), True)

        # pd_op.index_select: (-1x384x7x7xf32) <- (-1x384x7x7xf32, 384xi64)
        index_select_14 = paddle._C_ops.index_select(batch_norm__138, parameter_194, 1)

        # pd_op.split_with_num: ([-1x384x1x1xf32, -1x384x1x1xf32]) <- (-1x768x1x1xf32, 1xi32)
        split_with_num_14 = paddle._C_ops.split_with_num(scale__73, 2, constant_12)

        # builtin.slice: (-1x384x1x1xf32) <- ([-1x384x1x1xf32, -1x384x1x1xf32])
        slice_83 = split_with_num_14[0]

        # pd_op.scale_: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 1xf32)
        scale__74 = paddle._C_ops.scale(slice_83, constant_10, float('1'), True)

        # builtin.slice: (-1x384x1x1xf32) <- ([-1x384x1x1xf32, -1x384x1x1xf32])
        slice_84 = split_with_num_14[1]

        # pd_op.scale_: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 1xf32)
        scale__75 = paddle._C_ops.scale(slice_84, constant_10, float('0'), True)

        # pd_op.multiply_: (-1x384x7x7xf32) <- (-1x384x7x7xf32, -1x384x1x1xf32)
        multiply__28 = paddle._C_ops.multiply(batch_norm__138, scale__74)

        # pd_op.multiply_: (-1x384x7x7xf32) <- (-1x384x7x7xf32, -1x384x1x1xf32)
        multiply__29 = paddle._C_ops.multiply(index_select_14, scale__75)

        # pd_op.add_: (-1x384x7x7xf32) <- (-1x384x7x7xf32, -1x384x7x7xf32)
        add__52 = paddle._C_ops.add(multiply__28, multiply__29)

        # pd_op.relu6: (-1x384x7x7xf32) <- (-1x384x7x7xf32)
        relu6_2 = paddle._C_ops.relu6(add__52)

        # pd_op.pool2d: (-1x384x1x1xf32) <- (-1x384x7x7xf32, 2xi64)
        pool2d_15 = paddle._C_ops.pool2d(relu6_2, constant_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.hardswish: (-1x384x1x1xf32) <- (-1x384x1x1xf32)
        hardswish_0 = paddle._C_ops.hardswish(pool2d_15)

        # pd_op.flatten_: (-1x384xf32, None) <- (-1x384x1x1xf32)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten(hardswish_0, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x640xf32) <- (-1x384xf32, 384x640xf32)
        matmul_30 = paddle.matmul(flatten__0, parameter_195, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x640xf32) <- (-1x640xf32, 640xf32)
        add__53 = paddle._C_ops.add(matmul_30, parameter_196)

        # pd_op.batch_norm_: (-1x640xf32, 640xf32, 640xf32, xf32, xf32, None) <- (-1x640xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__53, parameter_197, parameter_198, parameter_199, parameter_200, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x640xf32) <- (-1x640xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__144)

        # pd_op.dropout: (-1x640xf32, None) <- (-1x640xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(hardswish_1, None, constant_31, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf32) <- (-1x640xf32, 640x1000xf32)
        matmul_31 = paddle.matmul(dropout_0, parameter_201, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__54 = paddle._C_ops.add(matmul_31, parameter_202)

        # pd_op.batch_norm_: (-1x1000xf32, 1000xf32, 1000xf32, xf32, xf32, None) <- (-1x1000xf32, 1000xf32, 1000xf32, 1000xf32, 1000xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__54, parameter_203, parameter_204, parameter_205, parameter_206, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x1000xf32) <- (-1x1000xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__150)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__0 = paddle._C_ops.softmax(hardswish_2, -1)
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

    def forward(self, constant_31, constant_30, constant_29, constant_28, constant_27, constant_26, constant_25, constant_24, constant_23, constant_22, constant_21, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, feed_0):
        return self.builtin_module_1508_0_0(constant_31, constant_30, constant_29, constant_28, constant_27, constant_26, constant_25, constant_24, constant_23, constant_22, constant_21, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1508_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_31
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_30
            paddle.to_tensor([768], dtype='int32').reshape([1]),
            # constant_29
            paddle.to_tensor([384], dtype='int32').reshape([1]),
            # constant_28
            paddle.to_tensor([96], dtype='int32').reshape([1]),
            # constant_27
            paddle.to_tensor([1024], dtype='int32').reshape([1]),
            # constant_26
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            # constant_25
            paddle.to_tensor([512], dtype='int32').reshape([1]),
            # constant_24
            paddle.to_tensor([256], dtype='int32').reshape([1]),
            # constant_23
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_22
            paddle.to_tensor([14], dtype='int32').reshape([1]),
            # constant_21
            paddle.to_tensor([192], dtype='int32').reshape([1]),
            # constant_20
            paddle.to_tensor([48], dtype='int32').reshape([1]),
            # constant_19
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            # constant_18
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            # constant_17
            paddle.to_tensor([24], dtype='int32').reshape([1]),
            # constant_16
            paddle.to_tensor([12], dtype='int32').reshape([1]),
            # constant_15
            paddle.to_tensor([28], dtype='int32').reshape([1]),
            # constant_14
            paddle.to_tensor([128], dtype='int32').reshape([1]),
            # constant_13
            paddle.to_tensor([32], dtype='int32').reshape([1]),
            # constant_12
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_11
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_10
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_9
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_8
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            # constant_7
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_6
            paddle.to_tensor([16], dtype='int32').reshape([1]),
            # constant_5
            paddle.to_tensor([56], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            # constant_3
            paddle.to_tensor([112], dtype='int32').reshape([1]),
            # constant_2
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_0
            paddle.uniform([2, 3, 3, 1], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([4, 1, 1, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([8, 1, 3, 1], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([16, 1, 1, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([8, 8, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([4, 16], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.to_tensor([5, 6, 7, 4, 1, 2, 3, 0], dtype='int64').reshape([8]),
            # parameter_30
            paddle.uniform([16, 1, 3, 1], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([32, 1, 1, 3], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([32, 4], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([4, 128], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.to_tensor([5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 17, 18, 19, 16, 21, 22, 23, 20, 25, 26, 27, 24, 29, 30, 31, 28, 1, 2, 3, 0], dtype='int64').reshape([32]),
            # parameter_45
            paddle.uniform([12, 8, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([12, 4], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([4, 24], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.to_tensor([4, 5, 3, 7, 8, 6, 10, 11, 9, 1, 2, 0], dtype='int64').reshape([12]),
            # parameter_55
            paddle.uniform([24, 1, 5, 1], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([48, 1, 1, 5], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([48, 8], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([8, 192], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.to_tensor([5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 17, 18, 19, 16, 21, 22, 23, 20, 25, 26, 27, 24, 29, 30, 31, 28, 33, 34, 35, 32, 37, 38, 39, 36, 41, 42, 43, 40, 45, 46, 47, 44, 1, 2, 3, 0], dtype='int64').reshape([48]),
            # parameter_70
            paddle.uniform([16, 12, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([16, 4], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([4, 32], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.to_tensor([5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0], dtype='int64').reshape([16]),
            # parameter_80
            paddle.uniform([64, 4, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([64, 8], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([8, 256], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.to_tensor([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 32, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 48, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], dtype='int64').reshape([64]),
            # parameter_90
            paddle.uniform([64, 1, 5, 1], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([64, 1, 1, 5], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([64, 8], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([8, 256], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.to_tensor([5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 17, 18, 19, 16, 21, 22, 23, 20, 25, 26, 27, 24, 29, 30, 31, 28, 33, 34, 35, 32, 37, 38, 39, 36, 41, 42, 43, 40, 45, 46, 47, 44, 49, 50, 51, 48, 53, 54, 55, 52, 57, 58, 59, 56, 61, 62, 63, 60, 1, 2, 3, 0], dtype='int64').reshape([64]),
            # parameter_105
            paddle.uniform([32, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([32, 8], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([8, 64], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.to_tensor([9, 10, 11, 12, 13, 14, 15, 8, 17, 18, 19, 20, 21, 22, 23, 16, 25, 26, 27, 28, 29, 30, 31, 24, 1, 2, 3, 4, 5, 6, 7, 0], dtype='int64').reshape([32]),
            # parameter_115
            paddle.uniform([128, 4, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([128, 16], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([16, 512], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.to_tensor([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 32, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 48, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 64, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 80, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 96, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 112, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], dtype='int64').reshape([128]),
            # parameter_125
            paddle.uniform([128, 1, 5, 1], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([128, 1, 1, 5], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128, 16], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([16, 512], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.to_tensor([9, 10, 11, 12, 13, 14, 15, 8, 17, 18, 19, 20, 21, 22, 23, 16, 25, 26, 27, 28, 29, 30, 31, 24, 33, 34, 35, 36, 37, 38, 39, 32, 41, 42, 43, 44, 45, 46, 47, 40, 49, 50, 51, 52, 53, 54, 55, 48, 57, 58, 59, 60, 61, 62, 63, 56, 65, 66, 67, 68, 69, 70, 71, 64, 73, 74, 75, 76, 77, 78, 79, 72, 81, 82, 83, 84, 85, 86, 87, 80, 89, 90, 91, 92, 93, 94, 95, 88, 97, 98, 99, 100, 101, 102, 103, 96, 105, 106, 107, 108, 109, 110, 111, 104, 113, 114, 115, 116, 117, 118, 119, 112, 121, 122, 123, 124, 125, 126, 127, 120, 1, 2, 3, 4, 5, 6, 7, 0], dtype='int64').reshape([128]),
            # parameter_140
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([64, 16], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([16, 128], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.to_tensor([9, 10, 11, 12, 13, 14, 15, 8, 17, 18, 19, 20, 21, 22, 23, 16, 25, 26, 27, 28, 29, 30, 31, 24, 33, 34, 35, 36, 37, 38, 39, 32, 41, 42, 43, 44, 45, 46, 47, 40, 49, 50, 51, 52, 53, 54, 55, 48, 57, 58, 59, 60, 61, 62, 63, 56, 1, 2, 3, 4, 5, 6, 7, 0], dtype='int64').reshape([64]),
            # parameter_150
            paddle.uniform([256, 8, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([256, 16], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([16, 1024], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.to_tensor([33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 32, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 64, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 96, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 128, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 160, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 192, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 224, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0], dtype='int64').reshape([256]),
            # parameter_160
            paddle.uniform([256, 1, 3, 1], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256, 1, 1, 3], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([256, 16], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([16, 1024], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.to_tensor([9, 10, 11, 12, 13, 14, 15, 8, 17, 18, 19, 20, 21, 22, 23, 16, 25, 26, 27, 28, 29, 30, 31, 24, 33, 34, 35, 36, 37, 38, 39, 32, 41, 42, 43, 44, 45, 46, 47, 40, 49, 50, 51, 52, 53, 54, 55, 48, 57, 58, 59, 60, 61, 62, 63, 56, 65, 66, 67, 68, 69, 70, 71, 64, 73, 74, 75, 76, 77, 78, 79, 72, 81, 82, 83, 84, 85, 86, 87, 80, 89, 90, 91, 92, 93, 94, 95, 88, 97, 98, 99, 100, 101, 102, 103, 96, 105, 106, 107, 108, 109, 110, 111, 104, 113, 114, 115, 116, 117, 118, 119, 112, 121, 122, 123, 124, 125, 126, 127, 120, 129, 130, 131, 132, 133, 134, 135, 128, 137, 138, 139, 140, 141, 142, 143, 136, 145, 146, 147, 148, 149, 150, 151, 144, 153, 154, 155, 156, 157, 158, 159, 152, 161, 162, 163, 164, 165, 166, 167, 160, 169, 170, 171, 172, 173, 174, 175, 168, 177, 178, 179, 180, 181, 182, 183, 176, 185, 186, 187, 188, 189, 190, 191, 184, 193, 194, 195, 196, 197, 198, 199, 192, 201, 202, 203, 204, 205, 206, 207, 200, 209, 210, 211, 212, 213, 214, 215, 208, 217, 218, 219, 220, 221, 222, 223, 216, 225, 226, 227, 228, 229, 230, 231, 224, 233, 234, 235, 236, 237, 238, 239, 232, 241, 242, 243, 244, 245, 246, 247, 240, 249, 250, 251, 252, 253, 254, 255, 248, 1, 2, 3, 4, 5, 6, 7, 0], dtype='int64').reshape([256]),
            # parameter_175
            paddle.uniform([96, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([96, 12], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([12, 192], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.to_tensor([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 24, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 36, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 60, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 72, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 84, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0], dtype='int64').reshape([96]),
            # parameter_185
            paddle.uniform([384, 8, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([384, 24], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([24, 768], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.to_tensor([33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 32, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 64, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 96, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 128, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 160, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 192, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 224, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 256, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 288, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 320, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 352, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0], dtype='int64').reshape([384]),
            # parameter_195
            paddle.uniform([384, 640], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([640, 1000], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_31
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_30
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_29
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_28
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_27
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_26
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_25
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_24
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_23
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_22
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_21
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_20
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_19
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_18
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_17
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_15
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_7
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int32'),
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
            paddle.static.InputSpec(shape=[2, 3, 3, 1], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[4, 1, 1, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[8, 1, 3, 1], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[16, 1, 1, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[8, 8, 1, 1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[4, 16], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[8], dtype='int64'),
            # parameter_30
            paddle.static.InputSpec(shape=[16, 1, 3, 1], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[32, 1, 1, 3], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[32, 4], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[4, 128], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[32], dtype='int64'),
            # parameter_45
            paddle.static.InputSpec(shape=[12, 8, 1, 1], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[12, 4], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[4, 24], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[12], dtype='int64'),
            # parameter_55
            paddle.static.InputSpec(shape=[24, 1, 5, 1], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[48, 1, 1, 5], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[48, 8], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[8, 192], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[48], dtype='int64'),
            # parameter_70
            paddle.static.InputSpec(shape=[16, 12, 1, 1], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[16, 4], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[4, 32], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[16], dtype='int64'),
            # parameter_80
            paddle.static.InputSpec(shape=[64, 4, 1, 1], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[64, 8], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[8, 256], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[64], dtype='int64'),
            # parameter_90
            paddle.static.InputSpec(shape=[64, 1, 5, 1], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[64, 1, 1, 5], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[64, 8], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[8, 256], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[64], dtype='int64'),
            # parameter_105
            paddle.static.InputSpec(shape=[32, 16, 1, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[32, 8], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[8, 64], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[32], dtype='int64'),
            # parameter_115
            paddle.static.InputSpec(shape=[128, 4, 1, 1], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[128, 16], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[16, 512], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[128], dtype='int64'),
            # parameter_125
            paddle.static.InputSpec(shape=[128, 1, 5, 1], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[128, 1, 1, 5], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128, 16], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[16, 512], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[128], dtype='int64'),
            # parameter_140
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[64, 16], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[16, 128], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[64], dtype='int64'),
            # parameter_150
            paddle.static.InputSpec(shape=[256, 8, 1, 1], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[256, 16], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[16, 1024], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[256], dtype='int64'),
            # parameter_160
            paddle.static.InputSpec(shape=[256, 1, 3, 1], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256, 1, 1, 3], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[256, 16], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[16, 1024], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[256], dtype='int64'),
            # parameter_175
            paddle.static.InputSpec(shape=[96, 32, 1, 1], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[96, 12], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[12, 192], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[96], dtype='int64'),
            # parameter_185
            paddle.static.InputSpec(shape=[384, 8, 1, 1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[384, 24], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[24, 768], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[384], dtype='int64'),
            # parameter_195
            paddle.static.InputSpec(shape=[384, 640], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[640, 1000], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
            # parameter_204
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