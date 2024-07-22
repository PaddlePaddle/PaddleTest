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
    return [429][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1321_0_0(self, parameter_171, parameter_155, constant_22, constant_21, constant_20, parameter_139, parameter_133, parameter_121, parameter_113, constant_19, parameter_101, constant_18, constant_17, constant_16, constant_15, parameter_93, parameter_87, parameter_75, parameter_67, constant_14, parameter_55, constant_13, constant_12, constant_11, parameter_47, parameter_41, parameter_29, parameter_21, constant_10, constant_9, constant_8, constant_7, constant_6, parameter_9, constant_5, constant_4, constant_3, constant_2, parameter_1, constant_1, constant_0, parameter_0, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_22, parameter_23, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_37, parameter_36, parameter_38, parameter_39, parameter_40, parameter_42, parameter_43, parameter_45, parameter_44, parameter_46, parameter_49, parameter_48, parameter_51, parameter_50, parameter_52, parameter_53, parameter_54, parameter_57, parameter_56, parameter_58, parameter_59, parameter_60, parameter_61, parameter_63, parameter_62, parameter_64, parameter_65, parameter_66, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_95, parameter_94, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_114, parameter_115, parameter_117, parameter_116, parameter_118, parameter_119, parameter_120, parameter_123, parameter_122, parameter_124, parameter_125, parameter_126, parameter_127, parameter_129, parameter_128, parameter_130, parameter_131, parameter_132, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_141, parameter_140, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_156, parameter_157, parameter_159, parameter_158, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, feed_0):

        # pd_op.shape: (4xi32) <- (-1x3x224x224xf32)
        shape_0 = paddle._C_ops.shape(feed_0)

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x3x224x224xf32, 32x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [4, 4], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, parameter_1)

        # pd_op.flatten_: (-1x32x3136xf32, None) <- (-1x32x56x56xf32)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x32xf32) <- (-1x32x3136xf32)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.layer_norm: (-1x3136x32xf32, -3136xf32, -3136xf32) <- (-1x3136x32xf32, 32xf32, 32xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_0, parameter_2, parameter_3, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x3136x32xf32, -3136xf32, -3136xf32) <- (-1x3136x32xf32, 32xf32, 32xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_0, parameter_4, parameter_5, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x3136x32xf32)
        shape_1 = paddle._C_ops.shape(layer_norm_3)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x3136x32xf32) <- (-1x3136x32xf32, 32x32xf32)
        matmul_0 = paddle.matmul(layer_norm_3, parameter_6, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, 32xf32)
        add__1 = paddle._C_ops.add_(matmul_0, parameter_7)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_1, constant_2, constant_3, constant_4]

        # pd_op.reshape_: (-1x3136x1x32xf32, 0x-1x3136x32xf32) <- (-1x3136x32xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__1, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x3136x32xf32) <- (-1x3136x1x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3])

        # pd_op.transpose: (-1x32x3136xf32) <- (-1x3136x32xf32)
        transpose_2 = paddle._C_ops.transpose(layer_norm_3, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, constant_4, constant_5, constant_5]

        # pd_op.reshape_: (-1x32x56x56xf32, 0x-1x32x3136xf32) <- (-1x32x3136xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_2, combine_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x32x7x7xf32) <- (-1x32x56x56xf32, 32x32x8x8xf32)
        conv2d_1 = paddle._C_ops.conv2d(reshape__2, parameter_8, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x7x7xf32) <- (-1x32x7x7xf32, 1x32x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_1, parameter_9)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_2 = [slice_1, constant_4, constant_6]

        # pd_op.reshape_: (-1x32x49xf32, 0x-1x32x7x7xf32) <- (-1x32x7x7xf32, [1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__2, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x32xf32) <- (-1x32x49xf32)
        transpose_3 = paddle._C_ops.transpose(reshape__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x32xf32, -49xf32, -49xf32) <- (-1x49x32xf32, 32xf32, 32xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_3, parameter_10, parameter_11, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x64xf32) <- (-1x49x32xf32, 32x64xf32)
        matmul_1 = paddle.matmul(layer_norm_6, parameter_12, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x64xf32) <- (-1x49x64xf32, 64xf32)
        add__3 = paddle._C_ops.add_(matmul_1, parameter_13)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_1, constant_6, constant_7, constant_3, constant_4]

        # pd_op.reshape_: (-1x49x2x1x32xf32, 0x-1x49x64xf32) <- (-1x49x64xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__3, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x1x49x32xf32) <- (-1x49x2x1x32xf32)
        transpose_4 = paddle._C_ops.transpose(reshape__6, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x1x49x32xf32) <- (2x-1x1x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_4, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x1x49x32xf32) <- (2x-1x1x49x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_4, [0], constant_1, constant_8, [1], [0])

        # pd_op.transpose: (-1x1x32x49xf32) <- (-1x1x49x32xf32)
        transpose_5 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])

        # pd_op.matmul: (-1x1x3136x49xf32) <- (-1x1x3136x32xf32, -1x1x32x49xf32)
        matmul_2 = paddle.matmul(transpose_1, transpose_5, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x1x3136x49xf32) <- (-1x1x3136x49xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_2, constant_9, float('0'), True)

        # pd_op.softmax_: (-1x1x3136x49xf32) <- (-1x1x3136x49xf32)
        softmax__0 = paddle._C_ops.softmax_(scale__0, -1)

        # pd_op.matmul: (-1x1x3136x32xf32) <- (-1x1x3136x49xf32, -1x1x49x32xf32)
        matmul_3 = paddle.matmul(softmax__0, slice_3, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x3136x1x32xf32) <- (-1x1x3136x32xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_1, constant_2, constant_4]

        # pd_op.reshape_: (-1x3136x32xf32, 0x-1x3136x1x32xf32) <- (-1x3136x1x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, combine_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x3136x32xf32) <- (-1x3136x32xf32, 32x32xf32)
        matmul_4 = paddle.matmul(reshape__8, parameter_14, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, 32xf32)
        add__4 = paddle._C_ops.add_(matmul_4, parameter_15)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, -1x3136x32xf32)
        add__5 = paddle._C_ops.add_(layer_norm_0, add__4)

        # pd_op.layer_norm: (-1x3136x32xf32, -3136xf32, -3136xf32) <- (-1x3136x32xf32, 32xf32, 32xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__5, parameter_16, parameter_17, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x3136x256xf32) <- (-1x3136x32xf32, 32x256xf32)
        matmul_5 = paddle.matmul(layer_norm_9, parameter_18, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x256xf32) <- (-1x3136x256xf32, 256xf32)
        add__6 = paddle._C_ops.add_(matmul_5, parameter_19)

        # pd_op.shape: (3xi32) <- (-1x3136x256xf32)
        shape_2 = paddle._C_ops.shape(add__6)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_2, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x256x3136xf32) <- (-1x3136x256xf32)
        transpose_7 = paddle._C_ops.transpose(add__6, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_4, constant_10, constant_5, constant_5]

        # pd_op.reshape_: (-1x256x56x56xf32, 0x-1x256x3136xf32) <- (-1x256x3136xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 256x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(reshape__10, parameter_20, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__7 = paddle._C_ops.add_(depthwise_conv2d_0, parameter_21)

        # pd_op.flatten_: (-1x256x3136xf32, None) <- (-1x256x56x56xf32)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__7, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x256xf32) <- (-1x256x3136xf32)
        transpose_8 = paddle._C_ops.transpose(flatten__2, [0, 2, 1])

        # pd_op.gelu: (-1x3136x256xf32) <- (-1x3136x256xf32)
        gelu_0 = paddle._C_ops.gelu(transpose_8, False)

        # pd_op.matmul: (-1x3136x32xf32) <- (-1x3136x256xf32, 256x32xf32)
        matmul_6 = paddle.matmul(gelu_0, parameter_22, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, 32xf32)
        add__8 = paddle._C_ops.add_(matmul_6, parameter_23)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, -1x3136x32xf32)
        add__9 = paddle._C_ops.add_(add__5, add__8)

        # pd_op.layer_norm: (-1x3136x32xf32, -3136xf32, -3136xf32) <- (-1x3136x32xf32, 32xf32, 32xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__9, parameter_24, parameter_25, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x3136x32xf32)
        shape_3 = paddle._C_ops.shape(layer_norm_12)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_3, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x3136x32xf32) <- (-1x3136x32xf32, 32x32xf32)
        matmul_7 = paddle.matmul(layer_norm_12, parameter_26, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, 32xf32)
        add__10 = paddle._C_ops.add_(matmul_7, parameter_27)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_5, constant_2, constant_3, constant_4]

        # pd_op.reshape_: (-1x3136x1x32xf32, 0x-1x3136x32xf32) <- (-1x3136x32xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__10, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x3136x32xf32) <- (-1x3136x1x32xf32)
        transpose_9 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3])

        # pd_op.transpose: (-1x32x3136xf32) <- (-1x3136x32xf32)
        transpose_10 = paddle._C_ops.transpose(layer_norm_12, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_5, constant_4, constant_5, constant_5]

        # pd_op.reshape_: (-1x32x56x56xf32, 0x-1x32x3136xf32) <- (-1x32x3136xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x32x7x7xf32) <- (-1x32x56x56xf32, 32x32x8x8xf32)
        conv2d_2 = paddle._C_ops.conv2d(reshape__14, parameter_28, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x7x7xf32) <- (-1x32x7x7xf32, 1x32x1x1xf32)
        add__11 = paddle._C_ops.add_(conv2d_2, parameter_29)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_8 = [slice_5, constant_4, constant_6]

        # pd_op.reshape_: (-1x32x49xf32, 0x-1x32x7x7xf32) <- (-1x32x7x7xf32, [1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__11, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x32xf32) <- (-1x32x49xf32)
        transpose_11 = paddle._C_ops.transpose(reshape__16, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x32xf32, -49xf32, -49xf32) <- (-1x49x32xf32, 32xf32, 32xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_11, parameter_30, parameter_31, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x64xf32) <- (-1x49x32xf32, 32x64xf32)
        matmul_8 = paddle.matmul(layer_norm_15, parameter_32, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x64xf32) <- (-1x49x64xf32, 64xf32)
        add__12 = paddle._C_ops.add_(matmul_8, parameter_33)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_5, constant_6, constant_7, constant_3, constant_4]

        # pd_op.reshape_: (-1x49x2x1x32xf32, 0x-1x49x64xf32) <- (-1x49x64xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__12, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x1x49x32xf32) <- (-1x49x2x1x32xf32)
        transpose_12 = paddle._C_ops.transpose(reshape__18, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x1x49x32xf32) <- (2x-1x1x49x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_12, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x1x49x32xf32) <- (2x-1x1x49x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_12, [0], constant_1, constant_8, [1], [0])

        # pd_op.transpose: (-1x1x32x49xf32) <- (-1x1x49x32xf32)
        transpose_13 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])

        # pd_op.matmul: (-1x1x3136x49xf32) <- (-1x1x3136x32xf32, -1x1x32x49xf32)
        matmul_9 = paddle.matmul(transpose_9, transpose_13, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x1x3136x49xf32) <- (-1x1x3136x49xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_9, constant_9, float('0'), True)

        # pd_op.softmax_: (-1x1x3136x49xf32) <- (-1x1x3136x49xf32)
        softmax__1 = paddle._C_ops.softmax_(scale__1, -1)

        # pd_op.matmul: (-1x1x3136x32xf32) <- (-1x1x3136x49xf32, -1x1x49x32xf32)
        matmul_10 = paddle.matmul(softmax__1, slice_7, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x3136x1x32xf32) <- (-1x1x3136x32xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_10, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_5, constant_2, constant_4]

        # pd_op.reshape_: (-1x3136x32xf32, 0x-1x3136x1x32xf32) <- (-1x3136x1x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x3136x32xf32) <- (-1x3136x32xf32, 32x32xf32)
        matmul_11 = paddle.matmul(reshape__20, parameter_34, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, 32xf32)
        add__13 = paddle._C_ops.add_(matmul_11, parameter_35)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, -1x3136x32xf32)
        add__14 = paddle._C_ops.add_(add__9, add__13)

        # pd_op.layer_norm: (-1x3136x32xf32, -3136xf32, -3136xf32) <- (-1x3136x32xf32, 32xf32, 32xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__14, parameter_36, parameter_37, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x3136x256xf32) <- (-1x3136x32xf32, 32x256xf32)
        matmul_12 = paddle.matmul(layer_norm_18, parameter_38, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x256xf32) <- (-1x3136x256xf32, 256xf32)
        add__15 = paddle._C_ops.add_(matmul_12, parameter_39)

        # pd_op.shape: (3xi32) <- (-1x3136x256xf32)
        shape_4 = paddle._C_ops.shape(add__15)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_4, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x256x3136xf32) <- (-1x3136x256xf32)
        transpose_15 = paddle._C_ops.transpose(add__15, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_8, constant_10, constant_5, constant_5]

        # pd_op.reshape_: (-1x256x56x56xf32, 0x-1x256x3136xf32) <- (-1x256x3136xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_15, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 256x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(reshape__22, parameter_40, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__16 = paddle._C_ops.add_(depthwise_conv2d_1, parameter_41)

        # pd_op.flatten_: (-1x256x3136xf32, None) <- (-1x256x56x56xf32)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__16, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x256xf32) <- (-1x256x3136xf32)
        transpose_16 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.gelu: (-1x3136x256xf32) <- (-1x3136x256xf32)
        gelu_1 = paddle._C_ops.gelu(transpose_16, False)

        # pd_op.matmul: (-1x3136x32xf32) <- (-1x3136x256xf32, 256x32xf32)
        matmul_13 = paddle.matmul(gelu_1, parameter_42, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, 32xf32)
        add__17 = paddle._C_ops.add_(matmul_13, parameter_43)

        # pd_op.add_: (-1x3136x32xf32) <- (-1x3136x32xf32, -1x3136x32xf32)
        add__18 = paddle._C_ops.add_(add__14, add__17)

        # pd_op.layer_norm: (-1x3136x32xf32, -3136xf32, -3136xf32) <- (-1x3136x32xf32, 32xf32, 32xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__18, parameter_44, parameter_45, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_0, constant_5, constant_5, constant_4]

        # pd_op.reshape_: (-1x56x56x32xf32, 0x-1x3136x32xf32) <- (-1x3136x32xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_21, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x32x56x56xf32) <- (-1x56x56x32xf32)
        transpose_17 = paddle._C_ops.transpose(reshape__24, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x32x56x56xf32, 64x32x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(transpose_17, parameter_46, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_3, parameter_47)

        # pd_op.flatten_: (-1x64x784xf32, None) <- (-1x64x28x28xf32)
        flatten__6, flatten__7 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__19, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x64xf32) <- (-1x64x784xf32)
        transpose_18 = paddle._C_ops.transpose(flatten__6, [0, 2, 1])

        # pd_op.layer_norm: (-1x784x64xf32, -784xf32, -784xf32) <- (-1x784x64xf32, 64xf32, 64xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_18, parameter_48, parameter_49, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x784x64xf32, -784xf32, -784xf32) <- (-1x784x64xf32, 64xf32, 64xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_24, parameter_50, parameter_51, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x64xf32)
        shape_5 = paddle._C_ops.shape(layer_norm_27)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_5, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x784x64xf32) <- (-1x784x64xf32, 64x64xf32)
        matmul_14 = paddle.matmul(layer_norm_27, parameter_52, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, 64xf32)
        add__20 = paddle._C_ops.add_(matmul_14, parameter_53)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_9, constant_11, constant_7, constant_4]

        # pd_op.reshape_: (-1x784x2x32xf32, 0x-1x784x64xf32) <- (-1x784x64xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__20, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x784x32xf32) <- (-1x784x2x32xf32)
        transpose_19 = paddle._C_ops.transpose(reshape__26, [0, 2, 1, 3])

        # pd_op.transpose: (-1x64x784xf32) <- (-1x784x64xf32)
        transpose_20 = paddle._C_ops.transpose(layer_norm_27, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_9, constant_12, constant_13, constant_13]

        # pd_op.reshape_: (-1x64x28x28xf32, 0x-1x64x784xf32) <- (-1x64x784xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x7x7xf32) <- (-1x64x28x28xf32, 64x64x4x4xf32)
        conv2d_4 = paddle._C_ops.conv2d(reshape__28, parameter_54, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x7x7xf32) <- (-1x64x7x7xf32, 1x64x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_4, parameter_55)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_15 = [slice_9, constant_12, constant_6]

        # pd_op.reshape_: (-1x64x49xf32, 0x-1x64x7x7xf32) <- (-1x64x7x7xf32, [1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__21, combine_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x64xf32) <- (-1x64x49xf32)
        transpose_21 = paddle._C_ops.transpose(reshape__30, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x64xf32, -49xf32, -49xf32) <- (-1x49x64xf32, 64xf32, 64xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_21, parameter_56, parameter_57, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x128xf32) <- (-1x49x64xf32, 64x128xf32)
        matmul_15 = paddle.matmul(layer_norm_30, parameter_58, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x128xf32) <- (-1x49x128xf32, 128xf32)
        add__22 = paddle._C_ops.add_(matmul_15, parameter_59)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_9, constant_6, constant_7, constant_7, constant_4]

        # pd_op.reshape_: (-1x49x2x2x32xf32, 0x-1x49x128xf32) <- (-1x49x128xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__22, combine_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x2x49x32xf32) <- (-1x49x2x2x32xf32)
        transpose_22 = paddle._C_ops.transpose(reshape__32, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x2x49x32xf32) <- (2x-1x2x49x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_22, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x2x49x32xf32) <- (2x-1x2x49x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_22, [0], constant_1, constant_8, [1], [0])

        # pd_op.transpose: (-1x2x32x49xf32) <- (-1x2x49x32xf32)
        transpose_23 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x784x49xf32) <- (-1x2x784x32xf32, -1x2x32x49xf32)
        matmul_16 = paddle.matmul(transpose_19, transpose_23, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x2x784x49xf32) <- (-1x2x784x49xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_16, constant_9, float('0'), True)

        # pd_op.softmax_: (-1x2x784x49xf32) <- (-1x2x784x49xf32)
        softmax__2 = paddle._C_ops.softmax_(scale__2, -1)

        # pd_op.matmul: (-1x2x784x32xf32) <- (-1x2x784x49xf32, -1x2x49x32xf32)
        matmul_17 = paddle.matmul(softmax__2, slice_11, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x784x2x32xf32) <- (-1x2x784x32xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_17, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_17 = [slice_9, constant_11, constant_12]

        # pd_op.reshape_: (-1x784x64xf32, 0x-1x784x2x32xf32) <- (-1x784x2x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, combine_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x64xf32) <- (-1x784x64xf32, 64x64xf32)
        matmul_18 = paddle.matmul(reshape__34, parameter_60, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, 64xf32)
        add__23 = paddle._C_ops.add_(matmul_18, parameter_61)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, -1x784x64xf32)
        add__24 = paddle._C_ops.add_(layer_norm_24, add__23)

        # pd_op.layer_norm: (-1x784x64xf32, -784xf32, -784xf32) <- (-1x784x64xf32, 64xf32, 64xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__24, parameter_62, parameter_63, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x512xf32) <- (-1x784x64xf32, 64x512xf32)
        matmul_19 = paddle.matmul(layer_norm_33, parameter_64, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x512xf32) <- (-1x784x512xf32, 512xf32)
        add__25 = paddle._C_ops.add_(matmul_19, parameter_65)

        # pd_op.shape: (3xi32) <- (-1x784x512xf32)
        shape_6 = paddle._C_ops.shape(add__25)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_6, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x512x784xf32) <- (-1x784x512xf32)
        transpose_25 = paddle._C_ops.transpose(add__25, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_12, constant_14, constant_13, constant_13]

        # pd_op.reshape_: (-1x512x28x28xf32, 0x-1x512x784xf32) <- (-1x512x784xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_25, combine_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 512x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(reshape__36, parameter_66, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__26 = paddle._C_ops.add_(depthwise_conv2d_2, parameter_67)

        # pd_op.flatten_: (-1x512x784xf32, None) <- (-1x512x28x28xf32)
        flatten__8, flatten__9 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__26, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x512xf32) <- (-1x512x784xf32)
        transpose_26 = paddle._C_ops.transpose(flatten__8, [0, 2, 1])

        # pd_op.gelu: (-1x784x512xf32) <- (-1x784x512xf32)
        gelu_2 = paddle._C_ops.gelu(transpose_26, False)

        # pd_op.matmul: (-1x784x64xf32) <- (-1x784x512xf32, 512x64xf32)
        matmul_20 = paddle.matmul(gelu_2, parameter_68, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, 64xf32)
        add__27 = paddle._C_ops.add_(matmul_20, parameter_69)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, -1x784x64xf32)
        add__28 = paddle._C_ops.add_(add__24, add__27)

        # pd_op.layer_norm: (-1x784x64xf32, -784xf32, -784xf32) <- (-1x784x64xf32, 64xf32, 64xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__28, parameter_70, parameter_71, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x64xf32)
        shape_7 = paddle._C_ops.shape(layer_norm_36)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_7, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x784x64xf32) <- (-1x784x64xf32, 64x64xf32)
        matmul_21 = paddle.matmul(layer_norm_36, parameter_72, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, 64xf32)
        add__29 = paddle._C_ops.add_(matmul_21, parameter_73)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_13, constant_11, constant_7, constant_4]

        # pd_op.reshape_: (-1x784x2x32xf32, 0x-1x784x64xf32) <- (-1x784x64xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__29, combine_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x784x32xf32) <- (-1x784x2x32xf32)
        transpose_27 = paddle._C_ops.transpose(reshape__38, [0, 2, 1, 3])

        # pd_op.transpose: (-1x64x784xf32) <- (-1x784x64xf32)
        transpose_28 = paddle._C_ops.transpose(layer_norm_36, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_13, constant_12, constant_13, constant_13]

        # pd_op.reshape_: (-1x64x28x28xf32, 0x-1x64x784xf32) <- (-1x64x784xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, combine_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x7x7xf32) <- (-1x64x28x28xf32, 64x64x4x4xf32)
        conv2d_5 = paddle._C_ops.conv2d(reshape__40, parameter_74, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x7x7xf32) <- (-1x64x7x7xf32, 1x64x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_5, parameter_75)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_21 = [slice_13, constant_12, constant_6]

        # pd_op.reshape_: (-1x64x49xf32, 0x-1x64x7x7xf32) <- (-1x64x7x7xf32, [1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__30, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x64xf32) <- (-1x64x49xf32)
        transpose_29 = paddle._C_ops.transpose(reshape__42, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x64xf32, -49xf32, -49xf32) <- (-1x49x64xf32, 64xf32, 64xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_29, parameter_76, parameter_77, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x128xf32) <- (-1x49x64xf32, 64x128xf32)
        matmul_22 = paddle.matmul(layer_norm_39, parameter_78, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x128xf32) <- (-1x49x128xf32, 128xf32)
        add__31 = paddle._C_ops.add_(matmul_22, parameter_79)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_22 = [slice_13, constant_6, constant_7, constant_7, constant_4]

        # pd_op.reshape_: (-1x49x2x2x32xf32, 0x-1x49x128xf32) <- (-1x49x128xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__31, combine_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x2x49x32xf32) <- (-1x49x2x2x32xf32)
        transpose_30 = paddle._C_ops.transpose(reshape__44, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x2x49x32xf32) <- (2x-1x2x49x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(transpose_30, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x2x49x32xf32) <- (2x-1x2x49x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_30, [0], constant_1, constant_8, [1], [0])

        # pd_op.transpose: (-1x2x32x49xf32) <- (-1x2x49x32xf32)
        transpose_31 = paddle._C_ops.transpose(slice_14, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x784x49xf32) <- (-1x2x784x32xf32, -1x2x32x49xf32)
        matmul_23 = paddle.matmul(transpose_27, transpose_31, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x2x784x49xf32) <- (-1x2x784x49xf32, 1xf32)
        scale__3 = paddle._C_ops.scale_(matmul_23, constant_9, float('0'), True)

        # pd_op.softmax_: (-1x2x784x49xf32) <- (-1x2x784x49xf32)
        softmax__3 = paddle._C_ops.softmax_(scale__3, -1)

        # pd_op.matmul: (-1x2x784x32xf32) <- (-1x2x784x49xf32, -1x2x49x32xf32)
        matmul_24 = paddle.matmul(softmax__3, slice_15, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x784x2x32xf32) <- (-1x2x784x32xf32)
        transpose_32 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_23 = [slice_13, constant_11, constant_12]

        # pd_op.reshape_: (-1x784x64xf32, 0x-1x784x2x32xf32) <- (-1x784x2x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_32, combine_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x64xf32) <- (-1x784x64xf32, 64x64xf32)
        matmul_25 = paddle.matmul(reshape__46, parameter_80, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, 64xf32)
        add__32 = paddle._C_ops.add_(matmul_25, parameter_81)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, -1x784x64xf32)
        add__33 = paddle._C_ops.add_(add__28, add__32)

        # pd_op.layer_norm: (-1x784x64xf32, -784xf32, -784xf32) <- (-1x784x64xf32, 64xf32, 64xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__33, parameter_82, parameter_83, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x512xf32) <- (-1x784x64xf32, 64x512xf32)
        matmul_26 = paddle.matmul(layer_norm_42, parameter_84, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x512xf32) <- (-1x784x512xf32, 512xf32)
        add__34 = paddle._C_ops.add_(matmul_26, parameter_85)

        # pd_op.shape: (3xi32) <- (-1x784x512xf32)
        shape_8 = paddle._C_ops.shape(add__34)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_8, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x512x784xf32) <- (-1x784x512xf32)
        transpose_33 = paddle._C_ops.transpose(add__34, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_16, constant_14, constant_13, constant_13]

        # pd_op.reshape_: (-1x512x28x28xf32, 0x-1x512x784xf32) <- (-1x512x784xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_33, combine_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 512x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(reshape__48, parameter_86, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__35 = paddle._C_ops.add_(depthwise_conv2d_3, parameter_87)

        # pd_op.flatten_: (-1x512x784xf32, None) <- (-1x512x28x28xf32)
        flatten__10, flatten__11 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__35, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x512xf32) <- (-1x512x784xf32)
        transpose_34 = paddle._C_ops.transpose(flatten__10, [0, 2, 1])

        # pd_op.gelu: (-1x784x512xf32) <- (-1x784x512xf32)
        gelu_3 = paddle._C_ops.gelu(transpose_34, False)

        # pd_op.matmul: (-1x784x64xf32) <- (-1x784x512xf32, 512x64xf32)
        matmul_27 = paddle.matmul(gelu_3, parameter_88, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, 64xf32)
        add__36 = paddle._C_ops.add_(matmul_27, parameter_89)

        # pd_op.add_: (-1x784x64xf32) <- (-1x784x64xf32, -1x784x64xf32)
        add__37 = paddle._C_ops.add_(add__33, add__36)

        # pd_op.layer_norm: (-1x784x64xf32, -784xf32, -784xf32) <- (-1x784x64xf32, 64xf32, 64xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__37, parameter_90, parameter_91, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_25 = [slice_0, constant_13, constant_13, constant_12]

        # pd_op.reshape_: (-1x28x28x64xf32, 0x-1x784x64xf32) <- (-1x784x64xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_45, combine_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x28x28xf32) <- (-1x28x28x64xf32)
        transpose_35 = paddle._C_ops.transpose(reshape__50, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x64x28x28xf32, 160x64x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(transpose_35, parameter_92, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__38 = paddle._C_ops.add_(conv2d_6, parameter_93)

        # pd_op.flatten_: (-1x160x196xf32, None) <- (-1x160x14x14xf32)
        flatten__12, flatten__13 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__38, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x160xf32) <- (-1x160x196xf32)
        transpose_36 = paddle._C_ops.transpose(flatten__12, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x160xf32, -196xf32, -196xf32) <- (-1x196x160xf32, 160xf32, 160xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_36, parameter_94, parameter_95, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x196x160xf32, -196xf32, -196xf32) <- (-1x196x160xf32, 160xf32, 160xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_48, parameter_96, parameter_97, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x160xf32)
        shape_9 = paddle._C_ops.shape(layer_norm_51)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_9, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x160xf32) <- (-1x196x160xf32, 160x160xf32)
        matmul_28 = paddle.matmul(layer_norm_51, parameter_98, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, 160xf32)
        add__39 = paddle._C_ops.add_(matmul_28, parameter_99)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_17, constant_15, constant_16, constant_4]

        # pd_op.reshape_: (-1x196x5x32xf32, 0x-1x196x160xf32) <- (-1x196x160xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__39, combine_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x32xf32) <- (-1x196x5x32xf32)
        transpose_37 = paddle._C_ops.transpose(reshape__52, [0, 2, 1, 3])

        # pd_op.transpose: (-1x160x196xf32) <- (-1x196x160xf32)
        transpose_38 = paddle._C_ops.transpose(layer_norm_51, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_27 = [slice_17, constant_17, constant_18, constant_18]

        # pd_op.reshape_: (-1x160x14x14xf32, 0x-1x160x196xf32) <- (-1x160x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_38, combine_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x160x7x7xf32) <- (-1x160x14x14xf32, 160x160x2x2xf32)
        conv2d_7 = paddle._C_ops.conv2d(reshape__54, parameter_100, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x7x7xf32) <- (-1x160x7x7xf32, 1x160x1x1xf32)
        add__40 = paddle._C_ops.add_(conv2d_7, parameter_101)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_28 = [slice_17, constant_17, constant_6]

        # pd_op.reshape_: (-1x160x49xf32, 0x-1x160x7x7xf32) <- (-1x160x7x7xf32, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__40, combine_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x160xf32) <- (-1x160x49xf32)
        transpose_39 = paddle._C_ops.transpose(reshape__56, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x160xf32, -49xf32, -49xf32) <- (-1x49x160xf32, 160xf32, 160xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_39, parameter_102, parameter_103, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x320xf32) <- (-1x49x160xf32, 160x320xf32)
        matmul_29 = paddle.matmul(layer_norm_54, parameter_104, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x320xf32) <- (-1x49x320xf32, 320xf32)
        add__41 = paddle._C_ops.add_(matmul_29, parameter_105)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_17, constant_6, constant_7, constant_16, constant_4]

        # pd_op.reshape_: (-1x49x2x5x32xf32, 0x-1x49x320xf32) <- (-1x49x320xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__41, combine_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x32xf32) <- (-1x49x2x5x32xf32)
        transpose_40 = paddle._C_ops.transpose(reshape__58, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x49x32xf32) <- (2x-1x5x49x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_40, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x49x32xf32) <- (2x-1x5x49x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_40, [0], constant_1, constant_8, [1], [0])

        # pd_op.transpose: (-1x5x32x49xf32) <- (-1x5x49x32xf32)
        transpose_41 = paddle._C_ops.transpose(slice_18, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf32) <- (-1x5x196x32xf32, -1x5x32x49xf32)
        matmul_30 = paddle.matmul(transpose_37, transpose_41, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x49xf32) <- (-1x5x196x49xf32, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_30, constant_9, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf32) <- (-1x5x196x49xf32)
        softmax__4 = paddle._C_ops.softmax_(scale__4, -1)

        # pd_op.matmul: (-1x5x196x32xf32) <- (-1x5x196x49xf32, -1x5x49x32xf32)
        matmul_31 = paddle.matmul(softmax__4, slice_19, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x32xf32) <- (-1x5x196x32xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_31, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_30 = [slice_17, constant_15, constant_17]

        # pd_op.reshape_: (-1x196x160xf32, 0x-1x196x5x32xf32) <- (-1x196x5x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_42, combine_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x160xf32) <- (-1x196x160xf32, 160x160xf32)
        matmul_32 = paddle.matmul(reshape__60, parameter_106, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, 160xf32)
        add__42 = paddle._C_ops.add_(matmul_32, parameter_107)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, -1x196x160xf32)
        add__43 = paddle._C_ops.add_(layer_norm_48, add__42)

        # pd_op.layer_norm: (-1x196x160xf32, -196xf32, -196xf32) <- (-1x196x160xf32, 160xf32, 160xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__43, parameter_108, parameter_109, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x640xf32) <- (-1x196x160xf32, 160x640xf32)
        matmul_33 = paddle.matmul(layer_norm_57, parameter_110, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x640xf32) <- (-1x196x640xf32, 640xf32)
        add__44 = paddle._C_ops.add_(matmul_33, parameter_111)

        # pd_op.shape: (3xi32) <- (-1x196x640xf32)
        shape_10 = paddle._C_ops.shape(add__44)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_10, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x640x196xf32) <- (-1x196x640xf32)
        transpose_43 = paddle._C_ops.transpose(add__44, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [slice_20, constant_19, constant_18, constant_18]

        # pd_op.reshape_: (-1x640x14x14xf32, 0x-1x640x196xf32) <- (-1x640x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_43, combine_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(reshape__62, parameter_112, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__45 = paddle._C_ops.add_(depthwise_conv2d_4, parameter_113)

        # pd_op.flatten_: (-1x640x196xf32, None) <- (-1x640x14x14xf32)
        flatten__14, flatten__15 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__45, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x640xf32) <- (-1x640x196xf32)
        transpose_44 = paddle._C_ops.transpose(flatten__14, [0, 2, 1])

        # pd_op.gelu: (-1x196x640xf32) <- (-1x196x640xf32)
        gelu_4 = paddle._C_ops.gelu(transpose_44, False)

        # pd_op.matmul: (-1x196x160xf32) <- (-1x196x640xf32, 640x160xf32)
        matmul_34 = paddle.matmul(gelu_4, parameter_114, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, 160xf32)
        add__46 = paddle._C_ops.add_(matmul_34, parameter_115)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, -1x196x160xf32)
        add__47 = paddle._C_ops.add_(add__43, add__46)

        # pd_op.layer_norm: (-1x196x160xf32, -196xf32, -196xf32) <- (-1x196x160xf32, 160xf32, 160xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__47, parameter_116, parameter_117, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x160xf32)
        shape_11 = paddle._C_ops.shape(layer_norm_60)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(shape_11, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x196x160xf32) <- (-1x196x160xf32, 160x160xf32)
        matmul_35 = paddle.matmul(layer_norm_60, parameter_118, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, 160xf32)
        add__48 = paddle._C_ops.add_(matmul_35, parameter_119)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_21, constant_15, constant_16, constant_4]

        # pd_op.reshape_: (-1x196x5x32xf32, 0x-1x196x160xf32) <- (-1x196x160xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__48, combine_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x32xf32) <- (-1x196x5x32xf32)
        transpose_45 = paddle._C_ops.transpose(reshape__64, [0, 2, 1, 3])

        # pd_op.transpose: (-1x160x196xf32) <- (-1x196x160xf32)
        transpose_46 = paddle._C_ops.transpose(layer_norm_60, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_21, constant_17, constant_18, constant_18]

        # pd_op.reshape_: (-1x160x14x14xf32, 0x-1x160x196xf32) <- (-1x160x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_46, combine_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x160x7x7xf32) <- (-1x160x14x14xf32, 160x160x2x2xf32)
        conv2d_8 = paddle._C_ops.conv2d(reshape__66, parameter_120, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x160x7x7xf32) <- (-1x160x7x7xf32, 1x160x1x1xf32)
        add__49 = paddle._C_ops.add_(conv2d_8, parameter_121)

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_34 = [slice_21, constant_17, constant_6]

        # pd_op.reshape_: (-1x160x49xf32, 0x-1x160x7x7xf32) <- (-1x160x7x7xf32, [1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__49, combine_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x160xf32) <- (-1x160x49xf32)
        transpose_47 = paddle._C_ops.transpose(reshape__68, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x160xf32, -49xf32, -49xf32) <- (-1x49x160xf32, 160xf32, 160xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_47, parameter_122, parameter_123, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x320xf32) <- (-1x49x160xf32, 160x320xf32)
        matmul_36 = paddle.matmul(layer_norm_63, parameter_124, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x320xf32) <- (-1x49x320xf32, 320xf32)
        add__50 = paddle._C_ops.add_(matmul_36, parameter_125)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_21, constant_6, constant_7, constant_16, constant_4]

        # pd_op.reshape_: (-1x49x2x5x32xf32, 0x-1x49x320xf32) <- (-1x49x320xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__50, combine_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x32xf32) <- (-1x49x2x5x32xf32)
        transpose_48 = paddle._C_ops.transpose(reshape__70, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x5x49x32xf32) <- (2x-1x5x49x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_48, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x5x49x32xf32) <- (2x-1x5x49x32xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_48, [0], constant_1, constant_8, [1], [0])

        # pd_op.transpose: (-1x5x32x49xf32) <- (-1x5x49x32xf32)
        transpose_49 = paddle._C_ops.transpose(slice_22, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf32) <- (-1x5x196x32xf32, -1x5x32x49xf32)
        matmul_37 = paddle.matmul(transpose_45, transpose_49, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x5x196x49xf32) <- (-1x5x196x49xf32, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_37, constant_9, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf32) <- (-1x5x196x49xf32)
        softmax__5 = paddle._C_ops.softmax_(scale__5, -1)

        # pd_op.matmul: (-1x5x196x32xf32) <- (-1x5x196x49xf32, -1x5x49x32xf32)
        matmul_38 = paddle.matmul(softmax__5, slice_23, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x196x5x32xf32) <- (-1x5x196x32xf32)
        transpose_50 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_36 = [slice_21, constant_15, constant_17]

        # pd_op.reshape_: (-1x196x160xf32, 0x-1x196x5x32xf32) <- (-1x196x5x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_50, combine_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x160xf32) <- (-1x196x160xf32, 160x160xf32)
        matmul_39 = paddle.matmul(reshape__72, parameter_126, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, 160xf32)
        add__51 = paddle._C_ops.add_(matmul_39, parameter_127)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, -1x196x160xf32)
        add__52 = paddle._C_ops.add_(add__47, add__51)

        # pd_op.layer_norm: (-1x196x160xf32, -196xf32, -196xf32) <- (-1x196x160xf32, 160xf32, 160xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__52, parameter_128, parameter_129, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x640xf32) <- (-1x196x160xf32, 160x640xf32)
        matmul_40 = paddle.matmul(layer_norm_66, parameter_130, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x640xf32) <- (-1x196x640xf32, 640xf32)
        add__53 = paddle._C_ops.add_(matmul_40, parameter_131)

        # pd_op.shape: (3xi32) <- (-1x196x640xf32)
        shape_12 = paddle._C_ops.shape(add__53)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_12, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x640x196xf32) <- (-1x196x640xf32)
        transpose_51 = paddle._C_ops.transpose(add__53, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_24, constant_19, constant_18, constant_18]

        # pd_op.reshape_: (-1x640x14x14xf32, 0x-1x640x196xf32) <- (-1x640x196xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_51, combine_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(reshape__74, parameter_132, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__54 = paddle._C_ops.add_(depthwise_conv2d_5, parameter_133)

        # pd_op.flatten_: (-1x640x196xf32, None) <- (-1x640x14x14xf32)
        flatten__16, flatten__17 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__54, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x640xf32) <- (-1x640x196xf32)
        transpose_52 = paddle._C_ops.transpose(flatten__16, [0, 2, 1])

        # pd_op.gelu: (-1x196x640xf32) <- (-1x196x640xf32)
        gelu_5 = paddle._C_ops.gelu(transpose_52, False)

        # pd_op.matmul: (-1x196x160xf32) <- (-1x196x640xf32, 640x160xf32)
        matmul_41 = paddle.matmul(gelu_5, parameter_134, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, 160xf32)
        add__55 = paddle._C_ops.add_(matmul_41, parameter_135)

        # pd_op.add_: (-1x196x160xf32) <- (-1x196x160xf32, -1x196x160xf32)
        add__56 = paddle._C_ops.add_(add__52, add__55)

        # pd_op.layer_norm: (-1x196x160xf32, -196xf32, -196xf32) <- (-1x196x160xf32, 160xf32, 160xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__56, parameter_136, parameter_137, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_0, constant_18, constant_18, constant_17]

        # pd_op.reshape_: (-1x14x14x160xf32, 0x-1x196x160xf32) <- (-1x196x160xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_69, combine_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x160x14x14xf32) <- (-1x14x14x160xf32)
        transpose_53 = paddle._C_ops.transpose(reshape__76, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x160x14x14xf32, 256x160x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(transpose_53, parameter_138, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__57 = paddle._C_ops.add_(conv2d_9, parameter_139)

        # pd_op.flatten_: (-1x256x49xf32, None) <- (-1x256x7x7xf32)
        flatten__18, flatten__19 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__57, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x256xf32) <- (-1x256x49xf32)
        transpose_54 = paddle._C_ops.transpose(flatten__18, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x256xf32, -49xf32, -49xf32) <- (-1x49x256xf32, 256xf32, 256xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_54, parameter_140, parameter_141, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x49x256xf32, -49xf32, -49xf32) <- (-1x49x256xf32, 256xf32, 256xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_72, parameter_142, parameter_143, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x256xf32)
        shape_13 = paddle._C_ops.shape(layer_norm_75)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_13, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x256xf32) <- (-1x49x256xf32, 256x256xf32)
        matmul_42 = paddle.matmul(layer_norm_75, parameter_144, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, 256xf32)
        add__58 = paddle._C_ops.add_(matmul_42, parameter_145)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_39 = [slice_25, constant_6, constant_20, constant_4]

        # pd_op.reshape_: (-1x49x8x32xf32, 0x-1x49x256xf32) <- (-1x49x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__58, combine_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x49x32xf32) <- (-1x49x8x32xf32)
        transpose_55 = paddle._C_ops.transpose(reshape__78, [0, 2, 1, 3])

        # pd_op.matmul: (-1x49x512xf32) <- (-1x49x256xf32, 256x512xf32)
        matmul_43 = paddle.matmul(layer_norm_75, parameter_146, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, 512xf32)
        add__59 = paddle._C_ops.add_(matmul_43, parameter_147)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_40 = [slice_25, constant_6, constant_7, constant_20, constant_4]

        # pd_op.reshape_: (-1x49x2x8x32xf32, 0x-1x49x512xf32) <- (-1x49x512xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__59, combine_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x49x32xf32) <- (-1x49x2x8x32xf32)
        transpose_56 = paddle._C_ops.transpose(reshape__80, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x32xf32) <- (2x-1x8x49x32xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_56, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x32xf32) <- (2x-1x8x49x32xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_56, [0], constant_1, constant_8, [1], [0])

        # pd_op.transpose: (-1x8x32x49xf32) <- (-1x8x49x32xf32)
        transpose_57 = paddle._C_ops.transpose(slice_26, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf32) <- (-1x8x49x32xf32, -1x8x32x49xf32)
        matmul_44 = paddle.matmul(transpose_55, transpose_57, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf32) <- (-1x8x49x49xf32, 1xf32)
        scale__6 = paddle._C_ops.scale_(matmul_44, constant_9, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf32) <- (-1x8x49x49xf32)
        softmax__6 = paddle._C_ops.softmax_(scale__6, -1)

        # pd_op.matmul: (-1x8x49x32xf32) <- (-1x8x49x49xf32, -1x8x49x32xf32)
        matmul_45 = paddle.matmul(softmax__6, slice_27, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x32xf32) <- (-1x8x49x32xf32)
        transpose_58 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_41 = [slice_25, constant_6, constant_10]

        # pd_op.reshape_: (-1x49x256xf32, 0x-1x49x8x32xf32) <- (-1x49x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_58, combine_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x256xf32) <- (-1x49x256xf32, 256x256xf32)
        matmul_46 = paddle.matmul(reshape__82, parameter_148, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, 256xf32)
        add__60 = paddle._C_ops.add_(matmul_46, parameter_149)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, -1x49x256xf32)
        add__61 = paddle._C_ops.add_(layer_norm_72, add__60)

        # pd_op.layer_norm: (-1x49x256xf32, -49xf32, -49xf32) <- (-1x49x256xf32, 256xf32, 256xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__61, parameter_150, parameter_151, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x1024xf32) <- (-1x49x256xf32, 256x1024xf32)
        matmul_47 = paddle.matmul(layer_norm_78, parameter_152, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1024xf32) <- (-1x49x1024xf32, 1024xf32)
        add__62 = paddle._C_ops.add_(matmul_47, parameter_153)

        # pd_op.shape: (3xi32) <- (-1x49x1024xf32)
        shape_14 = paddle._C_ops.shape(add__62)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_14, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x1024x49xf32) <- (-1x49x1024xf32)
        transpose_59 = paddle._C_ops.transpose(add__62, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_28, constant_21, constant_22, constant_22]

        # pd_op.reshape_: (-1x1024x7x7xf32, 0x-1x1024x49xf32) <- (-1x1024x49xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_59, combine_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1024x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(reshape__84, parameter_154, [1, 1], [1, 1], 'EXPLICIT', 1024, [1, 1], 'NCHW')

        # pd_op.add_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add__63 = paddle._C_ops.add_(depthwise_conv2d_6, parameter_155)

        # pd_op.flatten_: (-1x1024x49xf32, None) <- (-1x1024x7x7xf32)
        flatten__20, flatten__21 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__63, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x1024xf32) <- (-1x1024x49xf32)
        transpose_60 = paddle._C_ops.transpose(flatten__20, [0, 2, 1])

        # pd_op.gelu: (-1x49x1024xf32) <- (-1x49x1024xf32)
        gelu_6 = paddle._C_ops.gelu(transpose_60, False)

        # pd_op.matmul: (-1x49x256xf32) <- (-1x49x1024xf32, 1024x256xf32)
        matmul_48 = paddle.matmul(gelu_6, parameter_156, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, 256xf32)
        add__64 = paddle._C_ops.add_(matmul_48, parameter_157)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, -1x49x256xf32)
        add__65 = paddle._C_ops.add_(add__61, add__64)

        # pd_op.layer_norm: (-1x49x256xf32, -49xf32, -49xf32) <- (-1x49x256xf32, 256xf32, 256xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__65, parameter_158, parameter_159, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x256xf32)
        shape_15 = paddle._C_ops.shape(layer_norm_81)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_15, [0], constant_0, constant_1, [1], [0])

        # pd_op.matmul: (-1x49x256xf32) <- (-1x49x256xf32, 256x256xf32)
        matmul_49 = paddle.matmul(layer_norm_81, parameter_160, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, 256xf32)
        add__66 = paddle._C_ops.add_(matmul_49, parameter_161)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_43 = [slice_29, constant_6, constant_20, constant_4]

        # pd_op.reshape_: (-1x49x8x32xf32, 0x-1x49x256xf32) <- (-1x49x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__66, combine_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x49x32xf32) <- (-1x49x8x32xf32)
        transpose_61 = paddle._C_ops.transpose(reshape__86, [0, 2, 1, 3])

        # pd_op.matmul: (-1x49x512xf32) <- (-1x49x256xf32, 256x512xf32)
        matmul_50 = paddle.matmul(layer_norm_81, parameter_162, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x512xf32) <- (-1x49x512xf32, 512xf32)
        add__67 = paddle._C_ops.add_(matmul_50, parameter_163)

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_29, constant_6, constant_7, constant_20, constant_4]

        # pd_op.reshape_: (-1x49x2x8x32xf32, 0x-1x49x512xf32) <- (-1x49x512xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__67, combine_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x49x32xf32) <- (-1x49x2x8x32xf32)
        transpose_62 = paddle._C_ops.transpose(reshape__88, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x49x32xf32) <- (2x-1x8x49x32xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_62, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x49x32xf32) <- (2x-1x8x49x32xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_62, [0], constant_1, constant_8, [1], [0])

        # pd_op.transpose: (-1x8x32x49xf32) <- (-1x8x49x32xf32)
        transpose_63 = paddle._C_ops.transpose(slice_30, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf32) <- (-1x8x49x32xf32, -1x8x32x49xf32)
        matmul_51 = paddle.matmul(transpose_61, transpose_63, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x8x49x49xf32) <- (-1x8x49x49xf32, 1xf32)
        scale__7 = paddle._C_ops.scale_(matmul_51, constant_9, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf32) <- (-1x8x49x49xf32)
        softmax__7 = paddle._C_ops.softmax_(scale__7, -1)

        # pd_op.matmul: (-1x8x49x32xf32) <- (-1x8x49x49xf32, -1x8x49x32xf32)
        matmul_52 = paddle.matmul(softmax__7, slice_31, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x49x8x32xf32) <- (-1x8x49x32xf32)
        transpose_64 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_45 = [slice_29, constant_6, constant_10]

        # pd_op.reshape_: (-1x49x256xf32, 0x-1x49x8x32xf32) <- (-1x49x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_64, combine_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x256xf32) <- (-1x49x256xf32, 256x256xf32)
        matmul_53 = paddle.matmul(reshape__90, parameter_164, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, 256xf32)
        add__68 = paddle._C_ops.add_(matmul_53, parameter_165)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, -1x49x256xf32)
        add__69 = paddle._C_ops.add_(add__65, add__68)

        # pd_op.layer_norm: (-1x49x256xf32, -49xf32, -49xf32) <- (-1x49x256xf32, 256xf32, 256xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__69, parameter_166, parameter_167, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x1024xf32) <- (-1x49x256xf32, 256x1024xf32)
        matmul_54 = paddle.matmul(layer_norm_84, parameter_168, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x1024xf32) <- (-1x49x1024xf32, 1024xf32)
        add__70 = paddle._C_ops.add_(matmul_54, parameter_169)

        # pd_op.shape: (3xi32) <- (-1x49x1024xf32)
        shape_16 = paddle._C_ops.shape(add__70)

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_16, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x1024x49xf32) <- (-1x49x1024xf32)
        transpose_65 = paddle._C_ops.transpose(add__70, [0, 2, 1])

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_46 = [slice_32, constant_21, constant_22, constant_22]

        # pd_op.reshape_: (-1x1024x7x7xf32, 0x-1x1024x49xf32) <- (-1x1024x49xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_65, combine_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1024x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(reshape__92, parameter_170, [1, 1], [1, 1], 'EXPLICIT', 1024, [1, 1], 'NCHW')

        # pd_op.add_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add__71 = paddle._C_ops.add_(depthwise_conv2d_7, parameter_171)

        # pd_op.flatten_: (-1x1024x49xf32, None) <- (-1x1024x7x7xf32)
        flatten__22, flatten__23 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__71, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x1024xf32) <- (-1x1024x49xf32)
        transpose_66 = paddle._C_ops.transpose(flatten__22, [0, 2, 1])

        # pd_op.gelu: (-1x49x1024xf32) <- (-1x49x1024xf32)
        gelu_7 = paddle._C_ops.gelu(transpose_66, False)

        # pd_op.matmul: (-1x49x256xf32) <- (-1x49x1024xf32, 1024x256xf32)
        matmul_55 = paddle.matmul(gelu_7, parameter_172, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, 256xf32)
        add__72 = paddle._C_ops.add_(matmul_55, parameter_173)

        # pd_op.add_: (-1x49x256xf32) <- (-1x49x256xf32, -1x49x256xf32)
        add__73 = paddle._C_ops.add_(add__69, add__72)

        # pd_op.layer_norm: (-1x49x256xf32, -49xf32, -49xf32) <- (-1x49x256xf32, 256xf32, 256xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__73, parameter_174, parameter_175, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.mean: (-1x256xf32) <- (-1x49x256xf32)
        mean_0 = paddle._C_ops.mean(layer_norm_87, [1], False)

        # pd_op.matmul: (-1x1000xf32) <- (-1x256xf32, 256x1000xf32)
        matmul_56 = paddle.matmul(mean_0, parameter_176, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__74 = paddle._C_ops.add_(matmul_56, parameter_177)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__8 = paddle._C_ops.softmax_(add__74, -1)
        return softmax__8



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

    def forward(self, parameter_171, parameter_155, constant_22, constant_21, constant_20, parameter_139, parameter_133, parameter_121, parameter_113, constant_19, parameter_101, constant_18, constant_17, constant_16, constant_15, parameter_93, parameter_87, parameter_75, parameter_67, constant_14, parameter_55, constant_13, constant_12, constant_11, parameter_47, parameter_41, parameter_29, parameter_21, constant_10, constant_9, constant_8, constant_7, constant_6, parameter_9, constant_5, constant_4, constant_3, constant_2, parameter_1, constant_1, constant_0, parameter_0, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_22, parameter_23, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_37, parameter_36, parameter_38, parameter_39, parameter_40, parameter_42, parameter_43, parameter_45, parameter_44, parameter_46, parameter_49, parameter_48, parameter_51, parameter_50, parameter_52, parameter_53, parameter_54, parameter_57, parameter_56, parameter_58, parameter_59, parameter_60, parameter_61, parameter_63, parameter_62, parameter_64, parameter_65, parameter_66, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_95, parameter_94, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_114, parameter_115, parameter_117, parameter_116, parameter_118, parameter_119, parameter_120, parameter_123, parameter_122, parameter_124, parameter_125, parameter_126, parameter_127, parameter_129, parameter_128, parameter_130, parameter_131, parameter_132, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_141, parameter_140, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_156, parameter_157, parameter_159, parameter_158, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, feed_0):
        return self.builtin_module_1321_0_0(parameter_171, parameter_155, constant_22, constant_21, constant_20, parameter_139, parameter_133, parameter_121, parameter_113, constant_19, parameter_101, constant_18, constant_17, constant_16, constant_15, parameter_93, parameter_87, parameter_75, parameter_67, constant_14, parameter_55, constant_13, constant_12, constant_11, parameter_47, parameter_41, parameter_29, parameter_21, constant_10, constant_9, constant_8, constant_7, constant_6, parameter_9, constant_5, constant_4, constant_3, constant_2, parameter_1, constant_1, constant_0, parameter_0, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_22, parameter_23, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_37, parameter_36, parameter_38, parameter_39, parameter_40, parameter_42, parameter_43, parameter_45, parameter_44, parameter_46, parameter_49, parameter_48, parameter_51, parameter_50, parameter_52, parameter_53, parameter_54, parameter_57, parameter_56, parameter_58, parameter_59, parameter_60, parameter_61, parameter_63, parameter_62, parameter_64, parameter_65, parameter_66, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_95, parameter_94, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_114, parameter_115, parameter_117, parameter_116, parameter_118, parameter_119, parameter_120, parameter_123, parameter_122, parameter_124, parameter_125, parameter_126, parameter_127, parameter_129, parameter_128, parameter_130, parameter_131, parameter_132, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_141, parameter_140, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_156, parameter_157, parameter_159, parameter_158, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1321_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_171
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_22
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            # constant_21
            paddle.to_tensor([1024], dtype='int32').reshape([1]),
            # constant_20
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            # parameter_139
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([1, 640, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([1, 640, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_19
            paddle.to_tensor([640], dtype='int32').reshape([1]),
            # parameter_101
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_18
            paddle.to_tensor([14], dtype='int32').reshape([1]),
            # constant_17
            paddle.to_tensor([160], dtype='int32').reshape([1]),
            # constant_16
            paddle.to_tensor([5], dtype='int32').reshape([1]),
            # constant_15
            paddle.to_tensor([196], dtype='int32').reshape([1]),
            # parameter_93
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_14
            paddle.to_tensor([512], dtype='int32').reshape([1]),
            # parameter_55
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_13
            paddle.to_tensor([28], dtype='int32').reshape([1]),
            # constant_12
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_11
            paddle.to_tensor([784], dtype='int32').reshape([1]),
            # parameter_47
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_10
            paddle.to_tensor([256], dtype='int32').reshape([1]),
            # constant_9
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_8
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_7
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            # constant_6
            paddle.to_tensor([49], dtype='int32').reshape([1]),
            # parameter_9
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_5
            paddle.to_tensor([56], dtype='int32').reshape([1]),
            # constant_4
            paddle.to_tensor([32], dtype='int32').reshape([1]),
            # constant_3
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_2
            paddle.to_tensor([3136], dtype='int32').reshape([1]),
            # parameter_1
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_0
            paddle.uniform([32, 3, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32, 32, 8, 8], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32, 256], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256, 32], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([32, 32, 8, 8], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([32, 256], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256, 32], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64, 64, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64, 512], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([512, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([512, 64], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([64, 64, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([64, 512], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([512, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([512, 64], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([160, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([160, 160, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([160, 640], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([640, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([640, 160], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([160, 160, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([160, 640], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([640, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([640, 160], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([256, 160, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([256, 1024], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1024, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([256, 1024], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([1024, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([256, 1000], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_171
            paddle.static.InputSpec(shape=[1, 1024, 1, 1], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[1, 1024, 1, 1], dtype='float32'),
            # constant_22
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_21
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_20
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_139
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[1, 640, 1, 1], dtype='float32'),
            # constant_19
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_101
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            # constant_18
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_17
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_15
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_93
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_55
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_47
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_9
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 7, 7], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32, 32, 8, 8], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[32, 64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[32, 256], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256, 32], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[32, 32, 8, 8], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[32, 64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[32, 256], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256, 32], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[64, 64, 4, 4], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64, 512], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[512, 64], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[64, 64, 4, 4], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[64, 512], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[512, 64], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[160, 64, 3, 3], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[160, 160, 2, 2], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[160, 320], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[160, 640], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[640, 160], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[160, 160, 2, 2], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[160, 320], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[160, 640], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[640, 160], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[256, 160, 3, 3], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[256, 1024], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[1024, 1, 3, 3], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[256, 1024], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[1024, 1, 3, 3], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[256, 1000], dtype='float32'),
            # parameter_177
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