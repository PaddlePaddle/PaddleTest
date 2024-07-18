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
    return [1, 0, 10, 0, 2, 0, 437, 252][block_idx] - 1 # number-of-ops-in-block

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
    def pd_op_if_2464_0_0(self, full_0):

        # pd_op.assign: (xb) <- (xb)
        assign_0 = full_0
        return assign_0
    def pd_op_if_2464_1_0(self, parameter_0):
        return parameter_0
    def pd_op_if_2471_0_0(self, softmax__0, constant_0, argmax_0, constant_1, full_with_tensor_0, constant_2, full_with_tensor_1):

        # pd_op.max: (-1xf32) <- (-1x99xf32, 1xi64)
        max_0 = paddle._C_ops.max(softmax__0, constant_0, False)

        # pd_op.reshape: (-1x1xi64, 0x-1xi64) <- (-1xi64, 2xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(argmax_0, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x-1xi64, -1x1xi64]) <- (-1x-1xi64, -1x1xi64)
        combine_0 = [full_with_tensor_0, reshape_0]

        # pd_op.concat: (-1x2xi64) <- ([-1x-1xi64, -1x1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, constant_2)

        # pd_op.reshape: (-1x1xf32, 0x-1xf32) <- (-1xf32, 2xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(max_0, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x-1xf32, -1x1xf32]) <- (-1x-1xf32, -1x1xf32)
        combine_1 = [full_with_tensor_1, reshape_2]

        # pd_op.concat: (-1x2xf32) <- ([-1x-1xf32, -1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, constant_2)

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_0 = max_0

        # pd_op.assign_: (-1x2xi64) <- (-1x2xi64)
        assign__0 = paddle._C_ops.assign(concat_0)

        # pd_op.assign_: (-1x2xf32) <- (-1x2xf32)
        assign__1 = paddle._C_ops.assign(concat_1)
        return assign_0, assign__1, assign__0
    def pd_op_if_2471_1_0(self, parameter_0, parameter_1, parameter_2):
        return parameter_0, parameter_1, parameter_2
    def pd_op_if_2485_0_0(self, full_with_tensor_0, full_with_tensor_1):

        # pd_op.assign: (-1x1xi64) <- (-1x-1xi64)
        assign_0 = full_with_tensor_0

        # pd_op.assign: (-1x1xf32) <- (-1x-1xf32)
        assign_1 = full_with_tensor_1
        return assign_0, assign_1
    def pd_op_if_2485_1_0(self, parameter_0, parameter_1):
        return parameter_0, parameter_1
    def pd_op_while_2053_0_0(self, parameter_0, constant_0, constant_1, constant_2, parameter_1, constant_3, constant_4, constant_5, parameter_2, parameter_3, constant_6, constant_7, constant_8, constant_9, constant_10, constant_11, constant_12, parameter_4, parameter_5, parameter_6, parameter_7, layer_norm_3, parameter_8, parameter_9, parameter_10, parameter_11, constant_13, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_120, parameter_121, constant_14, constant_15, parameter_122, constant_16, parameter_123, parameter_124, constant_17, constant_18, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, full_5, parameter_131, logical_and__1, assign_value_2, full_with_tensor_0, assign_value_6, assign_value_1, assign_value_4, full_3, full_4, assign_value_5, full_with_tensor_4, assign_value_0, assign_value_3):

        # pd_op.embedding: (-1x1x512xf32) <- (-1x-1xi64, 99x512xf32)
        embedding_0 = paddle._C_ops.embedding(full_with_tensor_0, parameter_0, 0, False)

        # pd_op.scale_: (-1x1x512xf32) <- (-1x1x512xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(embedding_0, constant_0, float('0'), True)

        # pd_op.transpose: (1x-1x512xf32) <- (-1x1x512xf32)
        transpose_0 = paddle._C_ops.transpose(scale__0, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (1x-1x512xf32)
        shape_0 = paddle._C_ops.shape(transpose_0)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_1, constant_2, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_0 = [slice_0]

        # pd_op.slice: (-1x1x512xf32) <- (5000x1x512xf32, 1xi64, [xi32])
        slice_1 = paddle._C_ops.slice(parameter_1, [0], constant_1, combine_0, [-1], [])

        # pd_op.add: (-1x-1x512xf32) <- (1x-1x512xf32, -1x1x512xf32)
        add_0 = transpose_0 + slice_1

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_0, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x512xf32) <- (-1x-1x512xf32)
        transpose_1 = paddle._C_ops.transpose(dropout_0, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_1 = paddle._C_ops.shape(transpose_1)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_1, [0], constant_2, constant_4, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_1 = [slice_2, slice_2]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_0 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi32)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_0, stack_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('-inf'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_2 = [slice_2, slice_2]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_1 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi32)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(full_1, stack_1, paddle.float32)

        # pd_op.triu: (-1x-1xf32) <- (-1x-1xf32)
        triu_0 = paddle._C_ops.triu(full_with_tensor_2, 1)

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        add_1 = full_with_tensor_1 + triu_0

        # pd_op.unsqueeze_: (1x1x-1x-1xf32, None) <- (-1x-1xf32, 2xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_1, constant_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_2 = paddle._C_ops.shape(transpose_1)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_2, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_0 = paddle.matmul(transpose_1, parameter_2, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_2 = matmul_0 + parameter_3

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [constant_6, slice_3, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_2, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape__0, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(transpose_2, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(transpose_2, [0], constant_2, constant_4, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_2, [0], constant_4, constant_10, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_3 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_1 = paddle.matmul(slice_4, transpose_3, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_1, constant_11, float('0'), True)

        # pd_op.add: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1x1x-1x-1xf32)
        add_3 = scale_0 + unsqueeze__0

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(add_3, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_2 = paddle.matmul(softmax_0, slice_6, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_4 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_4 = [constant_6, slice_3, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_4, combine_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_3 = paddle.matmul(reshape__2, parameter_4, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_4 = matmul_3 + parameter_5

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_4, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_5 = transpose_1 + dropout_2

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_5, parameter_6, parameter_7, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_3 = paddle._C_ops.shape(layer_norm_0)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_3, [0], constant_2, constant_4, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_4 = paddle._C_ops.shape(layer_norm_3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_4, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_4 = paddle.matmul(layer_norm_0, parameter_8, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_6 = matmul_4 + parameter_9

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32)
        combine_5 = [constant_6, slice_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_6, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x-1x64xf32) <- (-1x-1x8x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape__4, [0, 2, 1, 3])

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_5 = paddle.matmul(layer_norm_3, parameter_10, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_7 = matmul_5 + parameter_11

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [constant_6, slice_8, constant_13, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x2x8x64xf32, 0x-1x-1x1024xf32) <- (-1x-1x1024xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_7, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x-1x64xf32) <- (-1x-1x2x8x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape__6, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_6, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_6, [0], constant_2, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_7 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_6 = paddle.matmul(transpose_5, transpose_7, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_6, constant_11, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_7 = paddle.matmul(softmax_1, slice_10, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_7, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_7 = [constant_6, slice_7, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_8, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_8 = paddle.matmul(reshape__8, parameter_12, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_8 = matmul_8 + parameter_13

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_8, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_9 = layer_norm_0 + dropout_4

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_9, parameter_14, parameter_15, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_9 = paddle.matmul(layer_norm_4, parameter_16, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_10 = matmul_9 + parameter_17

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_0 = paddle._C_ops.relu(add_10)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_0, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_10 = paddle.matmul(dropout_6, parameter_18, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_11 = matmul_10 + parameter_19

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_11, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_8, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_12 = layer_norm_4 + dropout_10

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_7, layer_norm_8, layer_norm_9 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_12, parameter_20, parameter_21, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_5 = paddle._C_ops.shape(layer_norm_7)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_5, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_11 = paddle.matmul(layer_norm_7, parameter_22, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_13 = matmul_11 + parameter_23

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [constant_6, slice_11, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_13, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape__10, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(transpose_9, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(transpose_9, [0], constant_2, constant_4, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(transpose_9, [0], constant_4, constant_10, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_10 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_12 = paddle.matmul(slice_12, transpose_10, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_12, constant_11, float('0'), True)

        # pd_op.add: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1x1x-1x-1xf32)
        add_14 = scale_2 + unsqueeze__0

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(add_14, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_13 = paddle.matmul(softmax_2, slice_14, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_13, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_9 = [constant_6, slice_11, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_11, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_14 = paddle.matmul(reshape__12, parameter_24, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_15 = matmul_14 + parameter_25

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_15, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_16 = layer_norm_7 + dropout_12

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_10, layer_norm_11, layer_norm_12 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_16, parameter_26, parameter_27, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_6 = paddle._C_ops.shape(layer_norm_10)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_6, [0], constant_2, constant_4, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_7 = paddle._C_ops.shape(layer_norm_3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_7, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_15 = paddle.matmul(layer_norm_10, parameter_28, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_17 = matmul_15 + parameter_29

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32)
        combine_10 = [constant_6, slice_15, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_17, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x-1x64xf32) <- (-1x-1x8x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape__14, [0, 2, 1, 3])

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_16 = paddle.matmul(layer_norm_3, parameter_30, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_18 = matmul_16 + parameter_31

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [constant_6, slice_16, constant_13, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x2x8x64xf32, 0x-1x-1x1024xf32) <- (-1x-1x1024xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_18, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x-1x64xf32) <- (-1x-1x2x8x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape__16, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_13, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_13, [0], constant_2, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_14 = paddle._C_ops.transpose(slice_17, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_17 = paddle.matmul(transpose_12, transpose_14, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_17, constant_11, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_18 = paddle.matmul(softmax_3, slice_18, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_18, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_12 = [constant_6, slice_15, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_15, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_19 = paddle.matmul(reshape__18, parameter_32, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_19 = matmul_19 + parameter_33

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_19, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_20 = layer_norm_10 + dropout_14

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_13, layer_norm_14, layer_norm_15 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_20, parameter_34, parameter_35, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_20 = paddle.matmul(layer_norm_13, parameter_36, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_21 = matmul_20 + parameter_37

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_1 = paddle._C_ops.relu(add_21)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_1, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_21 = paddle.matmul(dropout_16, parameter_38, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_22 = matmul_21 + parameter_39

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_22, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_18, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_23 = layer_norm_13 + dropout_20

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_16, layer_norm_17, layer_norm_18 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_23, parameter_40, parameter_41, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_8 = paddle._C_ops.shape(layer_norm_16)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(shape_8, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_22 = paddle.matmul(layer_norm_16, parameter_42, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_24 = matmul_22 + parameter_43

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [constant_6, slice_19, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_24, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape__20, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(transpose_16, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(transpose_16, [0], constant_2, constant_4, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_16, [0], constant_4, constant_10, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_17 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_23 = paddle.matmul(slice_20, transpose_17, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_23, constant_11, float('0'), True)

        # pd_op.add: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1x1x-1x-1xf32)
        add_25 = scale_4 + unsqueeze__0

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_4 = paddle._C_ops.softmax(add_25, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_24 = paddle.matmul(softmax_4, slice_22, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_14 = [constant_6, slice_19, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_18, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_25 = paddle.matmul(reshape__22, parameter_44, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_26 = matmul_25 + parameter_45

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_26, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_27 = layer_norm_16 + dropout_22

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_19, layer_norm_20, layer_norm_21 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_27, parameter_46, parameter_47, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_9 = paddle._C_ops.shape(layer_norm_19)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(shape_9, [0], constant_2, constant_4, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_10 = paddle._C_ops.shape(layer_norm_3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_10, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_26 = paddle.matmul(layer_norm_19, parameter_48, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_28 = matmul_26 + parameter_49

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32)
        combine_15 = [constant_6, slice_23, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_28, combine_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x-1x64xf32) <- (-1x-1x8x64xf32)
        transpose_19 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3])

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_27 = paddle.matmul(layer_norm_3, parameter_50, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_29 = matmul_27 + parameter_51

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [constant_6, slice_24, constant_13, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x2x8x64xf32, 0x-1x-1x1024xf32) <- (-1x-1x1024xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_29, combine_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x-1x64xf32) <- (-1x-1x2x8x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape__26, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_20, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_20, [0], constant_2, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_21 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_28 = paddle.matmul(transpose_19, transpose_21, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_28, constant_11, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_29 = paddle.matmul(softmax_5, slice_26, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_22 = paddle._C_ops.transpose(matmul_29, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_17 = [constant_6, slice_23, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_22, combine_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_30 = paddle.matmul(reshape__28, parameter_52, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_30 = matmul_30 + parameter_53

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_30, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_31 = layer_norm_19 + dropout_24

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_22, layer_norm_23, layer_norm_24 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_31, parameter_54, parameter_55, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_31 = paddle.matmul(layer_norm_22, parameter_56, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_32 = matmul_31 + parameter_57

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_2 = paddle._C_ops.relu(add_32)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_2, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_32 = paddle.matmul(dropout_26, parameter_58, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_33 = matmul_32 + parameter_59

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_33, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_28, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_34 = layer_norm_22 + dropout_30

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_25, layer_norm_26, layer_norm_27 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_34, parameter_60, parameter_61, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_11 = paddle._C_ops.shape(layer_norm_25)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(shape_11, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_33 = paddle.matmul(layer_norm_25, parameter_62, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_35 = matmul_33 + parameter_63

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [constant_6, slice_27, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_35, combine_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_23 = paddle._C_ops.transpose(reshape__30, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(transpose_23, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_23, [0], constant_2, constant_4, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_23, [0], constant_4, constant_10, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_24 = paddle._C_ops.transpose(slice_29, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_34 = paddle.matmul(slice_28, transpose_24, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_34, constant_11, float('0'), True)

        # pd_op.add: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1x1x-1x-1xf32)
        add_36 = scale_6 + unsqueeze__0

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_6 = paddle._C_ops.softmax(add_36, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_35 = paddle.matmul(softmax_6, slice_30, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_25 = paddle._C_ops.transpose(matmul_35, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_19 = [constant_6, slice_27, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_25, combine_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_36 = paddle.matmul(reshape__32, parameter_64, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_37 = matmul_36 + parameter_65

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_37, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_38 = layer_norm_25 + dropout_32

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_28, layer_norm_29, layer_norm_30 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_38, parameter_66, parameter_67, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_12 = paddle._C_ops.shape(layer_norm_28)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(shape_12, [0], constant_2, constant_4, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_13 = paddle._C_ops.shape(layer_norm_3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_13, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_37 = paddle.matmul(layer_norm_28, parameter_68, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_39 = matmul_37 + parameter_69

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32)
        combine_20 = [constant_6, slice_31, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_39, combine_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x-1x64xf32) <- (-1x-1x8x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape__34, [0, 2, 1, 3])

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_38 = paddle.matmul(layer_norm_3, parameter_70, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_40 = matmul_38 + parameter_71

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [constant_6, slice_32, constant_13, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x2x8x64xf32, 0x-1x-1x1024xf32) <- (-1x-1x1024xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_40, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x-1x64xf32) <- (-1x-1x2x8x64xf32)
        transpose_27 = paddle._C_ops.transpose(reshape__36, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_27, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_27, [0], constant_2, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_28 = paddle._C_ops.transpose(slice_33, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_39 = paddle.matmul(transpose_26, transpose_28, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_39, constant_11, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_7 = paddle._C_ops.softmax(scale_7, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_40 = paddle.matmul(softmax_7, slice_34, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_29 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_22 = [constant_6, slice_31, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_29, combine_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_41 = paddle.matmul(reshape__38, parameter_72, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_41 = matmul_41 + parameter_73

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_41, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_42 = layer_norm_28 + dropout_34

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_31, layer_norm_32, layer_norm_33 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_42, parameter_74, parameter_75, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_42 = paddle.matmul(layer_norm_31, parameter_76, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_43 = matmul_42 + parameter_77

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_3 = paddle._C_ops.relu(add_43)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_3, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_43 = paddle.matmul(dropout_36, parameter_78, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_44 = matmul_43 + parameter_79

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_44, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_38, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_45 = layer_norm_31 + dropout_40

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_34, layer_norm_35, layer_norm_36 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_45, parameter_80, parameter_81, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_14 = paddle._C_ops.shape(layer_norm_34)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(shape_14, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_44 = paddle.matmul(layer_norm_34, parameter_82, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_46 = matmul_44 + parameter_83

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [constant_6, slice_35, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_46, combine_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape__40, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(transpose_30, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(transpose_30, [0], constant_2, constant_4, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(transpose_30, [0], constant_4, constant_10, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_31 = paddle._C_ops.transpose(slice_37, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_45 = paddle.matmul(slice_36, transpose_31, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_45, constant_11, float('0'), True)

        # pd_op.add: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1x1x-1x-1xf32)
        add_47 = scale_8 + unsqueeze__0

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_8 = paddle._C_ops.softmax(add_47, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_46 = paddle.matmul(softmax_8, slice_38, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_32 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_24 = [constant_6, slice_35, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_32, combine_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_47 = paddle.matmul(reshape__42, parameter_84, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_48 = matmul_47 + parameter_85

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_48, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_49 = layer_norm_34 + dropout_42

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_37, layer_norm_38, layer_norm_39 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_49, parameter_86, parameter_87, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_15 = paddle._C_ops.shape(layer_norm_37)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(shape_15, [0], constant_2, constant_4, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_16 = paddle._C_ops.shape(layer_norm_3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(shape_16, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_48 = paddle.matmul(layer_norm_37, parameter_88, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_50 = matmul_48 + parameter_89

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32)
        combine_25 = [constant_6, slice_39, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_50, combine_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x-1x64xf32) <- (-1x-1x8x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape__44, [0, 2, 1, 3])

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_49 = paddle.matmul(layer_norm_3, parameter_90, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_51 = matmul_49 + parameter_91

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [constant_6, slice_40, constant_13, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x2x8x64xf32, 0x-1x-1x1024xf32) <- (-1x-1x1024xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_51, combine_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x-1x64xf32) <- (-1x-1x2x8x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape__46, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(transpose_34, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(transpose_34, [0], constant_2, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_35 = paddle._C_ops.transpose(slice_41, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_50 = paddle.matmul(transpose_33, transpose_35, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_50, constant_11, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_9 = paddle._C_ops.softmax(scale_9, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_51 = paddle.matmul(softmax_9, slice_42, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_36 = paddle._C_ops.transpose(matmul_51, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_27 = [constant_6, slice_39, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_36, combine_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_52 = paddle.matmul(reshape__48, parameter_92, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_52 = matmul_52 + parameter_93

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_52, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_53 = layer_norm_37 + dropout_44

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_40, layer_norm_41, layer_norm_42 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_53, parameter_94, parameter_95, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_53 = paddle.matmul(layer_norm_40, parameter_96, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_54 = matmul_53 + parameter_97

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_4 = paddle._C_ops.relu(add_54)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_4, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_54 = paddle.matmul(dropout_46, parameter_98, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_55 = matmul_54 + parameter_99

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_55, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_48, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_56 = layer_norm_40 + dropout_50

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_43, layer_norm_44, layer_norm_45 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_56, parameter_100, parameter_101, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_17 = paddle._C_ops.shape(layer_norm_43)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(shape_17, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_55 = paddle.matmul(layer_norm_43, parameter_102, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_57 = matmul_55 + parameter_103

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_28 = [constant_6, slice_43, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_57, combine_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape__50, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(transpose_37, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(transpose_37, [0], constant_2, constant_4, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(transpose_37, [0], constant_4, constant_10, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_38 = paddle._C_ops.transpose(slice_45, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_56 = paddle.matmul(slice_44, transpose_38, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_56, constant_11, float('0'), True)

        # pd_op.add: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1x1x-1x-1xf32)
        add_58 = scale_10 + unsqueeze__0

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_10 = paddle._C_ops.softmax(add_58, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_57 = paddle.matmul(softmax_10, slice_46, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_57, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_29 = [constant_6, slice_43, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_39, combine_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_58 = paddle.matmul(reshape__52, parameter_104, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_59 = matmul_58 + parameter_105

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_59, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_60 = layer_norm_43 + dropout_52

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_46, layer_norm_47, layer_norm_48 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_60, parameter_106, parameter_107, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_18 = paddle._C_ops.shape(layer_norm_46)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(shape_18, [0], constant_2, constant_4, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_19 = paddle._C_ops.shape(layer_norm_3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(shape_19, [0], constant_2, constant_4, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_59 = paddle.matmul(layer_norm_46, parameter_108, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_61 = matmul_59 + parameter_109

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32)
        combine_30 = [constant_6, slice_47, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_61, combine_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x-1x64xf32) <- (-1x-1x8x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape__54, [0, 2, 1, 3])

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_60 = paddle.matmul(layer_norm_3, parameter_110, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_62 = matmul_60 + parameter_111

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [constant_6, slice_48, constant_13, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x2x8x64xf32, 0x-1x-1x1024xf32) <- (-1x-1x1024xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_62, combine_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x-1x64xf32) <- (-1x-1x2x8x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape__56, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(transpose_41, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(transpose_41, [0], constant_2, constant_4, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_42 = paddle._C_ops.transpose(slice_49, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_61 = paddle.matmul(transpose_40, transpose_42, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_61, constant_11, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_11 = paddle._C_ops.softmax(scale_11, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_62 = paddle.matmul(softmax_11, slice_50, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_62, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_32 = [constant_6, slice_47, constant_12]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_43, combine_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_63 = paddle.matmul(reshape__58, parameter_112, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_63 = matmul_63 + parameter_113

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_63, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_64 = layer_norm_46 + dropout_54

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_49, layer_norm_50, layer_norm_51 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_64, parameter_114, parameter_115, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_64 = paddle.matmul(layer_norm_49, parameter_116, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_65 = matmul_64 + parameter_117

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_5 = paddle._C_ops.relu(add_65)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_5, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_65 = paddle.matmul(dropout_56, parameter_118, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_66 = matmul_65 + parameter_119

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_66, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_58, None, constant_3, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_67 = layer_norm_49 + dropout_60

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_52, layer_norm_53, layer_norm_54 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_67, parameter_120, parameter_121, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.slice: (-1x512xf32) <- (-1x-1x512xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(layer_norm_52, [1], constant_14, constant_15, [1], [1])

        # pd_op.matmul: (-1x99xf32) <- (-1x512xf32, 512x99xf32)
        matmul_66 = paddle.matmul(slice_51, parameter_122, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x99xf32) <- (-1x99xf32)
        softmax__0 = paddle._C_ops.softmax(matmul_66, -1)

        # pd_op.argmax: (-1xi64) <- (-1x99xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(softmax__0, constant_16, False, False, paddle.int64)

        # pd_op.shape: (1xi32) <- (-1xi64)
        shape_20 = paddle._C_ops.shape(argmax_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('3'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full_with_tensor: (-1xi64) <- (1xf32, 1xi32)
        full_with_tensor_3 = paddle._C_ops.full_with_tensor(full_2, shape_20, paddle.int64)

        # pd_op.equal_all: (xb) <- (-1xi64, -1xi64)
        equal_all_0 = paddle._C_ops.equal_all(argmax_0, full_with_tensor_3)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(equal_all_0)

        # pd_op.if: (xb) <- (xb)
        if logical_not_0:
            if_0, = self.pd_op_if_2464_0_0(full_3)
        else:
            if_0, = self.pd_op_if_2464_1_0(parameter_123)

        # pd_op.cast: (xi32) <- (xb)
        cast_0 = paddle._C_ops.cast(equal_all_0, paddle.int32)

        # pd_op.select_input: (xb) <- (xi32, xb, xb)
        select_input_0 = [if_0, parameter_124][int(cast_0)]

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_1 = paddle._C_ops.logical_not(select_input_0)

        # pd_op.if: (-1xf32, -1x2xf32, -1x2xi64) <- (xb)
        if logical_not_1:
            if_1, if_2, if_3, = self.pd_op_if_2471_0_0(softmax__0, constant_14, argmax_0, constant_17, full_with_tensor_0, constant_18, full_with_tensor_4)
        else:
            if_1, if_2, if_3, = self.pd_op_if_2471_1_0(parameter_125, parameter_126, parameter_127)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_2 = paddle._C_ops.logical_not(logical_not_1)

        # pd_op.if: (-1x1xi64, -1x1xf32) <- (xb)
        if logical_not_2:
            if_4, if_5, = self.pd_op_if_2485_0_0(full_with_tensor_0, full_with_tensor_4)
        else:
            if_4, if_5, = self.pd_op_if_2485_1_0(parameter_128, parameter_129)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(logical_not_1, paddle.int32)

        # pd_op.select_input: (-1xf32) <- (xi32, -1xf32, -1xf32)
        select_input_1 = [parameter_130, if_1][int(cast_1)]

        # pd_op.select_input: (-1x1xi64) <- (xi32, -1x1xi64, -1x2xi64)
        select_input_2 = [if_4, if_3][int(cast_1)]

        # pd_op.select_input: (-1x1xf32) <- (xi32, -1x1xf32, -1x2xf32)
        select_input_3 = [if_5, if_2][int(cast_1)]

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_12 = paddle._C_ops.scale(full_4, full_5, float('1'), True)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = scale_12 < parameter_131

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_3 = paddle._C_ops.logical_not(select_input_0)

        # pd_op.logical_and_: (xb) <- (xb, xb)
        logical_and__0 = paddle._C_ops.logical_and(less_than_0, logical_not_3)

        # pd_op.assign_out_: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        assign_out__0 = paddle._C_ops.assign_out_(layer_norm_52, assign_value_0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__1 = paddle._C_ops.assign_out_(select_input_0, full_3)

        # pd_op.assign_out_: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        assign_out__2 = paddle._C_ops.assign_out_(transpose_1, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(scale_12, full_4)

        # pd_op.assign_out_: (-1x-1xi64) <- (-1x1xi64, -1x-1xi64)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_2, full_with_tensor_0)

        # pd_op.assign_out_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        assign_out__5 = paddle._C_ops.assign_out_(slice_51, assign_value_2)

        # pd_op.assign_out_: (-1x99xf32) <- (-1x99xf32, -1x99xf32)
        assign_out__6 = paddle._C_ops.assign_out_(softmax__0, assign_value_3)

        # pd_op.assign_out_: (-1xi64) <- (-1xi64, -1xi64)
        assign_out__7 = paddle._C_ops.assign_out_(argmax_0, assign_value_4)

        # pd_op.assign_out_: (-1xf32) <- (-1xf32, -1xf32)
        assign_out__8 = paddle._C_ops.assign_out_(select_input_1, assign_value_5)

        # pd_op.assign_out_: (1x1x-1x-1xf32) <- (1x1x-1x-1xf32, 1x1x-1x-1xf32)
        assign_out__9 = paddle._C_ops.assign_out_(unsqueeze__0, assign_value_6)

        # pd_op.assign_out_: (-1x-1xf32) <- (-1x1xf32, -1x-1xf32)
        assign_out__10 = paddle._C_ops.assign_out_(select_input_3, full_with_tensor_4)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__11 = paddle._C_ops.assign_out_(logical_and__0, logical_and__1)
        return assign_out__11, assign_out__5, assign_out__4, assign_out__9, assign_out__2, assign_out__7, assign_out__1, assign_out__3, assign_out__8, assign_out__10, assign_out__0, assign_out__6
    def builtin_module_1564_0_0(self, parameter_214, parameter_215, parameter_211, parameter_212, parameter_213, constant_19, constant_18, parameter_209, parameter_210, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, parameter_216, parameter_86, parameter_85, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_24, parameter_23, parameter_25, parameter_26, parameter_27, parameter_28, parameter_30, parameter_29, parameter_31, parameter_32, parameter_33, parameter_34, parameter_36, parameter_35, parameter_37, parameter_38, parameter_39, parameter_40, parameter_42, parameter_41, parameter_43, parameter_44, parameter_45, parameter_46, parameter_48, parameter_47, parameter_49, parameter_50, parameter_51, parameter_52, parameter_54, parameter_53, parameter_55, parameter_56, parameter_57, parameter_58, parameter_60, parameter_59, parameter_61, parameter_62, parameter_63, parameter_64, parameter_66, parameter_65, parameter_67, parameter_68, parameter_69, parameter_70, parameter_72, parameter_71, parameter_73, parameter_74, parameter_75, parameter_76, parameter_78, parameter_77, parameter_79, parameter_80, parameter_81, parameter_82, parameter_84, parameter_83, parameter_109, parameter_134, parameter_94, parameter_197, parameter_166, parameter_194, parameter_172, parameter_133, parameter_195, parameter_167, parameter_208, parameter_122, parameter_159, parameter_87, parameter_136, parameter_149, parameter_121, parameter_181, parameter_89, parameter_114, parameter_102, parameter_152, parameter_198, parameter_128, parameter_174, parameter_161, parameter_145, parameter_199, parameter_137, parameter_130, parameter_99, parameter_126, parameter_148, parameter_178, parameter_111, parameter_158, parameter_202, parameter_110, parameter_135, parameter_162, parameter_200, parameter_104, parameter_168, parameter_103, parameter_96, parameter_173, parameter_189, parameter_147, parameter_129, parameter_92, parameter_95, parameter_140, parameter_106, parameter_155, parameter_177, parameter_160, parameter_93, parameter_176, parameter_112, parameter_97, parameter_98, parameter_164, parameter_196, parameter_188, parameter_154, parameter_107, parameter_157, parameter_182, parameter_125, parameter_120, parameter_191, parameter_165, parameter_123, parameter_205, parameter_193, parameter_131, parameter_142, parameter_144, parameter_90, parameter_88, parameter_113, parameter_190, parameter_116, parameter_169, parameter_91, parameter_108, parameter_192, parameter_139, parameter_170, parameter_127, parameter_100, parameter_151, parameter_156, parameter_163, parameter_101, parameter_138, parameter_141, parameter_105, parameter_132, parameter_184, parameter_119, parameter_179, parameter_118, parameter_143, parameter_180, parameter_201, parameter_207, parameter_175, parameter_187, parameter_124, parameter_204, parameter_153, parameter_146, parameter_117, parameter_171, parameter_203, parameter_206, parameter_115, parameter_186, parameter_185, parameter_150, parameter_183, feed_0):

        # pd_op.conv2d: (-1x32x16x50xf32) <- (-1x1x32x100xf32, 32x1x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x32x16x50xf32) <- (-1x32x16x50xf32, 1x32x1x1xf32)
        add__0 = paddle._C_ops.add(conv2d_0, parameter_1)

        # pd_op.relu_: (-1x32x16x50xf32) <- (-1x32x16x50xf32)
        relu__0 = paddle._C_ops.relu(add__0)

        # pd_op.batch_norm_: (-1x32x16x50xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x16x50xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(relu__0, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x8x25xf32) <- (-1x32x16x50xf32, 64x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(batch_norm__0, parameter_6, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x8x25xf32) <- (-1x64x8x25xf32, 1x64x1x1xf32)
        add__1 = paddle._C_ops.add(conv2d_1, parameter_7)

        # pd_op.relu_: (-1x64x8x25xf32) <- (-1x64x8x25xf32)
        relu__1 = paddle._C_ops.relu(add__1)

        # pd_op.batch_norm_: (-1x64x8x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x8x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(relu__1, parameter_8, parameter_9, parameter_10, parameter_11, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.transpose: (-1x25x8x64xf32) <- (-1x64x8x25xf32)
        transpose_0 = paddle._C_ops.transpose(batch_norm__6, [0, 3, 2, 1])

        # pd_op.shape: (4xi32) <- (-1x25x8x64xf32)
        shape_0 = paddle._C_ops.shape(transpose_0)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_0, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_0, [0], constant_2, constant_3, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_0, [0], constant_3, constant_4, [1], [0])

        # pd_op.multiply_: (xi32) <- (xi32, xi32)
        multiply__0 = paddle._C_ops.multiply(slice_2, slice_3)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_0 = [slice_0, slice_1, multiply__0]

        # pd_op.reshape_: (-1x-1x-1xf32, 0x-1x25x8x64xf32) <- (-1x25x8x64xf32, [xi32, xi32, xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_0, combine_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x-1x-1xf32)
        shape_1 = paddle._C_ops.shape(reshape__0)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_1, [0], constant_0, constant_1, [1], [0])

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (-1x-1x-1xf32)
        shape_2 = paddle._C_ops.shape(transpose_1)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_2, [0], constant_0, constant_1, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_1 = [slice_5]

        # pd_op.slice: (-1x1x512xf32) <- (5000x1x512xf32, 1xi64, [xi32])
        slice_6 = paddle._C_ops.slice(parameter_12, [0], constant_0, combine_1, [-1], [])

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x-1xf32, -1x1x512xf32)
        add_0 = transpose_1 + slice_6

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_0, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x512xf32) <- (-1x-1x512xf32)
        transpose_2 = paddle._C_ops.transpose(dropout_0, [1, 0, 2])

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_3 = paddle._C_ops.shape(transpose_2)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_3, [0], constant_1, constant_2, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_0 = paddle.matmul(transpose_2, parameter_13, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_1 = matmul_0 + parameter_14

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [constant_6, slice_7, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_1, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_3 = paddle._C_ops.transpose(reshape__2, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(transpose_3, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_3, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_3, [0], constant_2, constant_3, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_4 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_1 = paddle.matmul(slice_8, transpose_4, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_1, constant_10, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_2 = paddle.matmul(softmax_0, slice_10, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_3 = [constant_6, slice_7, constant_11]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_5, combine_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_3 = paddle.matmul(reshape__4, parameter_15, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_2 = matmul_3 + parameter_16

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_2, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_3 = transpose_2 + dropout_2

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_3, parameter_17, parameter_18, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_4 = paddle.matmul(layer_norm_0, parameter_19, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_4 = matmul_4 + parameter_20

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_0 = paddle._C_ops.relu(add_4)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_0, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_5 = paddle.matmul(dropout_4, parameter_21, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_5 = matmul_5 + parameter_22

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_5, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_6, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_6 = layer_norm_0 + dropout_8

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_6, parameter_23, parameter_24, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_4 = paddle._C_ops.shape(layer_norm_3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_4, [0], constant_1, constant_2, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_6 = paddle.matmul(layer_norm_3, parameter_25, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_7 = matmul_6 + parameter_26

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_4 = [constant_6, slice_11, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_7, combine_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape__6, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(transpose_6, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(transpose_6, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(transpose_6, [0], constant_2, constant_3, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_7 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_7 = paddle.matmul(slice_12, transpose_7, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_7, constant_10, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_8 = paddle.matmul(softmax_1, slice_14, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_5 = [constant_6, slice_11, constant_11]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_8, combine_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_9 = paddle.matmul(reshape__8, parameter_27, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_8 = matmul_9 + parameter_28

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_8, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_9 = layer_norm_3 + dropout_10

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_9, parameter_29, parameter_30, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_10 = paddle.matmul(layer_norm_6, parameter_31, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_10 = matmul_10 + parameter_32

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_1 = paddle._C_ops.relu(add_10)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_1, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_11 = paddle.matmul(dropout_12, parameter_33, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_11 = matmul_11 + parameter_34

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_11, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_14, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_12 = layer_norm_6 + dropout_16

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_12, parameter_35, parameter_36, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_5 = paddle._C_ops.shape(layer_norm_9)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_5, [0], constant_1, constant_2, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_12 = paddle.matmul(layer_norm_9, parameter_37, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_13 = matmul_12 + parameter_38

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [constant_6, slice_15, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_13, combine_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape__10, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_9, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_9, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_9, [0], constant_2, constant_3, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_10 = paddle._C_ops.transpose(slice_17, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_13 = paddle.matmul(slice_16, transpose_10, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_13, constant_10, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_14 = paddle.matmul(softmax_2, slice_18, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_7 = [constant_6, slice_15, constant_11]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_11, combine_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_15 = paddle.matmul(reshape__12, parameter_39, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_14 = matmul_15 + parameter_40

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_14, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_15 = layer_norm_9 + dropout_18

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_15, parameter_41, parameter_42, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_16 = paddle.matmul(layer_norm_12, parameter_43, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_16 = matmul_16 + parameter_44

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_2 = paddle._C_ops.relu(add_16)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_2, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_17 = paddle.matmul(dropout_20, parameter_45, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_17 = matmul_17 + parameter_46

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_17, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_22, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_18 = layer_norm_12 + dropout_24

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_18, parameter_47, parameter_48, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_6 = paddle._C_ops.shape(layer_norm_15)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(shape_6, [0], constant_1, constant_2, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_18 = paddle.matmul(layer_norm_15, parameter_49, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_19 = matmul_18 + parameter_50

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [constant_6, slice_19, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_19, combine_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape__14, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(transpose_12, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(transpose_12, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_12, [0], constant_2, constant_3, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_13 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_19 = paddle.matmul(slice_20, transpose_13, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_19, constant_10, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_20 = paddle.matmul(softmax_3, slice_22, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_9 = [constant_6, slice_19, constant_11]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_14, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_21 = paddle.matmul(reshape__16, parameter_51, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_20 = matmul_21 + parameter_52

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_20, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_21 = layer_norm_15 + dropout_26

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_21, parameter_53, parameter_54, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_22 = paddle.matmul(layer_norm_18, parameter_55, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_22 = matmul_22 + parameter_56

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_3 = paddle._C_ops.relu(add_22)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_3, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_23 = paddle.matmul(dropout_28, parameter_57, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_23 = matmul_23 + parameter_58

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_23, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_30, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_24 = layer_norm_18 + dropout_32

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_24, parameter_59, parameter_60, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_7 = paddle._C_ops.shape(layer_norm_21)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(shape_7, [0], constant_1, constant_2, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_24 = paddle.matmul(layer_norm_21, parameter_61, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_25 = matmul_24 + parameter_62

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [constant_6, slice_23, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_25, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_15 = paddle._C_ops.transpose(reshape__18, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_15, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_15, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_15, [0], constant_2, constant_3, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_16 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_25 = paddle.matmul(slice_24, transpose_16, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_25, constant_10, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_26 = paddle.matmul(softmax_4, slice_26, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_11 = [constant_6, slice_23, constant_11]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_17, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_27 = paddle.matmul(reshape__20, parameter_63, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_26 = matmul_27 + parameter_64

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_26, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_27 = layer_norm_21 + dropout_34

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_27, parameter_65, parameter_66, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_28 = paddle.matmul(layer_norm_24, parameter_67, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_28 = matmul_28 + parameter_68

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_4 = paddle._C_ops.relu(add_28)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_4, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_29 = paddle.matmul(dropout_36, parameter_69, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_29 = matmul_29 + parameter_70

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_29, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_38, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_30 = layer_norm_24 + dropout_40

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_30, parameter_71, parameter_72, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_8 = paddle._C_ops.shape(layer_norm_27)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(shape_8, [0], constant_1, constant_2, [1], [0])

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x512xf32, 512x1536xf32)
        matmul_30 = paddle.matmul(layer_norm_27, parameter_73, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_31 = matmul_30 + parameter_74

        # builtin.combine: ([1xi32, xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [constant_6, slice_27, constant_7, constant_8, constant_9]

        # pd_op.reshape_: (-1x-1x3x8x64xf32, 0x-1x-1x1536xf32) <- (-1x-1x1536xf32, [1xi32, xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape(add_31, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x64xf32) <- (-1x-1x3x8x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape__22, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(transpose_18, [0], constant_0, constant_1, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_18, [0], constant_1, constant_2, [1], [0])

        # pd_op.slice: (-1x8x-1x64xf32) <- (3x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_18, [0], constant_2, constant_3, [1], [0])

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_19 = paddle._C_ops.transpose(slice_29, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_31 = paddle.matmul(slice_28, transpose_19, transpose_x=False, transpose_y=False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_31, constant_10, float('0'), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_32 = paddle.matmul(softmax_5, slice_30, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_20 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])

        # builtin.combine: ([1xi32, xi32, 1xi32]) <- (1xi32, xi32, 1xi32)
        combine_13 = [constant_6, slice_27, constant_11]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x-1x8x64xf32) <- (-1x-1x8x64xf32, [1xi32, xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_20, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_33 = paddle.matmul(reshape__24, parameter_75, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_32 = matmul_33 + parameter_76

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_32, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_33 = layer_norm_27 + dropout_42

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_33, parameter_77, parameter_78, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_34 = paddle.matmul(layer_norm_30, parameter_79, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_34 = matmul_34 + parameter_80

        # pd_op.relu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        relu_5 = paddle._C_ops.relu(add_34)

        # pd_op.dropout: (-1x-1x1024xf32, None) <- (-1x-1x1024xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu_5, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x1024xf32, 1024x512xf32)
        matmul_35 = paddle.matmul(dropout_44, parameter_81, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_35 = matmul_35 + parameter_82

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(paddle._C_ops.dropout(add_35, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.dropout: (-1x-1x512xf32, None) <- (-1x-1x512xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(paddle._C_ops.dropout(dropout_46, None, constant_5, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_36 = layer_norm_30 + dropout_48

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add_36, parameter_83, parameter_84, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_14 = [slice_4, parameter_85]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(slice_4, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_15 = [memcpy_h2d_0, parameter_85]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_0 = paddle._C_ops.stack(combine_15, 0)

        # pd_op.full_with_tensor: (-1x-1xi64) <- (1xf32, 2xi32)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(full_0, stack_0, paddle.int64)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_16 = [slice_4, parameter_85]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_1 = paddle._C_ops.memcpy_h2d(slice_4, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_17 = [memcpy_h2d_1, parameter_85]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_1 = paddle._C_ops.stack(combine_17, 0)

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi32)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_1, stack_1, paddle.float32)

        # pd_op.full: (xb) <- ()
        full_2 = paddle._C_ops.full([], float('0'), paddle.bool, paddle.framework._current_expected_place())

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full([], float('1'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = full_3 < parameter_86

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(full_2)

        # pd_op.logical_and_: (xb) <- (xb, xb)
        logical_and__0 = paddle._C_ops.logical_and(less_than_0, logical_not_0)

        # pd_op.assign_value: (-1x-1x512xf32) <- ()
        assign_value_0 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (-1x-1x512xf32) <- ()
        assign_value_1 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (-1x512xf32) <- ()
        assign_value_2 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (-1x99xf32) <- ()
        assign_value_3 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (-1xi64) <- ()
        assign_value_4 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (-1xf32) <- ()
        assign_value_5 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (1x1x-1x-1xf32) <- ()
        assign_value_6 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x512xf32, -1x-1xi64, 1x1x-1x-1xf32, -1x-1x512xf32, -1xi64, xb, xi64, -1xf32, -1x-1xf32, -1x-1x512xf32, -1x99xf32) <- (xb, -1x512xf32, -1x-1xi64, 1x1x-1x-1xf32, -1x-1x512xf32, -1xi64, xb, xi64, -1xf32, -1x-1xf32, -1x-1x512xf32, -1x99xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_2053 = 0
        while logical_and__0:
            logical_and__0, assign_value_2, full_with_tensor_0, assign_value_6, assign_value_1, assign_value_4, full_2, full_3, assign_value_5, full_with_tensor_1, assign_value_0, assign_value_3, = self.pd_op_while_2053_0_0(parameter_87, constant_12, constant_0, constant_1, parameter_12, constant_5, constant_2, constant_13, parameter_88, parameter_89, constant_6, constant_7, constant_8, constant_9, constant_3, constant_10, constant_11, parameter_90, parameter_91, parameter_92, parameter_93, layer_norm_33, parameter_94, parameter_95, parameter_96, parameter_97, constant_14, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, constant_15, constant_16, parameter_208, constant_17, parameter_209, parameter_210, constant_18, constant_19, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, full_1, parameter_86, logical_and__0, assign_value_2, full_with_tensor_0, assign_value_6, assign_value_1, assign_value_4, full_2, full_3, assign_value_5, full_with_tensor_1, assign_value_0, assign_value_3)
            while_loop_counter_2053 += 1
            if while_loop_counter_2053 > kWhileLoopLimit:
                break
            
        while_0, while_1, while_2, while_3, while_4, while_5, while_6, while_7, while_8, while_9, while_10, = assign_value_2, full_with_tensor_0, assign_value_6, assign_value_1, assign_value_4, full_2, full_3, assign_value_5, full_with_tensor_1, assign_value_0, assign_value_3,

        # pd_op.scale: (-1x-1xi64) <- (-1x-1xi64, 1xf32)
        scale_6 = paddle._C_ops.scale(while_1, full_1, float('0'), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(while_8, full_1, float('0'), True)
        return scale_6, scale_7



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

    def forward(self, parameter_214, parameter_215, parameter_211, parameter_212, parameter_213, constant_19, constant_18, parameter_209, parameter_210, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, parameter_216, parameter_86, parameter_85, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_24, parameter_23, parameter_25, parameter_26, parameter_27, parameter_28, parameter_30, parameter_29, parameter_31, parameter_32, parameter_33, parameter_34, parameter_36, parameter_35, parameter_37, parameter_38, parameter_39, parameter_40, parameter_42, parameter_41, parameter_43, parameter_44, parameter_45, parameter_46, parameter_48, parameter_47, parameter_49, parameter_50, parameter_51, parameter_52, parameter_54, parameter_53, parameter_55, parameter_56, parameter_57, parameter_58, parameter_60, parameter_59, parameter_61, parameter_62, parameter_63, parameter_64, parameter_66, parameter_65, parameter_67, parameter_68, parameter_69, parameter_70, parameter_72, parameter_71, parameter_73, parameter_74, parameter_75, parameter_76, parameter_78, parameter_77, parameter_79, parameter_80, parameter_81, parameter_82, parameter_84, parameter_83, parameter_109, parameter_134, parameter_94, parameter_197, parameter_166, parameter_194, parameter_172, parameter_133, parameter_195, parameter_167, parameter_208, parameter_122, parameter_159, parameter_87, parameter_136, parameter_149, parameter_121, parameter_181, parameter_89, parameter_114, parameter_102, parameter_152, parameter_198, parameter_128, parameter_174, parameter_161, parameter_145, parameter_199, parameter_137, parameter_130, parameter_99, parameter_126, parameter_148, parameter_178, parameter_111, parameter_158, parameter_202, parameter_110, parameter_135, parameter_162, parameter_200, parameter_104, parameter_168, parameter_103, parameter_96, parameter_173, parameter_189, parameter_147, parameter_129, parameter_92, parameter_95, parameter_140, parameter_106, parameter_155, parameter_177, parameter_160, parameter_93, parameter_176, parameter_112, parameter_97, parameter_98, parameter_164, parameter_196, parameter_188, parameter_154, parameter_107, parameter_157, parameter_182, parameter_125, parameter_120, parameter_191, parameter_165, parameter_123, parameter_205, parameter_193, parameter_131, parameter_142, parameter_144, parameter_90, parameter_88, parameter_113, parameter_190, parameter_116, parameter_169, parameter_91, parameter_108, parameter_192, parameter_139, parameter_170, parameter_127, parameter_100, parameter_151, parameter_156, parameter_163, parameter_101, parameter_138, parameter_141, parameter_105, parameter_132, parameter_184, parameter_119, parameter_179, parameter_118, parameter_143, parameter_180, parameter_201, parameter_207, parameter_175, parameter_187, parameter_124, parameter_204, parameter_153, parameter_146, parameter_117, parameter_171, parameter_203, parameter_206, parameter_115, parameter_186, parameter_185, parameter_150, parameter_183, feed_0):
        return self.builtin_module_1564_0_0(parameter_214, parameter_215, parameter_211, parameter_212, parameter_213, constant_19, constant_18, parameter_209, parameter_210, constant_17, constant_16, constant_15, constant_14, constant_13, constant_12, parameter_216, parameter_86, parameter_85, constant_11, constant_10, constant_9, constant_8, constant_7, constant_6, constant_5, constant_4, constant_3, constant_2, constant_1, constant_0, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_24, parameter_23, parameter_25, parameter_26, parameter_27, parameter_28, parameter_30, parameter_29, parameter_31, parameter_32, parameter_33, parameter_34, parameter_36, parameter_35, parameter_37, parameter_38, parameter_39, parameter_40, parameter_42, parameter_41, parameter_43, parameter_44, parameter_45, parameter_46, parameter_48, parameter_47, parameter_49, parameter_50, parameter_51, parameter_52, parameter_54, parameter_53, parameter_55, parameter_56, parameter_57, parameter_58, parameter_60, parameter_59, parameter_61, parameter_62, parameter_63, parameter_64, parameter_66, parameter_65, parameter_67, parameter_68, parameter_69, parameter_70, parameter_72, parameter_71, parameter_73, parameter_74, parameter_75, parameter_76, parameter_78, parameter_77, parameter_79, parameter_80, parameter_81, parameter_82, parameter_84, parameter_83, parameter_109, parameter_134, parameter_94, parameter_197, parameter_166, parameter_194, parameter_172, parameter_133, parameter_195, parameter_167, parameter_208, parameter_122, parameter_159, parameter_87, parameter_136, parameter_149, parameter_121, parameter_181, parameter_89, parameter_114, parameter_102, parameter_152, parameter_198, parameter_128, parameter_174, parameter_161, parameter_145, parameter_199, parameter_137, parameter_130, parameter_99, parameter_126, parameter_148, parameter_178, parameter_111, parameter_158, parameter_202, parameter_110, parameter_135, parameter_162, parameter_200, parameter_104, parameter_168, parameter_103, parameter_96, parameter_173, parameter_189, parameter_147, parameter_129, parameter_92, parameter_95, parameter_140, parameter_106, parameter_155, parameter_177, parameter_160, parameter_93, parameter_176, parameter_112, parameter_97, parameter_98, parameter_164, parameter_196, parameter_188, parameter_154, parameter_107, parameter_157, parameter_182, parameter_125, parameter_120, parameter_191, parameter_165, parameter_123, parameter_205, parameter_193, parameter_131, parameter_142, parameter_144, parameter_90, parameter_88, parameter_113, parameter_190, parameter_116, parameter_169, parameter_91, parameter_108, parameter_192, parameter_139, parameter_170, parameter_127, parameter_100, parameter_151, parameter_156, parameter_163, parameter_101, parameter_138, parameter_141, parameter_105, parameter_132, parameter_184, parameter_119, parameter_179, parameter_118, parameter_143, parameter_180, parameter_201, parameter_207, parameter_175, parameter_187, parameter_124, parameter_204, parameter_153, parameter_146, parameter_117, parameter_171, parameter_203, parameter_206, parameter_115, parameter_186, parameter_185, parameter_150, parameter_183, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1564_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_214
            paddle.to_tensor([0], dtype='int64').reshape([1, 1]),
            # parameter_215
            paddle.uniform([1, 1], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([1, 2], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.to_tensor([0, 0], dtype='int64').reshape([1, 2]),
            # constant_19
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_18
            paddle.to_tensor([-1, 1], dtype='int64').reshape([2]),
            # parameter_209
            paddle.cast(paddle.randint(low=0, high=2, shape=[], dtype='int32'), 'bool'),
            # parameter_210
            paddle.cast(paddle.randint(low=0, high=2, shape=[], dtype='int32'), 'bool'),
            # constant_17
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            # constant_16
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            # constant_15
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            # constant_14
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            # constant_13
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            # constant_12
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.to_tensor([25], dtype='int64').reshape([]),
            # parameter_85
            paddle.to_tensor([1], dtype='int32').reshape([]),
            # constant_11
            paddle.to_tensor([512], dtype='int32').reshape([1]),
            # constant_10
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_9
            paddle.to_tensor([64], dtype='int32').reshape([1]),
            # constant_8
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            # constant_7
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            # constant_6
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            # constant_5
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_4
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            # constant_3
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            # constant_2
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_7
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([5000, 1, 512], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([512, 99], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([99, 512], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512, 1536], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_214
            paddle.static.InputSpec(shape=[1, 1], dtype='int64'),
            # parameter_215
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[1, 2], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[1, 2], dtype='int64'),
            # constant_19
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_18
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_209
            paddle.static.InputSpec(shape=[], dtype='bool'),
            # parameter_210
            paddle.static.InputSpec(shape=[], dtype='bool'),
            # constant_17
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_15
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_13
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[], dtype='int64'),
            # parameter_85
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_7
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[5000, 1, 512], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[512, 99], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[99, 512], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512, 1536], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
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