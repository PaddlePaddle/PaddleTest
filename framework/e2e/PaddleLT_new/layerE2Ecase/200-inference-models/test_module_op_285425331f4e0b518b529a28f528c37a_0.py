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
    return [5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 5, 0, 1, 0, 22, 2668][block_idx] - 1 # number-of-ops-in-block

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
    def pd_op_if_9452_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_9452_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_9461_0_0(self, reshape__0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = reshape__0
        return assign_0
    def pd_op_if_9461_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_9442_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_2, assign_value_1, full_0, reshape__0, assign_value_2, assign_value_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9452_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_9452_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9461_0_0(reshape__0)
        else:
            if_2, = self.pd_op_if_9461_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_1, assign_out__2, assign_out__1, assign_out__4, assign_out__3, assign_out__0
    def pd_op_if_9532_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_9532_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_9541_0_0(self, reshape__0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = reshape__0
        return assign_0
    def pd_op_if_9541_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_9522_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_2, assign_value_1, assign_value_2, assign_value_0, full_0, reshape__0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9532_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_9532_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9541_0_0(reshape__0)
        else:
            if_2, = self.pd_op_if_9541_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_1, assign_out__2, assign_out__3, assign_out__0, assign_out__1, assign_out__4
    def pd_op_if_9627_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_9627_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_9636_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9636_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_9617_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_2, full_2, assign_value_0, reshape__0, full_0, assign_value_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9627_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_9627_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9636_0_0(if_0)
        else:
            if_2, = self.pd_op_if_9636_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, if_0, assign_out__0, assign_out__4, assign_out__1, assign_out__2
    def pd_op_if_9722_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_9722_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_9731_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9731_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_9712_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_2, assign_value_1, assign_value_0, reshape__0, full_2, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9722_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_9722_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9731_0_0(if_1)
        else:
            if_2, = self.pd_op_if_9731_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__2, assign_out__0, assign_out__4, if_1, assign_out__1
    def pd_op_if_9817_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_9817_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_9826_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9826_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_9807_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, reshape__0, full_0, assign_value_2, full_2, assign_value_1, assign_value_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9817_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_9817_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9826_0_0(if_1)
        else:
            if_2, = self.pd_op_if_9826_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__1, assign_out__3, if_1, assign_out__2, assign_out__0
    def pd_op_if_9912_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_9912_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_9921_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9921_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_9902_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, reshape__0, assign_value_2, full_0, full_2, assign_value_1, assign_value_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9912_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_9912_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9921_0_0(if_0)
        else:
            if_2, = self.pd_op_if_9921_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__3, assign_out__1, if_0, assign_out__2, assign_out__0
    def pd_op_if_10007_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_10007_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_10016_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10016_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_9997_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_0, assign_value_1, reshape__0, assign_value_0, full_2, assign_value_2):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10007_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10007_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10016_0_0(if_0)
        else:
            if_2, = self.pd_op_if_10016_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__4, assign_out__0, if_0, assign_out__3
    def pd_op_if_10102_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_10102_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_10111_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10111_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10092_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_2, full_0, reshape__0, assign_value_1, assign_value_2, assign_value_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10102_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10102_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10111_0_0(if_0)
        else:
            if_2, = self.pd_op_if_10111_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__1, assign_out__4, assign_out__2, assign_out__3, assign_out__0
    def pd_op_if_10197_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_10197_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_10206_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10206_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10187_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, reshape__0, assign_value_0, assign_value_2, full_2, assign_value_1, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10197_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10197_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10206_0_0(if_0)
        else:
            if_2, = self.pd_op_if_10206_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__0, assign_out__3, if_0, assign_out__2, assign_out__1
    def pd_op_if_10292_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_10292_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_10301_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10301_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10282_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_1, full_0, reshape__0, full_2, assign_value_2, assign_value_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10292_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10292_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10301_0_0(if_1)
        else:
            if_2, = self.pd_op_if_10301_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__1, assign_out__4, if_1, assign_out__3, assign_out__0
    def pd_op_if_10387_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_10387_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_10396_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10396_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10377_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_2, full_2, assign_value_0, reshape__0, full_0, assign_value_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10387_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10387_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10396_0_0(if_1)
        else:
            if_2, = self.pd_op_if_10396_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, if_1, assign_out__0, assign_out__4, assign_out__1, assign_out__2
    def pd_op_if_10482_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_10482_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_10491_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10491_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10472_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_1, assign_value_0, assign_value_2, full_2, full_0, reshape__0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10482_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10482_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10491_0_0(if_1)
        else:
            if_2, = self.pd_op_if_10491_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__0, assign_out__3, if_1, assign_out__1, assign_out__4
    def pd_op_if_10577_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_10577_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_10586_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10586_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10567_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_2, reshape__0, assign_value_1, assign_value_0, full_0, assign_value_2):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10577_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10577_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10586_0_0(if_0)
        else:
            if_2, = self.pd_op_if_10586_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__4, assign_out__2, assign_out__0, assign_out__1, assign_out__3
    def pd_op_if_10672_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_10672_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_10681_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10681_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10662_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_0, assign_value_1, reshape__0, full_2, assign_value_0, assign_value_2):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10672_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10672_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10681_0_0(if_1)
        else:
            if_2, = self.pd_op_if_10681_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__4, if_1, assign_out__0, assign_out__3
    def pd_op_if_10767_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_10767_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_10776_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10776_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10757_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_1, full_2, reshape__0, assign_value_0, full_0, assign_value_2):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10767_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10767_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10776_0_0(if_0)
        else:
            if_2, = self.pd_op_if_10776_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, if_0, assign_out__4, assign_out__0, assign_out__1, assign_out__3
    def pd_op_if_10862_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_10862_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_10871_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10871_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10852_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_2, full_2, assign_value_0, assign_value_1, reshape__0, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10862_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10862_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10871_0_0(if_1)
        else:
            if_2, = self.pd_op_if_10871_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, if_1, assign_out__0, assign_out__2, assign_out__4, assign_out__1
    def pd_op_if_10957_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_10957_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_10966_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10966_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_10947_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_0, full_0, full_2, assign_value_2, assign_value_1, reshape__0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10957_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_10957_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10966_0_0(if_0)
        else:
            if_2, = self.pd_op_if_10966_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__0, assign_out__1, if_0, assign_out__3, assign_out__2, assign_out__4
    def pd_op_if_11052_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_11052_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_11061_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11061_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11042_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_2, full_2, assign_value_0, assign_value_1, reshape__0, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11052_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11052_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11061_0_0(if_1)
        else:
            if_2, = self.pd_op_if_11061_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, if_1, assign_out__0, assign_out__2, assign_out__4, assign_out__1
    def pd_op_if_11147_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_11147_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_11156_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11156_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11137_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_1, full_0, full_2, reshape__0, assign_value_2, assign_value_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11147_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11147_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11156_0_0(if_0)
        else:
            if_2, = self.pd_op_if_11156_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__1, if_0, assign_out__4, assign_out__3, assign_out__0
    def pd_op_if_11242_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_11242_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_11251_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11251_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11232_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, reshape__0, full_2, full_0, assign_value_2, assign_value_0, assign_value_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11242_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11242_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11251_0_0(if_0)
        else:
            if_2, = self.pd_op_if_11251_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, if_0, assign_out__1, assign_out__3, assign_out__0, assign_out__2
    def pd_op_if_11337_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_11337_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_11346_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11346_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11327_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_1, assign_value_2, assign_value_0, reshape__0, full_0, full_2):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11337_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11337_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11346_0_0(if_0)
        else:
            if_2, = self.pd_op_if_11346_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__3, assign_out__0, assign_out__4, assign_out__1, if_0
    def pd_op_if_11432_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_11432_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_11441_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11441_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11422_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_0, assign_value_1, full_2, reshape__0, assign_value_2, assign_value_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11432_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11432_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11441_0_0(if_1)
        else:
            if_2, = self.pd_op_if_11441_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, if_1, assign_out__4, assign_out__3, assign_out__0
    def pd_op_if_11527_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_11527_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_11536_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11536_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11517_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_2, assign_value_1, reshape__0, full_2, assign_value_0, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11527_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11527_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11536_0_0(if_0)
        else:
            if_2, = self.pd_op_if_11536_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__2, assign_out__4, if_0, assign_out__0, assign_out__1
    def pd_op_if_11622_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_11622_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_11631_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11631_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11612_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_0, assign_value_1, reshape__0, assign_value_0, assign_value_2, full_2):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11622_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11622_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11631_0_0(if_1)
        else:
            if_2, = self.pd_op_if_11631_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__4, assign_out__0, assign_out__3, if_1
    def pd_op_if_11717_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_11717_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_11726_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11726_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11707_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_2, assign_value_0, reshape__0, assign_value_1, assign_value_2, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11717_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11717_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11726_0_0(if_1)
        else:
            if_2, = self.pd_op_if_11726_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_1, assign_out__0, assign_out__4, assign_out__2, assign_out__3, assign_out__1
    def pd_op_if_11812_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_11812_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_11821_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11821_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11802_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_2, assign_value_2, reshape__0, assign_value_0, assign_value_1, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11812_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11812_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11821_0_0(if_0)
        else:
            if_2, = self.pd_op_if_11821_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__3, assign_out__4, assign_out__0, assign_out__2, assign_out__1
    def pd_op_if_11907_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_11907_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_11916_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11916_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11897_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_0, assign_value_1, assign_value_0, reshape__0, full_2, assign_value_2):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11907_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_11907_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11916_0_0(if_1)
        else:
            if_2, = self.pd_op_if_11916_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__0, assign_out__4, if_1, assign_out__3
    def pd_op_if_12002_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_12002_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_12011_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12011_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_11992_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_2, full_0, reshape__0, assign_value_2, assign_value_0, assign_value_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12002_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12002_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12011_0_0(if_0)
        else:
            if_2, = self.pd_op_if_12011_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__1, assign_out__4, assign_out__3, assign_out__0, assign_out__2
    def pd_op_if_12097_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_12097_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_12106_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12106_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12087_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_2, reshape__0, assign_value_0, assign_value_1, full_0, full_2):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12097_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12097_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12106_0_0(if_0)
        else:
            if_2, = self.pd_op_if_12106_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__4, assign_out__0, assign_out__2, assign_out__1, if_0
    def pd_op_if_12192_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12192_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_12201_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12201_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12182_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_0, assign_value_2, full_2, assign_value_1, reshape__0, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12192_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12192_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12201_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12201_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__0, assign_out__3, if_1, assign_out__2, assign_out__4, assign_out__1
    def pd_op_if_12287_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_12287_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_12296_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12296_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12277_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_1, full_2, assign_value_2, full_0, assign_value_0, reshape__0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12287_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12287_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12296_0_0(if_0)
        else:
            if_2, = self.pd_op_if_12296_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, if_0, assign_out__3, assign_out__1, assign_out__0, assign_out__4
    def pd_op_if_12382_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12382_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_12391_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12391_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12372_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_1, full_2, assign_value_2, reshape__0, assign_value_0, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12382_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12382_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12391_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12391_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, if_1, assign_out__3, assign_out__4, assign_out__0, assign_out__1
    def pd_op_if_12477_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_12477_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_12486_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12486_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12467_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, reshape__0, full_0, assign_value_0, assign_value_2, full_2, assign_value_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12477_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12477_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12486_0_0(if_0)
        else:
            if_2, = self.pd_op_if_12486_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__1, assign_out__0, assign_out__3, if_0, assign_out__2
    def pd_op_if_12572_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12572_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_12581_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12581_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12562_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_2, full_0, reshape__0, full_2, assign_value_0, assign_value_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12572_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12572_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12581_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12581_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__1, assign_out__4, if_1, assign_out__0, assign_out__2
    def pd_op_if_12667_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12667_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_12676_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12676_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12657_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_2, assign_value_0, reshape__0, full_2, full_0, assign_value_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12667_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12667_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12676_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12676_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__0, assign_out__4, if_1, assign_out__1, assign_out__2
    def pd_op_if_12762_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_12762_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_12771_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12771_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12752_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_2, assign_value_0, assign_value_2, reshape__0, assign_value_1, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12762_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12762_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12771_0_0(if_0)
        else:
            if_2, = self.pd_op_if_12771_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__0, assign_out__3, assign_out__4, assign_out__2, assign_out__1
    def pd_op_if_12857_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12857_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_12866_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12866_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12847_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, reshape__0, assign_value_0, full_0, assign_value_2, full_2, assign_value_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12857_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12857_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12866_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12866_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__0, assign_out__1, assign_out__3, if_1, assign_out__2
    def pd_op_if_12952_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12952_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_12961_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12961_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_12942_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_1, full_0, full_2, assign_value_0, assign_value_2, reshape__0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12952_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_12952_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12961_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12961_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__1, if_1, assign_out__0, assign_out__3, assign_out__4
    def pd_op_if_13047_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_13047_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_13056_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_13056_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_13037_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_0, reshape__0, assign_value_2, full_2, assign_value_1, assign_value_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_13047_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_13047_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_13056_0_0(if_1)
        else:
            if_2, = self.pd_op_if_13056_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__4, assign_out__3, if_1, assign_out__2, assign_out__0
    def pd_op_if_13142_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_13142_1_0(self, parameter_0, full_0):
        return parameter_0, full_0
    def pd_op_if_13151_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_13151_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_13132_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, assign_value_1, assign_value_2, assign_value_0, reshape__0, full_2, full_0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_13142_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_13142_1_0(parameter_1, full_2)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_13151_0_0(if_1)
        else:
            if_2, = self.pd_op_if_13151_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_0][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__3, assign_out__0, assign_out__4, if_1, assign_out__1
    def pd_op_if_13237_0_0(self, slice_0, full_0, cast_0, constant_0, reshape__0, constant_1):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, constant_0]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, combine_0, combine_1, constant_1, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_13237_1_0(self, full_0, parameter_0):
        return full_0, parameter_0
    def pd_op_if_13246_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_13246_1_0(self, parameter_0):
        return parameter_0
    def pd_op_while_13227_0_0(self, full_1, arange_0, feed_0, constant_0, parameter_0, constant_1, constant_2, parameter_1, parameter_2, cast_2, less_than_2, full_0, assign_value_1, assign_value_2, full_2, assign_value_0, reshape__0):

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_0, full_1, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], combine_0, combine_1, [-1], [0])

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, constant_0, float('0'), True)

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale(scale_1, full_1, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = cast_0 < parameter_0

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_13237_0_0(slice_0, full_1, cast_0, constant_1, reshape__0, constant_2)
        else:
            if_0, if_1, = self.pd_op_if_13237_1_0(full_2, parameter_1)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_13246_0_0(if_0)
        else:
            if_2, = self.pd_op_if_13246_1_0(parameter_2)

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = [if_2, if_1][int(cast_1)]

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = scale_0 < memcpy_h2d_0

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_0, full_0)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__3, if_0, assign_out__0, assign_out__4
    def builtin_module_8767_0_0(self, parameter_310, parameter_309, constant_177, constant_176, parameter_308, constant_175, parameter_307, parameter_306, constant_173, constant_172, parameter_305, constant_171, parameter_304, parameter_303, constant_169, constant_168, parameter_302, constant_167, parameter_301, parameter_300, constant_165, constant_164, parameter_299, constant_163, parameter_298, parameter_297, constant_161, constant_160, parameter_296, constant_159, parameter_295, parameter_294, constant_157, constant_156, parameter_293, constant_155, parameter_292, parameter_291, constant_153, constant_152, parameter_290, constant_151, parameter_289, parameter_288, constant_149, constant_148, parameter_287, constant_147, parameter_286, parameter_285, constant_145, constant_144, parameter_284, constant_143, parameter_283, parameter_282, constant_141, constant_140, parameter_281, constant_139, parameter_280, parameter_279, constant_137, constant_136, parameter_278, constant_135, parameter_277, parameter_276, constant_133, constant_132, parameter_275, constant_131, parameter_274, parameter_273, constant_129, constant_128, parameter_272, constant_127, parameter_271, parameter_270, constant_125, constant_124, parameter_269, constant_123, parameter_268, parameter_267, constant_121, constant_120, parameter_266, constant_119, parameter_265, parameter_264, constant_117, constant_116, parameter_263, constant_115, parameter_262, parameter_261, constant_113, constant_112, parameter_260, constant_111, parameter_259, parameter_258, constant_109, constant_108, parameter_257, constant_107, parameter_256, parameter_255, constant_105, constant_104, parameter_254, constant_103, parameter_253, parameter_252, constant_101, constant_100, parameter_251, constant_99, parameter_250, parameter_249, constant_97, constant_96, parameter_248, constant_95, parameter_247, parameter_246, constant_93, constant_92, parameter_245, constant_91, parameter_244, parameter_243, constant_89, constant_88, parameter_242, constant_87, parameter_241, parameter_240, constant_85, constant_84, parameter_239, constant_83, parameter_238, parameter_237, constant_81, constant_80, parameter_236, constant_79, parameter_235, parameter_234, constant_77, constant_76, parameter_233, constant_75, parameter_232, parameter_231, constant_73, constant_72, parameter_230, constant_71, parameter_229, parameter_228, constant_69, constant_68, parameter_227, constant_67, parameter_226, parameter_225, constant_65, constant_64, parameter_224, constant_63, parameter_223, parameter_222, constant_61, constant_60, parameter_221, constant_59, parameter_220, parameter_219, constant_57, constant_56, parameter_218, constant_55, parameter_217, parameter_216, constant_53, constant_52, parameter_215, constant_51, parameter_214, parameter_213, constant_49, constant_48, parameter_212, constant_47, parameter_211, parameter_210, constant_45, constant_44, parameter_209, constant_43, parameter_208, parameter_207, constant_41, constant_40, parameter_206, constant_39, parameter_205, parameter_204, constant_37, constant_36, parameter_203, constant_35, parameter_202, parameter_201, constant_33, constant_32, parameter_200, constant_31, parameter_199, parameter_198, constant_29, constant_28, parameter_197, constant_27, parameter_196, parameter_195, constant_25, constant_24, parameter_194, constant_23, parameter_189, parameter_188, constant_18, constant_17, parameter_187, constant_16, parameter_177, parameter_176, constant_15, constant_14, parameter_175, constant_13, constant_174, constant_170, constant_166, constant_162, constant_158, constant_154, constant_150, constant_146, constant_142, constant_138, constant_134, constant_130, constant_126, constant_122, constant_118, constant_114, constant_110, constant_106, constant_102, constant_98, constant_94, constant_90, constant_86, constant_82, constant_78, constant_74, constant_70, constant_66, constant_62, constant_58, constant_54, constant_50, constant_46, constant_42, constant_38, constant_34, constant_30, constant_26, constant_22, constant_21, constant_20, constant_19, constant_12, constant_11, constant_10, constant_9, parameter_173, parameter_171, constant_8, parameter_169, parameter_160, parameter_159, constant_7, constant_6, constant_5, constant_4, parameter_158, constant_3, constant_2, parameter_157, parameter_151, parameter_115, constant_1, parameter_54, parameter_28, constant_0, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_32, parameter_29, parameter_31, parameter_30, parameter_33, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_170, parameter_172, parameter_174, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_190, parameter_191, parameter_192, parameter_193, feed_1, feed_2, feed_0):

        # pd_op.conv2d: (-1x64x48x160xf32) <- (-1x3x48x160xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x48x160xf32) <- (-1x64x48x160xf32, 1x64x1x1xf32)
        add__0 = paddle._C_ops.add(conv2d_0, parameter_1)

        # pd_op.batch_norm_: (-1x64x48x160xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x48x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x48x160xf32) <- (-1x64x48x160xf32)
        relu__0 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.conv2d: (-1x128x48x160xf32) <- (-1x64x48x160xf32, 128x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x48x160xf32) <- (-1x128x48x160xf32, 1x128x1x1xf32)
        add__1 = paddle._C_ops.add(conv2d_1, parameter_7)

        # pd_op.batch_norm_: (-1x128x48x160xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x48x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__1, parameter_8, parameter_9, parameter_10, parameter_11, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x48x160xf32) <- (-1x128x48x160xf32)
        relu__1 = paddle._C_ops.relu(batch_norm__6)

        # pd_op.pool2d: (-1x128x24x80xf32) <- (-1x128x48x160xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__1, constant_0, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x24x80xf32) <- (-1x128x24x80xf32, 256x128x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_0, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x24x80xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x24x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_13, parameter_14, parameter_15, parameter_16, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x24x80xf32) <- (-1x256x24x80xf32)
        relu__2 = paddle._C_ops.relu(batch_norm__12)

        # pd_op.conv2d: (-1x256x24x80xf32) <- (-1x256x24x80xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu__2, parameter_17, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x24x80xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x24x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_18, parameter_19, parameter_20, parameter_21, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x24x80xf32) <- (-1x128x24x80xf32, 256x128x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_0, parameter_22, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x24x80xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x24x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_23, parameter_24, parameter_25, parameter_26, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x24x80xf32) <- (-1x256x24x80xf32, -1x256x24x80xf32)
        add__2 = paddle._C_ops.add(batch_norm__18, batch_norm__24)

        # pd_op.relu_: (-1x256x24x80xf32) <- (-1x256x24x80xf32)
        relu__3 = paddle._C_ops.relu(add__2)

        # pd_op.conv2d: (-1x256x24x80xf32) <- (-1x256x24x80xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu__3, parameter_27, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x24x80xf32) <- (-1x256x24x80xf32, 1x256x1x1xf32)
        add__3 = paddle._C_ops.add(conv2d_5, parameter_28)

        # pd_op.batch_norm_: (-1x256x24x80xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x24x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__3, parameter_29, parameter_30, parameter_31, parameter_32, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x24x80xf32) <- (-1x256x24x80xf32)
        relu__4 = paddle._C_ops.relu(batch_norm__30)

        # pd_op.pool2d: (-1x256x12x40xf32) <- (-1x256x24x80xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__4, constant_0, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(pool2d_1, parameter_33, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_34, parameter_35, parameter_36, parameter_37, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__5 = paddle._C_ops.relu(batch_norm__36)

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu__5, parameter_38, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_39, parameter_40, parameter_41, parameter_42, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x12x40xf32) <- (-1x256x12x40xf32, -1x256x12x40xf32)
        add__4 = paddle._C_ops.add(batch_norm__42, pool2d_1)

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__6 = paddle._C_ops.relu(add__4)

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu__6, parameter_43, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_44, parameter_45, parameter_46, parameter_47, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__7 = paddle._C_ops.relu(batch_norm__48)

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__7, parameter_48, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_49, parameter_50, parameter_51, parameter_52, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x12x40xf32) <- (-1x256x12x40xf32, -1x256x12x40xf32)
        add__5 = paddle._C_ops.add(batch_norm__54, relu__6)

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__8 = paddle._C_ops.relu(add__5)

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu__8, parameter_53, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 1x256x1x1xf32)
        add__6 = paddle._C_ops.add(conv2d_10, parameter_54)

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__6, parameter_55, parameter_56, parameter_57, parameter_58, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__9 = paddle._C_ops.relu(batch_norm__60)

        # pd_op.pool2d: (-1x256x6x40xf32) <- (-1x256x12x40xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu__9, constant_1, [2, 1], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x256x6x40xf32, 512x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(pool2d_2, parameter_59, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__10 = paddle._C_ops.relu(batch_norm__66)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu__10, parameter_64, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_65, parameter_66, parameter_67, parameter_68, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x256x6x40xf32, 512x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(pool2d_2, parameter_69, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__7 = paddle._C_ops.add(batch_norm__72, batch_norm__78)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__11 = paddle._C_ops.relu(add__7)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu__11, parameter_74, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_75, parameter_76, parameter_77, parameter_78, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__12 = paddle._C_ops.relu(batch_norm__84)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__12, parameter_79, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_80, parameter_81, parameter_82, parameter_83, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__8 = paddle._C_ops.add(batch_norm__90, relu__11)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__13 = paddle._C_ops.relu(add__8)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu__13, parameter_84, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_85, parameter_86, parameter_87, parameter_88, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__14 = paddle._C_ops.relu(batch_norm__96)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu__14, parameter_89, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_90, parameter_91, parameter_92, parameter_93, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__9 = paddle._C_ops.add(batch_norm__102, relu__13)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__15 = paddle._C_ops.relu(add__9)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu__15, parameter_94, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_95, parameter_96, parameter_97, parameter_98, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__16 = paddle._C_ops.relu(batch_norm__108)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu__16, parameter_99, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_100, parameter_101, parameter_102, parameter_103, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__10 = paddle._C_ops.add(batch_norm__114, relu__15)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__17 = paddle._C_ops.relu(add__10)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu__17, parameter_104, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_105, parameter_106, parameter_107, parameter_108, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__18 = paddle._C_ops.relu(batch_norm__120)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu__18, parameter_109, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__11 = paddle._C_ops.add(batch_norm__126, relu__17)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__19 = paddle._C_ops.relu(add__11)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu__19, parameter_114, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 1x512x1x1xf32)
        add__12 = paddle._C_ops.add(conv2d_22, parameter_115)

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__12, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__20 = paddle._C_ops.relu(batch_norm__132)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu__20, parameter_120, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__21 = paddle._C_ops.relu(batch_norm__138)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__21, parameter_125, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__13 = paddle._C_ops.add(batch_norm__144, relu__20)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__22 = paddle._C_ops.relu(add__13)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu__22, parameter_130, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__23 = paddle._C_ops.relu(batch_norm__150)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu__23, parameter_135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__14 = paddle._C_ops.add(batch_norm__156, relu__22)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__24 = paddle._C_ops.relu(add__14)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu__24, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__25 = paddle._C_ops.relu(batch_norm__162)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__25, parameter_145, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__15 = paddle._C_ops.add(batch_norm__168, relu__24)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__26 = paddle._C_ops.relu(add__15)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu__26, parameter_150, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 1x512x1x1xf32)
        add__16 = paddle._C_ops.add(conv2d_29, parameter_151)

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__16, parameter_152, parameter_153, parameter_154, parameter_155, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__27 = paddle._C_ops.relu(batch_norm__174)

        # pd_op.conv2d: (-1x128x6x40xf32) <- (-1x512x6x40xf32, 128x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__27, parameter_156, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 1x128x1x1xf32)
        add__17 = paddle._C_ops.add(conv2d_30, parameter_157)

        # pd_op.shape: (4xi32) <- (-1x512x6x40xf32)
        shape_0 = paddle._C_ops.shape(relu__27)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], constant_2, constant_3, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_0 = [slice_0, parameter_158]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(slice_0, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_1 = [memcpy_h2d_0, parameter_158]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_0 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.full_with_tensor: (-1x40xi64) <- (1xf32, 2xi32)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(full_0, stack_0, paddle.int64)

        # pd_op.scale_: (-1x40xi64) <- (-1x40xi64, 1xf32)
        scale__0 = paddle._C_ops.scale(full_with_tensor_0, constant_4, float('0'), True)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_1 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_2 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], constant_2, constant_3, [1], [0])

        # pd_op.transpose: (-1x6x40x128xf32) <- (-1x128x6x40xf32)
        transpose_0 = paddle._C_ops.transpose(add__17, [0, 2, 3, 1])

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_2, constant_5, float('0'), True)

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_2 = [scale_0, constant_6, constant_7]

        # pd_op.reshape_: (-1x40x128xf32, 0x-1x6x40x128xf32) <- (-1x6x40x128xf32, [xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_0, combine_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_3 = paddle._C_ops.shape(reshape__0)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_3, [0], constant_2, constant_3, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_3 = [parameter_159, slice_3, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_1 = paddle._C_ops.memcpy_h2d(slice_3, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_4 = [parameter_159, memcpy_h2d_1, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_1 = paddle._C_ops.stack(combine_4, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_1, stack_1, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_5 = [parameter_159, slice_3, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_2 = paddle._C_ops.memcpy_h2d(slice_3, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_6 = [parameter_159, memcpy_h2d_2, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_2 = paddle._C_ops.stack(combine_6, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(full_1, stack_2, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_7 = [full_with_tensor_1, full_with_tensor_2]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_8 = [parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__0, rnn__1, rnn__2, rnn__3 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_1, combine_7, combine_8, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_2 = paddle._C_ops.transpose(rnn__0, [1, 0, 2])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_2, constant_8, constant_6, constant_7]

        # pd_op.reshape_: (-1x6x40x128xf32, 0x-1x40x128xf32) <- (-1x40x128xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_2, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x128x6x40xf32) <- (-1x6x40x128xf32)
        transpose_3 = paddle._C_ops.transpose(reshape__2, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 128x128x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(transpose_3, parameter_170, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 1x128x1x1xf32)
        add__18 = paddle._C_ops.add(conv2d_31, parameter_171)

        # pd_op.relu_: (-1x128x6x40xf32) <- (-1x128x6x40xf32)
        relu__28 = paddle._C_ops.relu(add__18)

        # pd_op.conv2d: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 128x128x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu__28, parameter_172, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 1x128x1x1xf32)
        add__19 = paddle._C_ops.add(conv2d_32, parameter_173)

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 41x128xf32)
        embedding_0 = paddle._C_ops.embedding(feed_1, parameter_174, -1, False)

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_4 = paddle._C_ops.transpose(embedding_0, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_10 = [slice_1, constant_7, constant_9]

        # pd_op.reshape_: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__19, combine_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_11 = [slice_1, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_5 = paddle._C_ops.transpose(transpose_4, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_0 = paddle.matmul(transpose_5, reshape__4, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__1 = paddle._C_ops.scale(matmul_0, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_4 = paddle._C_ops.shape(scale__1)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_4, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_4, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__1, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_5 = paddle._C_ops.shape(feed_2)

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_5, [0], constant_2, constant_3, [1], [0])

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_5, full_0, float('0'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(scale_1, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(constant_11, cast_0, constant_12, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(slice_5, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_3 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = full_2 < memcpy_h2d_3

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_3 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_0 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_1 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_2 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi32, xi64, -1x40x6x40xf32, xi64, xf32) <- (xb, -1x40x6x40xf32, xi32, xi64, -1x40x6x40xf32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9442 = 0
        while less_than_0:
            less_than_0, full_3, assign_value_1, full_2, reshape__6, assign_value_2, assign_value_0, = self.pd_op_while_9442_0_0(full_0, arange_0, feed_2, constant_13, parameter_175, constant_14, constant_15, parameter_176, parameter_177, cast_1, less_than_0, full_3, assign_value_1, full_2, reshape__6, assign_value_2, assign_value_0)
            while_loop_counter_9442 += 1
            if while_loop_counter_9442 > kWhileLoopLimit:
                break
            
        while_0, while_1, while_2, while_3, while_4, while_5, = full_3, assign_value_1, full_2, reshape__6, assign_value_2, assign_value_0,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_13 = [slice_4, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_3, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__0 = paddle._C_ops.softmax(reshape__8, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_0, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_1 = paddle.matmul(softmax__0, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_1, [0, 2, 1])

        # pd_op.transpose: (-1x40x512xf32) <- (-1x512x40xf32)
        transpose_8 = paddle._C_ops.transpose(transpose_7, [0, 2, 1])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_1 = paddle._C_ops.embedding(scale__0, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_6 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_6, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_7 = paddle._C_ops.shape(embedding_1)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_7, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_14 = [parameter_159, slice_7, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_4 = paddle._C_ops.memcpy_h2d(slice_7, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_15 = [parameter_159, memcpy_h2d_4, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_3 = paddle._C_ops.stack(combine_15, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_3 = paddle._C_ops.full_with_tensor(full_1, stack_3, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_16 = [parameter_159, slice_7, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_5 = paddle._C_ops.memcpy_h2d(slice_7, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_17 = [parameter_159, memcpy_h2d_5, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_4 = paddle._C_ops.stack(combine_17, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_4 = paddle._C_ops.full_with_tensor(full_1, stack_4, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_9 = paddle._C_ops.transpose(embedding_1, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_18 = [full_with_tensor_3, full_with_tensor_4]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_19 = [parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__4, rnn__5, rnn__6, rnn__7 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_9, combine_18, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_10 = paddle._C_ops.transpose(rnn__4, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_11 = paddle._C_ops.transpose(transpose_10, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_20 = [slice_6, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_21 = [slice_6, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_12 = paddle._C_ops.transpose(transpose_11, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_2 = paddle.matmul(transpose_12, reshape_2, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__2 = paddle._C_ops.scale(matmul_2, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_8 = paddle._C_ops.shape(scale__2)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_8, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_22 = [slice_8, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__2, combine_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_6 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = full_4 < memcpy_h2d_6

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_5 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_3 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_4 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_5 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi32, xi64, xf32, xi64, -1x40x6x40xf32) <- (xb, -1x40x6x40xf32, xi32, xi64, xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9522 = 0
        while less_than_1:
            less_than_1, full_5, assign_value_4, assign_value_5, assign_value_3, full_4, reshape__10, = self.pd_op_while_9522_0_0(full_0, arange_0, feed_2, constant_16, parameter_187, constant_17, constant_18, parameter_188, parameter_189, cast_1, less_than_1, full_5, assign_value_4, assign_value_5, assign_value_3, full_4, reshape__10)
            while_loop_counter_9522 += 1
            if while_loop_counter_9522 > kWhileLoopLimit:
                break
            
        while_6, while_7, while_8, while_9, while_10, while_11, = full_5, assign_value_4, assign_value_5, assign_value_3, full_4, reshape__10,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_23 = [slice_8, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_11, combine_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__1 = paddle._C_ops.softmax(reshape__12, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_4, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_3 = paddle.matmul(softmax__1, transpose_13, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_3, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_14, [2], constant_2, constant_3, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_8, [1], constant_2, constant_3, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_24 = [slice_9, slice_10]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_24, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_4 = paddle.matmul(concat_0, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__20 = paddle._C_ops.add(matmul_4, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add__20, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_11 = split_with_num_0[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__0 = paddle._C_ops.sigmoid(slice_11)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_12 = split_with_num_0[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__0 = paddle._C_ops.multiply(slice_12, sigmoid__0)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_5 = paddle.matmul(multiply__0, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__21 = paddle._C_ops.add(matmul_5, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__2 = paddle._C_ops.softmax(add__21, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(softmax__2, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_2 = paddle._C_ops.cast(argmax_0, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor(scale__0, cast_2, constant_3, constant_22, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_2 = paddle._C_ops.embedding(set_value_with_tensor__0, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_9 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_9, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_10 = paddle._C_ops.shape(embedding_2)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_10, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_25 = [parameter_159, slice_14, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_7 = paddle._C_ops.memcpy_h2d(slice_14, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_26 = [parameter_159, memcpy_h2d_7, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_5 = paddle._C_ops.stack(combine_26, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_5 = paddle._C_ops.full_with_tensor(full_1, stack_5, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_27 = [parameter_159, slice_14, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_8 = paddle._C_ops.memcpy_h2d(slice_14, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_28 = [parameter_159, memcpy_h2d_8, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_6 = paddle._C_ops.stack(combine_28, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_6 = paddle._C_ops.full_with_tensor(full_1, stack_6, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_15 = paddle._C_ops.transpose(embedding_2, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_29 = [full_with_tensor_5, full_with_tensor_6]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__8, rnn__9, rnn__10, rnn__11 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_15, combine_29, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_16 = paddle._C_ops.transpose(rnn__8, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_17 = paddle._C_ops.transpose(transpose_16, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_30 = [slice_13, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_31 = [slice_13, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_18 = paddle._C_ops.transpose(transpose_17, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_6 = paddle.matmul(transpose_18, reshape_6, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__3 = paddle._C_ops.scale(matmul_6, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_11 = paddle._C_ops.shape(scale__3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_11, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_15, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__3, combine_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_9 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_2 = full_6 < memcpy_h2d_9

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_7 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_6 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_7 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_8 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64, xi32) <- (xb, xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9617 = 0
        while less_than_2:
            less_than_2, assign_value_8, full_7, assign_value_6, reshape__14, full_6, assign_value_7, = self.pd_op_while_9617_0_0(full_0, arange_0, feed_2, constant_23, parameter_194, constant_24, constant_25, parameter_195, parameter_196, cast_1, less_than_2, assign_value_8, full_7, assign_value_6, reshape__14, full_6, assign_value_7)
            while_loop_counter_9617 += 1
            if while_loop_counter_9617 > kWhileLoopLimit:
                break
            
        while_12, while_13, while_14, while_15, while_16, while_17, = assign_value_8, full_7, assign_value_6, reshape__14, full_6, assign_value_7,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_33 = [slice_15, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_13, combine_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__3 = paddle._C_ops.softmax(reshape__16, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_8, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_7 = paddle.matmul(softmax__3, transpose_19, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_20 = paddle._C_ops.transpose(matmul_7, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_20, [2], constant_3, constant_22, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_8, [1], constant_3, constant_22, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_34 = [slice_16, slice_17]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_34, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_8 = paddle.matmul(concat_1, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__22 = paddle._C_ops.add(matmul_8, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(add__22, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_18 = split_with_num_1[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__1 = paddle._C_ops.sigmoid(slice_18)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_19 = split_with_num_1[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__1 = paddle._C_ops.multiply(slice_19, sigmoid__1)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_9 = paddle.matmul(multiply__1, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__23 = paddle._C_ops.add(matmul_9, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__4 = paddle._C_ops.softmax(add__23, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_1 = paddle._C_ops.argmax(softmax__4, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_3 = paddle._C_ops.cast(argmax_1, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__0, cast_3, constant_22, constant_26, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_3 = paddle._C_ops.embedding(set_value_with_tensor__1, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_12 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_12, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_13 = paddle._C_ops.shape(embedding_3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(shape_13, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_35 = [parameter_159, slice_21, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_10 = paddle._C_ops.memcpy_h2d(slice_21, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_36 = [parameter_159, memcpy_h2d_10, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_7 = paddle._C_ops.stack(combine_36, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_7 = paddle._C_ops.full_with_tensor(full_1, stack_7, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_37 = [parameter_159, slice_21, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_11 = paddle._C_ops.memcpy_h2d(slice_21, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_38 = [parameter_159, memcpy_h2d_11, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_8 = paddle._C_ops.stack(combine_38, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_8 = paddle._C_ops.full_with_tensor(full_1, stack_8, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_21 = paddle._C_ops.transpose(embedding_3, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_39 = [full_with_tensor_7, full_with_tensor_8]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__12, rnn__13, rnn__14, rnn__15 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_21, combine_39, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_22 = paddle._C_ops.transpose(rnn__12, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_23 = paddle._C_ops.transpose(transpose_22, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_40 = [slice_20, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_41 = [slice_20, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_24 = paddle._C_ops.transpose(transpose_23, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_10 = paddle.matmul(transpose_24, reshape_10, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__4 = paddle._C_ops.scale(matmul_10, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_14 = paddle._C_ops.shape(scale__4)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(shape_14, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_22, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__4, combine_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_12 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_3 = full_8 < memcpy_h2d_12

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_9 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_9 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_10 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_11 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64) <- (xb, xi64, xi32, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9712 = 0
        while less_than_3:
            less_than_3, assign_value_11, assign_value_10, assign_value_9, reshape__18, full_9, full_8, = self.pd_op_while_9712_0_0(full_0, arange_0, feed_2, constant_27, parameter_197, constant_28, constant_29, parameter_198, parameter_199, cast_1, less_than_3, assign_value_11, assign_value_10, assign_value_9, reshape__18, full_9, full_8)
            while_loop_counter_9712 += 1
            if while_loop_counter_9712 > kWhileLoopLimit:
                break
            
        while_18, while_19, while_20, while_21, while_22, while_23, = assign_value_11, assign_value_10, assign_value_9, reshape__18, full_9, full_8,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_43 = [slice_22, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_22, combine_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__5 = paddle._C_ops.softmax(reshape__20, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_12, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_11 = paddle.matmul(softmax__5, transpose_25, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_26 = paddle._C_ops.transpose(matmul_11, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_26, [2], constant_22, constant_26, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_8, [1], constant_22, constant_26, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_44 = [slice_23, slice_24]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_44, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_12 = paddle.matmul(concat_2, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__24 = paddle._C_ops.add(matmul_12, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(add__24, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_25 = split_with_num_2[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__2 = paddle._C_ops.sigmoid(slice_25)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_26 = split_with_num_2[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__2 = paddle._C_ops.multiply(slice_26, sigmoid__2)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_13 = paddle.matmul(multiply__2, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__25 = paddle._C_ops.add(matmul_13, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__6 = paddle._C_ops.softmax(add__25, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_2 = paddle._C_ops.argmax(softmax__6, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_4 = paddle._C_ops.cast(argmax_2, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__1, cast_4, constant_26, constant_30, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_4 = paddle._C_ops.embedding(set_value_with_tensor__2, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_15 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(shape_15, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_16 = paddle._C_ops.shape(embedding_4)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_16, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_45 = [parameter_159, slice_28, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_13 = paddle._C_ops.memcpy_h2d(slice_28, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_46 = [parameter_159, memcpy_h2d_13, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_9 = paddle._C_ops.stack(combine_46, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_9 = paddle._C_ops.full_with_tensor(full_1, stack_9, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_47 = [parameter_159, slice_28, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_14 = paddle._C_ops.memcpy_h2d(slice_28, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_48 = [parameter_159, memcpy_h2d_14, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_10 = paddle._C_ops.stack(combine_48, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_10 = paddle._C_ops.full_with_tensor(full_1, stack_10, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_27 = paddle._C_ops.transpose(embedding_4, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_49 = [full_with_tensor_9, full_with_tensor_10]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__16, rnn__17, rnn__18, rnn__19 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_27, combine_49, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_28 = paddle._C_ops.transpose(rnn__16, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_29 = paddle._C_ops.transpose(transpose_28, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_50 = [slice_27, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_51 = [slice_27, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_30 = paddle._C_ops.transpose(transpose_29, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_14 = paddle.matmul(transpose_30, reshape_14, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__5 = paddle._C_ops.scale(matmul_14, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_17 = paddle._C_ops.shape(scale__5)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_17, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_52 = [slice_29, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__5, combine_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_15 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_4 = full_10 < memcpy_h2d_15

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_11 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_12 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_13 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_14 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, xi64, -1x40x6x40xf32, xi32, xf32) <- (xb, -1x40x6x40xf32, xi64, xi64, -1x40x6x40xf32, xi32, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9807 = 0
        while less_than_4:
            less_than_4, reshape__22, full_10, assign_value_14, full_11, assign_value_13, assign_value_12, = self.pd_op_while_9807_0_0(full_0, arange_0, feed_2, constant_31, parameter_200, constant_32, constant_33, parameter_201, parameter_202, cast_1, less_than_4, reshape__22, full_10, assign_value_14, full_11, assign_value_13, assign_value_12)
            while_loop_counter_9807 += 1
            if while_loop_counter_9807 > kWhileLoopLimit:
                break
            
        while_24, while_25, while_26, while_27, while_28, while_29, = reshape__22, full_10, assign_value_14, full_11, assign_value_13, assign_value_12,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_53 = [slice_29, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_27, combine_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__7 = paddle._C_ops.softmax(reshape__24, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_16, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_15 = paddle.matmul(softmax__7, transpose_31, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_32 = paddle._C_ops.transpose(matmul_15, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_32, [2], constant_26, constant_30, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_8, [1], constant_26, constant_30, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_54 = [slice_30, slice_31]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_54, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_16 = paddle.matmul(concat_3, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__26 = paddle._C_ops.add(matmul_16, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(add__26, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_32 = split_with_num_3[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__3 = paddle._C_ops.sigmoid(slice_32)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_33 = split_with_num_3[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__3 = paddle._C_ops.multiply(slice_33, sigmoid__3)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_17 = paddle.matmul(multiply__3, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__27 = paddle._C_ops.add(matmul_17, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__8 = paddle._C_ops.softmax(add__27, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_3 = paddle._C_ops.argmax(softmax__8, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_5 = paddle._C_ops.cast(argmax_3, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__3 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__2, cast_5, constant_30, constant_34, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_5 = paddle._C_ops.embedding(set_value_with_tensor__3, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_18 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(shape_18, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_19 = paddle._C_ops.shape(embedding_5)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(shape_19, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_55 = [parameter_159, slice_35, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_16 = paddle._C_ops.memcpy_h2d(slice_35, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_56 = [parameter_159, memcpy_h2d_16, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_11 = paddle._C_ops.stack(combine_56, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_11 = paddle._C_ops.full_with_tensor(full_1, stack_11, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_57 = [parameter_159, slice_35, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_17 = paddle._C_ops.memcpy_h2d(slice_35, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_58 = [parameter_159, memcpy_h2d_17, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_12 = paddle._C_ops.stack(combine_58, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_12 = paddle._C_ops.full_with_tensor(full_1, stack_12, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_33 = paddle._C_ops.transpose(embedding_5, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_59 = [full_with_tensor_11, full_with_tensor_12]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__20, rnn__21, rnn__22, rnn__23 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_33, combine_59, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_34 = paddle._C_ops.transpose(rnn__20, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_35 = paddle._C_ops.transpose(transpose_34, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_60 = [slice_34, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_61 = [slice_34, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_36 = paddle._C_ops.transpose(transpose_35, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_18 = paddle.matmul(transpose_36, reshape_18, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__6 = paddle._C_ops.scale(matmul_18, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_20 = paddle._C_ops.shape(scale__6)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(shape_20, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_62 = [slice_36, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__6, combine_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_12 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_18 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_5 = full_12 < memcpy_h2d_18

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_13 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_15 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_16 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_17 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, xi64, -1x40x6x40xf32, xi32, xf32) <- (xb, -1x40x6x40xf32, xi64, xi64, -1x40x6x40xf32, xi32, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9902 = 0
        while less_than_5:
            less_than_5, reshape__26, assign_value_17, full_12, full_13, assign_value_16, assign_value_15, = self.pd_op_while_9902_0_0(full_0, arange_0, feed_2, constant_35, parameter_203, constant_36, constant_37, parameter_204, parameter_205, cast_1, less_than_5, reshape__26, assign_value_17, full_12, full_13, assign_value_16, assign_value_15)
            while_loop_counter_9902 += 1
            if while_loop_counter_9902 > kWhileLoopLimit:
                break
            
        while_30, while_31, while_32, while_33, while_34, while_35, = reshape__26, assign_value_17, full_12, full_13, assign_value_16, assign_value_15,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_63 = [slice_36, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_33, combine_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__9 = paddle._C_ops.softmax(reshape__28, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_20, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_19 = paddle.matmul(softmax__9, transpose_37, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_38 = paddle._C_ops.transpose(matmul_19, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(transpose_38, [2], constant_30, constant_34, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(transpose_8, [1], constant_30, constant_34, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_64 = [slice_37, slice_38]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_64, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_20 = paddle.matmul(concat_4, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__28 = paddle._C_ops.add(matmul_20, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(add__28, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_39 = split_with_num_4[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__4 = paddle._C_ops.sigmoid(slice_39)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_40 = split_with_num_4[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__4 = paddle._C_ops.multiply(slice_40, sigmoid__4)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_21 = paddle.matmul(multiply__4, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__29 = paddle._C_ops.add(matmul_21, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__10 = paddle._C_ops.softmax(add__29, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_4 = paddle._C_ops.argmax(softmax__10, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_6 = paddle._C_ops.cast(argmax_4, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__4 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__3, cast_6, constant_34, constant_38, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_6 = paddle._C_ops.embedding(set_value_with_tensor__4, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_21 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(shape_21, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_22 = paddle._C_ops.shape(embedding_6)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(shape_22, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_65 = [parameter_159, slice_42, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_19 = paddle._C_ops.memcpy_h2d(slice_42, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_66 = [parameter_159, memcpy_h2d_19, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_13 = paddle._C_ops.stack(combine_66, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_13 = paddle._C_ops.full_with_tensor(full_1, stack_13, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_67 = [parameter_159, slice_42, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_20 = paddle._C_ops.memcpy_h2d(slice_42, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_68 = [parameter_159, memcpy_h2d_20, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_14 = paddle._C_ops.stack(combine_68, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_14 = paddle._C_ops.full_with_tensor(full_1, stack_14, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_39 = paddle._C_ops.transpose(embedding_6, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_69 = [full_with_tensor_13, full_with_tensor_14]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__24, rnn__25, rnn__26, rnn__27 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_39, combine_69, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_40 = paddle._C_ops.transpose(rnn__24, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_41 = paddle._C_ops.transpose(transpose_40, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_70 = [slice_41, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_71 = [slice_41, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_42 = paddle._C_ops.transpose(transpose_41, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_22 = paddle.matmul(transpose_42, reshape_22, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__7 = paddle._C_ops.scale(matmul_22, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_23 = paddle._C_ops.shape(scale__7)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(shape_23, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_72 = [slice_43, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__7, combine_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_14 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_21 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_6 = full_14 < memcpy_h2d_21

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_15 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_18 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_19 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_20 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64) <- (xb, xi64, xi32, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9997 = 0
        while less_than_6:
            less_than_6, full_14, assign_value_19, reshape__30, assign_value_18, full_15, assign_value_20, = self.pd_op_while_9997_0_0(full_0, arange_0, feed_2, constant_39, parameter_206, constant_40, constant_41, parameter_207, parameter_208, cast_1, less_than_6, full_14, assign_value_19, reshape__30, assign_value_18, full_15, assign_value_20)
            while_loop_counter_9997 += 1
            if while_loop_counter_9997 > kWhileLoopLimit:
                break
            
        while_36, while_37, while_38, while_39, while_40, while_41, = full_14, assign_value_19, reshape__30, assign_value_18, full_15, assign_value_20,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_73 = [slice_43, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_40, combine_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__11 = paddle._C_ops.softmax(reshape__32, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_24, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_23 = paddle.matmul(softmax__11, transpose_43, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_44 = paddle._C_ops.transpose(matmul_23, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(transpose_44, [2], constant_34, constant_38, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(transpose_8, [1], constant_34, constant_38, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_74 = [slice_44, slice_45]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_74, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_24 = paddle.matmul(concat_5, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__30 = paddle._C_ops.add(matmul_24, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(add__30, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_46 = split_with_num_5[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__5 = paddle._C_ops.sigmoid(slice_46)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_47 = split_with_num_5[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__5 = paddle._C_ops.multiply(slice_47, sigmoid__5)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_25 = paddle.matmul(multiply__5, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__31 = paddle._C_ops.add(matmul_25, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__12 = paddle._C_ops.softmax(add__31, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_5 = paddle._C_ops.argmax(softmax__12, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_7 = paddle._C_ops.cast(argmax_5, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__5 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__4, cast_7, constant_38, constant_42, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_7 = paddle._C_ops.embedding(set_value_with_tensor__5, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_24 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(shape_24, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_25 = paddle._C_ops.shape(embedding_7)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(shape_25, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_75 = [parameter_159, slice_49, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_22 = paddle._C_ops.memcpy_h2d(slice_49, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_76 = [parameter_159, memcpy_h2d_22, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_15 = paddle._C_ops.stack(combine_76, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_15 = paddle._C_ops.full_with_tensor(full_1, stack_15, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_77 = [parameter_159, slice_49, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_23 = paddle._C_ops.memcpy_h2d(slice_49, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_78 = [parameter_159, memcpy_h2d_23, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_16 = paddle._C_ops.stack(combine_78, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_16 = paddle._C_ops.full_with_tensor(full_1, stack_16, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_45 = paddle._C_ops.transpose(embedding_7, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_79 = [full_with_tensor_15, full_with_tensor_16]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__28, rnn__29, rnn__30, rnn__31 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_45, combine_79, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_46 = paddle._C_ops.transpose(rnn__28, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_47 = paddle._C_ops.transpose(transpose_46, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_80 = [slice_48, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_81 = [slice_48, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_48 = paddle._C_ops.transpose(transpose_47, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_26 = paddle.matmul(transpose_48, reshape_26, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__8 = paddle._C_ops.scale(matmul_26, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_26 = paddle._C_ops.shape(scale__8)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(shape_26, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_82 = [slice_50, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__8, combine_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_16 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_24 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_7 = full_16 < memcpy_h2d_24

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_17 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_21 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_22 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_23 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, -1x40x6x40xf32, xi32, xi64, xf32) <- (xb, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xi32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10092 = 0
        while less_than_7:
            less_than_7, full_17, full_16, reshape__34, assign_value_22, assign_value_23, assign_value_21, = self.pd_op_while_10092_0_0(full_0, arange_0, feed_2, constant_43, parameter_209, constant_44, constant_45, parameter_210, parameter_211, cast_1, less_than_7, full_17, full_16, reshape__34, assign_value_22, assign_value_23, assign_value_21)
            while_loop_counter_10092 += 1
            if while_loop_counter_10092 > kWhileLoopLimit:
                break
            
        while_42, while_43, while_44, while_45, while_46, while_47, = full_17, full_16, reshape__34, assign_value_22, assign_value_23, assign_value_21,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_83 = [slice_50, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_42, combine_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__13 = paddle._C_ops.softmax(reshape__36, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_28, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_27 = paddle.matmul(softmax__13, transpose_49, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_50 = paddle._C_ops.transpose(matmul_27, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(transpose_50, [2], constant_38, constant_42, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(transpose_8, [1], constant_38, constant_42, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_84 = [slice_51, slice_52]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_84, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_28 = paddle.matmul(concat_6, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__32 = paddle._C_ops.add(matmul_28, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(add__32, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_53 = split_with_num_6[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__6 = paddle._C_ops.sigmoid(slice_53)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_54 = split_with_num_6[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__6 = paddle._C_ops.multiply(slice_54, sigmoid__6)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_29 = paddle.matmul(multiply__6, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__33 = paddle._C_ops.add(matmul_29, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__14 = paddle._C_ops.softmax(add__33, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_6 = paddle._C_ops.argmax(softmax__14, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_8 = paddle._C_ops.cast(argmax_6, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__6 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__5, cast_8, constant_42, constant_46, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_8 = paddle._C_ops.embedding(set_value_with_tensor__6, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_27 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(shape_27, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_28 = paddle._C_ops.shape(embedding_8)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(shape_28, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_85 = [parameter_159, slice_56, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_25 = paddle._C_ops.memcpy_h2d(slice_56, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_86 = [parameter_159, memcpy_h2d_25, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_17 = paddle._C_ops.stack(combine_86, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_17 = paddle._C_ops.full_with_tensor(full_1, stack_17, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_87 = [parameter_159, slice_56, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_26 = paddle._C_ops.memcpy_h2d(slice_56, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_88 = [parameter_159, memcpy_h2d_26, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_18 = paddle._C_ops.stack(combine_88, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_18 = paddle._C_ops.full_with_tensor(full_1, stack_18, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_51 = paddle._C_ops.transpose(embedding_8, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_89 = [full_with_tensor_17, full_with_tensor_18]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__32, rnn__33, rnn__34, rnn__35 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_51, combine_89, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_52 = paddle._C_ops.transpose(rnn__32, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_53 = paddle._C_ops.transpose(transpose_52, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_90 = [slice_55, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_90), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_91 = [slice_55, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_91), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_54 = paddle._C_ops.transpose(transpose_53, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_30 = paddle.matmul(transpose_54, reshape_30, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__9 = paddle._C_ops.scale(matmul_30, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_29 = paddle._C_ops.shape(scale__9)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(shape_29, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_92 = [slice_57, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__9, combine_92), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_18 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_27 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_8 = full_18 < memcpy_h2d_27

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_19 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_24 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_25 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_26 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32, xi32, xi64) <- (xb, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32, xi32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10187 = 0
        while less_than_8:
            less_than_8, reshape__38, assign_value_24, assign_value_26, full_19, assign_value_25, full_18, = self.pd_op_while_10187_0_0(full_0, arange_0, feed_2, constant_47, parameter_212, constant_48, constant_49, parameter_213, parameter_214, cast_1, less_than_8, reshape__38, assign_value_24, assign_value_26, full_19, assign_value_25, full_18)
            while_loop_counter_10187 += 1
            if while_loop_counter_10187 > kWhileLoopLimit:
                break
            
        while_48, while_49, while_50, while_51, while_52, while_53, = reshape__38, assign_value_24, assign_value_26, full_19, assign_value_25, full_18,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_93 = [slice_57, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_51, combine_93), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__15 = paddle._C_ops.softmax(reshape__40, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_32, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_31 = paddle.matmul(softmax__15, transpose_55, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_56 = paddle._C_ops.transpose(matmul_31, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(transpose_56, [2], constant_42, constant_46, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(transpose_8, [1], constant_42, constant_46, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_94 = [slice_58, slice_59]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_94, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_32 = paddle.matmul(concat_7, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__34 = paddle._C_ops.add(matmul_32, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(add__34, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_60 = split_with_num_7[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__7 = paddle._C_ops.sigmoid(slice_60)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_61 = split_with_num_7[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__7 = paddle._C_ops.multiply(slice_61, sigmoid__7)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_33 = paddle.matmul(multiply__7, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__35 = paddle._C_ops.add(matmul_33, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__16 = paddle._C_ops.softmax(add__35, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_7 = paddle._C_ops.argmax(softmax__16, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_9 = paddle._C_ops.cast(argmax_7, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__7 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__6, cast_9, constant_46, constant_50, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_9 = paddle._C_ops.embedding(set_value_with_tensor__7, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_30 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(shape_30, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_31 = paddle._C_ops.shape(embedding_9)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(shape_31, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_95 = [parameter_159, slice_63, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_28 = paddle._C_ops.memcpy_h2d(slice_63, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_96 = [parameter_159, memcpy_h2d_28, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_19 = paddle._C_ops.stack(combine_96, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_19 = paddle._C_ops.full_with_tensor(full_1, stack_19, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_97 = [parameter_159, slice_63, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_29 = paddle._C_ops.memcpy_h2d(slice_63, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_98 = [parameter_159, memcpy_h2d_29, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_20 = paddle._C_ops.stack(combine_98, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_20 = paddle._C_ops.full_with_tensor(full_1, stack_20, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_57 = paddle._C_ops.transpose(embedding_9, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_99 = [full_with_tensor_19, full_with_tensor_20]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__36, rnn__37, rnn__38, rnn__39 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_57, combine_99, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_58 = paddle._C_ops.transpose(rnn__36, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_59 = paddle._C_ops.transpose(transpose_58, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_100 = [slice_62, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_100), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_101 = [slice_62, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_101), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_60 = paddle._C_ops.transpose(transpose_59, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_34 = paddle.matmul(transpose_60, reshape_34, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__10 = paddle._C_ops.scale(matmul_34, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_32 = paddle._C_ops.shape(scale__10)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(shape_32, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_102 = [slice_64, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__10, combine_102), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_20 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_30 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_9 = full_20 < memcpy_h2d_30

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_21 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_27 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_28 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_29 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32) <- (xb, xi32, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10282 = 0
        while less_than_9:
            less_than_9, assign_value_28, full_20, reshape__42, full_21, assign_value_29, assign_value_27, = self.pd_op_while_10282_0_0(full_0, arange_0, feed_2, constant_51, parameter_215, constant_52, constant_53, parameter_216, parameter_217, cast_1, less_than_9, assign_value_28, full_20, reshape__42, full_21, assign_value_29, assign_value_27)
            while_loop_counter_10282 += 1
            if while_loop_counter_10282 > kWhileLoopLimit:
                break
            
        while_54, while_55, while_56, while_57, while_58, while_59, = assign_value_28, full_20, reshape__42, full_21, assign_value_29, assign_value_27,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_103 = [slice_64, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_57, combine_103), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__17 = paddle._C_ops.softmax(reshape__44, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_36, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_35 = paddle.matmul(softmax__17, transpose_61, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_62 = paddle._C_ops.transpose(matmul_35, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(transpose_62, [2], constant_46, constant_50, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(transpose_8, [1], constant_46, constant_50, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_104 = [slice_65, slice_66]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_104, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_36 = paddle.matmul(concat_8, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__36 = paddle._C_ops.add(matmul_36, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(add__36, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_67 = split_with_num_8[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__8 = paddle._C_ops.sigmoid(slice_67)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_68 = split_with_num_8[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__8 = paddle._C_ops.multiply(slice_68, sigmoid__8)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_37 = paddle.matmul(multiply__8, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__37 = paddle._C_ops.add(matmul_37, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__18 = paddle._C_ops.softmax(add__37, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_8 = paddle._C_ops.argmax(softmax__18, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_10 = paddle._C_ops.cast(argmax_8, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__8 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__7, cast_10, constant_50, constant_54, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_10 = paddle._C_ops.embedding(set_value_with_tensor__8, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_33 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(shape_33, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_34 = paddle._C_ops.shape(embedding_10)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(shape_34, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_105 = [parameter_159, slice_70, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_31 = paddle._C_ops.memcpy_h2d(slice_70, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_106 = [parameter_159, memcpy_h2d_31, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_21 = paddle._C_ops.stack(combine_106, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_21 = paddle._C_ops.full_with_tensor(full_1, stack_21, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_107 = [parameter_159, slice_70, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_32 = paddle._C_ops.memcpy_h2d(slice_70, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_108 = [parameter_159, memcpy_h2d_32, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_22 = paddle._C_ops.stack(combine_108, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_22 = paddle._C_ops.full_with_tensor(full_1, stack_22, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_63 = paddle._C_ops.transpose(embedding_10, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_109 = [full_with_tensor_21, full_with_tensor_22]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__40, rnn__41, rnn__42, rnn__43 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_63, combine_109, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_64 = paddle._C_ops.transpose(rnn__40, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_65 = paddle._C_ops.transpose(transpose_64, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_110 = [slice_69, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_110), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_111 = [slice_69, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_111), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_66 = paddle._C_ops.transpose(transpose_65, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_38 = paddle.matmul(transpose_66, reshape_38, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__11 = paddle._C_ops.scale(matmul_38, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_35 = paddle._C_ops.shape(scale__11)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(shape_35, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_112 = [slice_71, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__11, combine_112), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_22 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_33 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_10 = full_22 < memcpy_h2d_33

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_23 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_30 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_31 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_32 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64, xi32) <- (xb, xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10377 = 0
        while less_than_10:
            less_than_10, assign_value_32, full_23, assign_value_30, reshape__46, full_22, assign_value_31, = self.pd_op_while_10377_0_0(full_0, arange_0, feed_2, constant_55, parameter_218, constant_56, constant_57, parameter_219, parameter_220, cast_1, less_than_10, assign_value_32, full_23, assign_value_30, reshape__46, full_22, assign_value_31)
            while_loop_counter_10377 += 1
            if while_loop_counter_10377 > kWhileLoopLimit:
                break
            
        while_60, while_61, while_62, while_63, while_64, while_65, = assign_value_32, full_23, assign_value_30, reshape__46, full_22, assign_value_31,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_113 = [slice_71, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_61, combine_113), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__19 = paddle._C_ops.softmax(reshape__48, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_40, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_39 = paddle.matmul(softmax__19, transpose_67, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_68 = paddle._C_ops.transpose(matmul_39, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(transpose_68, [2], constant_50, constant_54, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(transpose_8, [1], constant_50, constant_54, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_114 = [slice_72, slice_73]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_114, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_40 = paddle.matmul(concat_9, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__38 = paddle._C_ops.add(matmul_40, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(add__38, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_74 = split_with_num_9[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__9 = paddle._C_ops.sigmoid(slice_74)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_75 = split_with_num_9[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__9 = paddle._C_ops.multiply(slice_75, sigmoid__9)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_41 = paddle.matmul(multiply__9, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__39 = paddle._C_ops.add(matmul_41, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__20 = paddle._C_ops.softmax(add__39, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_9 = paddle._C_ops.argmax(softmax__20, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_11 = paddle._C_ops.cast(argmax_9, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__9 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__8, cast_11, constant_54, constant_58, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_11 = paddle._C_ops.embedding(set_value_with_tensor__9, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_36 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(shape_36, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_37 = paddle._C_ops.shape(embedding_11)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(shape_37, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_115 = [parameter_159, slice_77, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_34 = paddle._C_ops.memcpy_h2d(slice_77, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_116 = [parameter_159, memcpy_h2d_34, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_23 = paddle._C_ops.stack(combine_116, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_23 = paddle._C_ops.full_with_tensor(full_1, stack_23, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_117 = [parameter_159, slice_77, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_35 = paddle._C_ops.memcpy_h2d(slice_77, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_118 = [parameter_159, memcpy_h2d_35, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_24 = paddle._C_ops.stack(combine_118, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_24 = paddle._C_ops.full_with_tensor(full_1, stack_24, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_69 = paddle._C_ops.transpose(embedding_11, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_119 = [full_with_tensor_23, full_with_tensor_24]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__44, rnn__45, rnn__46, rnn__47 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_69, combine_119, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_70 = paddle._C_ops.transpose(rnn__44, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_71 = paddle._C_ops.transpose(transpose_70, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_120 = [slice_76, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_120), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_121 = [slice_76, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_121), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_72 = paddle._C_ops.transpose(transpose_71, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_42 = paddle.matmul(transpose_72, reshape_42, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__12 = paddle._C_ops.scale(matmul_42, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_38 = paddle._C_ops.shape(scale__12)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(shape_38, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_122 = [slice_78, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__12, combine_122), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_24 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_36 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_11 = full_24 < memcpy_h2d_36

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_25 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_33 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_34 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_35 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xf32, xi64, -1x40x6x40xf32, xi64, -1x40x6x40xf32) <- (xb, xi32, xf32, xi64, -1x40x6x40xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10472 = 0
        while less_than_11:
            less_than_11, assign_value_34, assign_value_33, assign_value_35, full_25, full_24, reshape__50, = self.pd_op_while_10472_0_0(full_0, arange_0, feed_2, constant_59, parameter_221, constant_60, constant_61, parameter_222, parameter_223, cast_1, less_than_11, assign_value_34, assign_value_33, assign_value_35, full_25, full_24, reshape__50)
            while_loop_counter_10472 += 1
            if while_loop_counter_10472 > kWhileLoopLimit:
                break
            
        while_66, while_67, while_68, while_69, while_70, while_71, = assign_value_34, assign_value_33, assign_value_35, full_25, full_24, reshape__50,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_123 = [slice_78, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_69, combine_123), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__21 = paddle._C_ops.softmax(reshape__52, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_44, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_43 = paddle.matmul(softmax__21, transpose_73, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_74 = paddle._C_ops.transpose(matmul_43, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(transpose_74, [2], constant_54, constant_58, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(transpose_8, [1], constant_54, constant_58, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_124 = [slice_79, slice_80]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_124, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_44 = paddle.matmul(concat_10, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__40 = paddle._C_ops.add(matmul_44, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(add__40, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_81 = split_with_num_10[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__10 = paddle._C_ops.sigmoid(slice_81)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_82 = split_with_num_10[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__10 = paddle._C_ops.multiply(slice_82, sigmoid__10)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_45 = paddle.matmul(multiply__10, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__41 = paddle._C_ops.add(matmul_45, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__22 = paddle._C_ops.softmax(add__41, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_10 = paddle._C_ops.argmax(softmax__22, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_12 = paddle._C_ops.cast(argmax_10, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__10 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__9, cast_12, constant_58, constant_62, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_12 = paddle._C_ops.embedding(set_value_with_tensor__10, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_39 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(shape_39, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_40 = paddle._C_ops.shape(embedding_12)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(shape_40, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_125 = [parameter_159, slice_84, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_37 = paddle._C_ops.memcpy_h2d(slice_84, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_126 = [parameter_159, memcpy_h2d_37, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_25 = paddle._C_ops.stack(combine_126, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_25 = paddle._C_ops.full_with_tensor(full_1, stack_25, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_127 = [parameter_159, slice_84, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_38 = paddle._C_ops.memcpy_h2d(slice_84, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_128 = [parameter_159, memcpy_h2d_38, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_26 = paddle._C_ops.stack(combine_128, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_26 = paddle._C_ops.full_with_tensor(full_1, stack_26, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_75 = paddle._C_ops.transpose(embedding_12, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_129 = [full_with_tensor_25, full_with_tensor_26]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__48, rnn__49, rnn__50, rnn__51 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_75, combine_129, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_76 = paddle._C_ops.transpose(rnn__48, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_77 = paddle._C_ops.transpose(transpose_76, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_130 = [slice_83, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_130), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_131 = [slice_83, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_131), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_78 = paddle._C_ops.transpose(transpose_77, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_46 = paddle.matmul(transpose_78, reshape_46, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__13 = paddle._C_ops.scale(matmul_46, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_41 = paddle._C_ops.shape(scale__13)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(shape_41, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_132 = [slice_85, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__13, combine_132), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_26 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_39 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_12 = full_26 < memcpy_h2d_39

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_27 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_36 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_37 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_38 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, -1x40x6x40xf32, xi32, xf32, xi64, xi64) <- (xb, -1x40x6x40xf32, -1x40x6x40xf32, xi32, xf32, xi64, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10567 = 0
        while less_than_12:
            less_than_12, full_27, reshape__54, assign_value_37, assign_value_36, full_26, assign_value_38, = self.pd_op_while_10567_0_0(full_0, arange_0, feed_2, constant_63, parameter_224, constant_64, constant_65, parameter_225, parameter_226, cast_1, less_than_12, full_27, reshape__54, assign_value_37, assign_value_36, full_26, assign_value_38)
            while_loop_counter_10567 += 1
            if while_loop_counter_10567 > kWhileLoopLimit:
                break
            
        while_72, while_73, while_74, while_75, while_76, while_77, = full_27, reshape__54, assign_value_37, assign_value_36, full_26, assign_value_38,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_133 = [slice_85, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_72, combine_133), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__23 = paddle._C_ops.softmax(reshape__56, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_79 = paddle._C_ops.transpose(reshape_48, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_47 = paddle.matmul(softmax__23, transpose_79, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_80 = paddle._C_ops.transpose(matmul_47, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(transpose_80, [2], constant_58, constant_62, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(transpose_8, [1], constant_58, constant_62, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_134 = [slice_86, slice_87]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_134, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_48 = paddle.matmul(concat_11, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__42 = paddle._C_ops.add(matmul_48, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(add__42, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_88 = split_with_num_11[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__11 = paddle._C_ops.sigmoid(slice_88)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_89 = split_with_num_11[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__11 = paddle._C_ops.multiply(slice_89, sigmoid__11)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_49 = paddle.matmul(multiply__11, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__43 = paddle._C_ops.add(matmul_49, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__24 = paddle._C_ops.softmax(add__43, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_11 = paddle._C_ops.argmax(softmax__24, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_13 = paddle._C_ops.cast(argmax_11, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__11 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__10, cast_13, constant_62, constant_66, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_13 = paddle._C_ops.embedding(set_value_with_tensor__11, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_42 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(shape_42, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_43 = paddle._C_ops.shape(embedding_13)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(shape_43, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_135 = [parameter_159, slice_91, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_40 = paddle._C_ops.memcpy_h2d(slice_91, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_136 = [parameter_159, memcpy_h2d_40, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_27 = paddle._C_ops.stack(combine_136, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_27 = paddle._C_ops.full_with_tensor(full_1, stack_27, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_137 = [parameter_159, slice_91, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_41 = paddle._C_ops.memcpy_h2d(slice_91, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_138 = [parameter_159, memcpy_h2d_41, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_28 = paddle._C_ops.stack(combine_138, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_28 = paddle._C_ops.full_with_tensor(full_1, stack_28, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_81 = paddle._C_ops.transpose(embedding_13, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_139 = [full_with_tensor_27, full_with_tensor_28]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__52, rnn__53, rnn__54, rnn__55 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_81, combine_139, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_82 = paddle._C_ops.transpose(rnn__52, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_83 = paddle._C_ops.transpose(transpose_82, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_140 = [slice_90, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_140), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_141 = [slice_90, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_141), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_84 = paddle._C_ops.transpose(transpose_83, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_50 = paddle.matmul(transpose_84, reshape_50, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__14 = paddle._C_ops.scale(matmul_50, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_44 = paddle._C_ops.shape(scale__14)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(shape_44, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_142 = [slice_92, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__14, combine_142), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_28 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_42 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_13 = full_28 < memcpy_h2d_42

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_29 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_39 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_40 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_41 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64) <- (xb, xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10662 = 0
        while less_than_13:
            less_than_13, full_28, assign_value_40, reshape__58, full_29, assign_value_39, assign_value_41, = self.pd_op_while_10662_0_0(full_0, arange_0, feed_2, constant_67, parameter_227, constant_68, constant_69, parameter_228, parameter_229, cast_1, less_than_13, full_28, assign_value_40, reshape__58, full_29, assign_value_39, assign_value_41)
            while_loop_counter_10662 += 1
            if while_loop_counter_10662 > kWhileLoopLimit:
                break
            
        while_78, while_79, while_80, while_81, while_82, while_83, = full_28, assign_value_40, reshape__58, full_29, assign_value_39, assign_value_41,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_143 = [slice_92, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_81, combine_143), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__25 = paddle._C_ops.softmax(reshape__60, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_85 = paddle._C_ops.transpose(reshape_52, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_51 = paddle.matmul(softmax__25, transpose_85, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_86 = paddle._C_ops.transpose(matmul_51, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(transpose_86, [2], constant_62, constant_66, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(transpose_8, [1], constant_62, constant_66, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_144 = [slice_93, slice_94]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_144, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_52 = paddle.matmul(concat_12, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__44 = paddle._C_ops.add(matmul_52, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_12 = paddle._C_ops.split_with_num(add__44, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_95 = split_with_num_12[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__12 = paddle._C_ops.sigmoid(slice_95)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_96 = split_with_num_12[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__12 = paddle._C_ops.multiply(slice_96, sigmoid__12)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_53 = paddle.matmul(multiply__12, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__45 = paddle._C_ops.add(matmul_53, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__26 = paddle._C_ops.softmax(add__45, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_12 = paddle._C_ops.argmax(softmax__26, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_14 = paddle._C_ops.cast(argmax_12, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__12 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__11, cast_14, constant_66, constant_70, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_14 = paddle._C_ops.embedding(set_value_with_tensor__12, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_45 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(shape_45, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_46 = paddle._C_ops.shape(embedding_14)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(shape_46, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_145 = [parameter_159, slice_98, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_43 = paddle._C_ops.memcpy_h2d(slice_98, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_146 = [parameter_159, memcpy_h2d_43, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_29 = paddle._C_ops.stack(combine_146, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_29 = paddle._C_ops.full_with_tensor(full_1, stack_29, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_147 = [parameter_159, slice_98, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_44 = paddle._C_ops.memcpy_h2d(slice_98, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_148 = [parameter_159, memcpy_h2d_44, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_30 = paddle._C_ops.stack(combine_148, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_30 = paddle._C_ops.full_with_tensor(full_1, stack_30, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_87 = paddle._C_ops.transpose(embedding_14, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_149 = [full_with_tensor_29, full_with_tensor_30]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__56, rnn__57, rnn__58, rnn__59 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_87, combine_149, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_88 = paddle._C_ops.transpose(rnn__56, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_89 = paddle._C_ops.transpose(transpose_88, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_150 = [slice_97, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_150), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_151 = [slice_97, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_151), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_90 = paddle._C_ops.transpose(transpose_89, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_54 = paddle.matmul(transpose_90, reshape_54, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__15 = paddle._C_ops.scale(matmul_54, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_47 = paddle._C_ops.shape(scale__15)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(shape_47, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_152 = [slice_99, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__15, combine_152), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_30 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_45 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_14 = full_30 < memcpy_h2d_45

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_31 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_42 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_43 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_44 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64, xi64) <- (xb, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10757 = 0
        while less_than_14:
            less_than_14, assign_value_43, full_31, reshape__62, assign_value_42, full_30, assign_value_44, = self.pd_op_while_10757_0_0(full_0, arange_0, feed_2, constant_71, parameter_230, constant_72, constant_73, parameter_231, parameter_232, cast_1, less_than_14, assign_value_43, full_31, reshape__62, assign_value_42, full_30, assign_value_44)
            while_loop_counter_10757 += 1
            if while_loop_counter_10757 > kWhileLoopLimit:
                break
            
        while_84, while_85, while_86, while_87, while_88, while_89, = assign_value_43, full_31, reshape__62, assign_value_42, full_30, assign_value_44,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_153 = [slice_99, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_85, combine_153), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__27 = paddle._C_ops.softmax(reshape__64, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_91 = paddle._C_ops.transpose(reshape_56, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_55 = paddle.matmul(softmax__27, transpose_91, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_92 = paddle._C_ops.transpose(matmul_55, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_100 = paddle._C_ops.slice(transpose_92, [2], constant_66, constant_70, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_101 = paddle._C_ops.slice(transpose_8, [1], constant_66, constant_70, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_154 = [slice_100, slice_101]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_154, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_56 = paddle.matmul(concat_13, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__46 = paddle._C_ops.add(matmul_56, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_13 = paddle._C_ops.split_with_num(add__46, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_102 = split_with_num_13[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__13 = paddle._C_ops.sigmoid(slice_102)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_103 = split_with_num_13[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__13 = paddle._C_ops.multiply(slice_103, sigmoid__13)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_57 = paddle.matmul(multiply__13, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__47 = paddle._C_ops.add(matmul_57, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__28 = paddle._C_ops.softmax(add__47, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_13 = paddle._C_ops.argmax(softmax__28, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_15 = paddle._C_ops.cast(argmax_13, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__13 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__12, cast_15, constant_70, constant_74, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_15 = paddle._C_ops.embedding(set_value_with_tensor__13, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_48 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(shape_48, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_49 = paddle._C_ops.shape(embedding_15)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(shape_49, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_155 = [parameter_159, slice_105, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_46 = paddle._C_ops.memcpy_h2d(slice_105, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_156 = [parameter_159, memcpy_h2d_46, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_31 = paddle._C_ops.stack(combine_156, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_31 = paddle._C_ops.full_with_tensor(full_1, stack_31, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_157 = [parameter_159, slice_105, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_47 = paddle._C_ops.memcpy_h2d(slice_105, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_158 = [parameter_159, memcpy_h2d_47, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_32 = paddle._C_ops.stack(combine_158, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_32 = paddle._C_ops.full_with_tensor(full_1, stack_32, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_93 = paddle._C_ops.transpose(embedding_15, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_159 = [full_with_tensor_31, full_with_tensor_32]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__60, rnn__61, rnn__62, rnn__63 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_93, combine_159, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_94 = paddle._C_ops.transpose(rnn__60, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_95 = paddle._C_ops.transpose(transpose_94, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_160 = [slice_104, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_160), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_161 = [slice_104, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_161), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_96 = paddle._C_ops.transpose(transpose_95, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_58 = paddle.matmul(transpose_96, reshape_58, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__16 = paddle._C_ops.scale(matmul_58, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_50 = paddle._C_ops.shape(scale__16)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(shape_50, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_162 = [slice_106, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__16, combine_162), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_32 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_48 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_15 = full_32 < memcpy_h2d_48

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_33 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_45 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_46 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_47 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, xi32, -1x40x6x40xf32, xi64) <- (xb, xi64, -1x40x6x40xf32, xf32, xi32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10852 = 0
        while less_than_15:
            less_than_15, assign_value_47, full_33, assign_value_45, assign_value_46, reshape__66, full_32, = self.pd_op_while_10852_0_0(full_0, arange_0, feed_2, constant_75, parameter_233, constant_76, constant_77, parameter_234, parameter_235, cast_1, less_than_15, assign_value_47, full_33, assign_value_45, assign_value_46, reshape__66, full_32)
            while_loop_counter_10852 += 1
            if while_loop_counter_10852 > kWhileLoopLimit:
                break
            
        while_90, while_91, while_92, while_93, while_94, while_95, = assign_value_47, full_33, assign_value_45, assign_value_46, reshape__66, full_32,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_163 = [slice_106, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_91, combine_163), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__29 = paddle._C_ops.softmax(reshape__68, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_97 = paddle._C_ops.transpose(reshape_60, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_59 = paddle.matmul(softmax__29, transpose_97, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_98 = paddle._C_ops.transpose(matmul_59, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(transpose_98, [2], constant_70, constant_74, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_108 = paddle._C_ops.slice(transpose_8, [1], constant_70, constant_74, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_164 = [slice_107, slice_108]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_164, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_60 = paddle.matmul(concat_14, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__48 = paddle._C_ops.add(matmul_60, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_14 = paddle._C_ops.split_with_num(add__48, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_109 = split_with_num_14[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__14 = paddle._C_ops.sigmoid(slice_109)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_110 = split_with_num_14[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__14 = paddle._C_ops.multiply(slice_110, sigmoid__14)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_61 = paddle.matmul(multiply__14, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__49 = paddle._C_ops.add(matmul_61, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__30 = paddle._C_ops.softmax(add__49, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_14 = paddle._C_ops.argmax(softmax__30, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_16 = paddle._C_ops.cast(argmax_14, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__14 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__13, cast_16, constant_74, constant_78, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_16 = paddle._C_ops.embedding(set_value_with_tensor__14, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_51 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_111 = paddle._C_ops.slice(shape_51, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_52 = paddle._C_ops.shape(embedding_16)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(shape_52, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_165 = [parameter_159, slice_112, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_49 = paddle._C_ops.memcpy_h2d(slice_112, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_166 = [parameter_159, memcpy_h2d_49, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_33 = paddle._C_ops.stack(combine_166, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_33 = paddle._C_ops.full_with_tensor(full_1, stack_33, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_167 = [parameter_159, slice_112, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_50 = paddle._C_ops.memcpy_h2d(slice_112, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_168 = [parameter_159, memcpy_h2d_50, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_34 = paddle._C_ops.stack(combine_168, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_34 = paddle._C_ops.full_with_tensor(full_1, stack_34, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_99 = paddle._C_ops.transpose(embedding_16, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_169 = [full_with_tensor_33, full_with_tensor_34]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__64, rnn__65, rnn__66, rnn__67 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_99, combine_169, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_100 = paddle._C_ops.transpose(rnn__64, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_101 = paddle._C_ops.transpose(transpose_100, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_170 = [slice_111, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_170), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_171 = [slice_111, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_171), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_102 = paddle._C_ops.transpose(transpose_101, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_62 = paddle.matmul(transpose_102, reshape_62, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__17 = paddle._C_ops.scale(matmul_62, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_53 = paddle._C_ops.shape(scale__17)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_113 = paddle._C_ops.slice(shape_53, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_172 = [slice_113, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__17, combine_172), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_34 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_51 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_16 = full_34 < memcpy_h2d_51

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_35 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_48 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_49 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_50 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xf32, xi64, -1x40x6x40xf32, xi64, xi32, -1x40x6x40xf32) <- (xb, xf32, xi64, -1x40x6x40xf32, xi64, xi32, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10947 = 0
        while less_than_16:
            less_than_16, assign_value_48, full_34, full_35, assign_value_50, assign_value_49, reshape__70, = self.pd_op_while_10947_0_0(full_0, arange_0, feed_2, constant_79, parameter_236, constant_80, constant_81, parameter_237, parameter_238, cast_1, less_than_16, assign_value_48, full_34, full_35, assign_value_50, assign_value_49, reshape__70)
            while_loop_counter_10947 += 1
            if while_loop_counter_10947 > kWhileLoopLimit:
                break
            
        while_96, while_97, while_98, while_99, while_100, while_101, = assign_value_48, full_34, full_35, assign_value_50, assign_value_49, reshape__70,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_173 = [slice_113, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_98, combine_173), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__31 = paddle._C_ops.softmax(reshape__72, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_103 = paddle._C_ops.transpose(reshape_64, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_63 = paddle.matmul(softmax__31, transpose_103, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_104 = paddle._C_ops.transpose(matmul_63, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_114 = paddle._C_ops.slice(transpose_104, [2], constant_74, constant_78, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_115 = paddle._C_ops.slice(transpose_8, [1], constant_74, constant_78, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_174 = [slice_114, slice_115]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_174, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_64 = paddle.matmul(concat_15, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__50 = paddle._C_ops.add(matmul_64, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_15 = paddle._C_ops.split_with_num(add__50, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_116 = split_with_num_15[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__15 = paddle._C_ops.sigmoid(slice_116)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_117 = split_with_num_15[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__15 = paddle._C_ops.multiply(slice_117, sigmoid__15)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_65 = paddle.matmul(multiply__15, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__51 = paddle._C_ops.add(matmul_65, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__32 = paddle._C_ops.softmax(add__51, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_15 = paddle._C_ops.argmax(softmax__32, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_17 = paddle._C_ops.cast(argmax_15, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__15 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__14, cast_17, constant_78, constant_82, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_17 = paddle._C_ops.embedding(set_value_with_tensor__15, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_54 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_118 = paddle._C_ops.slice(shape_54, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_55 = paddle._C_ops.shape(embedding_17)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(shape_55, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_175 = [parameter_159, slice_119, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_52 = paddle._C_ops.memcpy_h2d(slice_119, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_176 = [parameter_159, memcpy_h2d_52, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_35 = paddle._C_ops.stack(combine_176, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_35 = paddle._C_ops.full_with_tensor(full_1, stack_35, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_177 = [parameter_159, slice_119, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_53 = paddle._C_ops.memcpy_h2d(slice_119, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_178 = [parameter_159, memcpy_h2d_53, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_36 = paddle._C_ops.stack(combine_178, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_36 = paddle._C_ops.full_with_tensor(full_1, stack_36, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_105 = paddle._C_ops.transpose(embedding_17, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_179 = [full_with_tensor_35, full_with_tensor_36]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__68, rnn__69, rnn__70, rnn__71 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_105, combine_179, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_106 = paddle._C_ops.transpose(rnn__68, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_107 = paddle._C_ops.transpose(transpose_106, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_180 = [slice_118, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_180), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_181 = [slice_118, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_181), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_108 = paddle._C_ops.transpose(transpose_107, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_66 = paddle.matmul(transpose_108, reshape_66, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__18 = paddle._C_ops.scale(matmul_66, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_56 = paddle._C_ops.shape(scale__18)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_120 = paddle._C_ops.slice(shape_56, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_182 = [slice_120, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__18, combine_182), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_36 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_54 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_17 = full_36 < memcpy_h2d_54

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_37 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_51 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_52 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_53 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, xi32, -1x40x6x40xf32, xi64) <- (xb, xi64, -1x40x6x40xf32, xf32, xi32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11042 = 0
        while less_than_17:
            less_than_17, assign_value_53, full_37, assign_value_51, assign_value_52, reshape__74, full_36, = self.pd_op_while_11042_0_0(full_0, arange_0, feed_2, constant_83, parameter_239, constant_84, constant_85, parameter_240, parameter_241, cast_1, less_than_17, assign_value_53, full_37, assign_value_51, assign_value_52, reshape__74, full_36)
            while_loop_counter_11042 += 1
            if while_loop_counter_11042 > kWhileLoopLimit:
                break
            
        while_102, while_103, while_104, while_105, while_106, while_107, = assign_value_53, full_37, assign_value_51, assign_value_52, reshape__74, full_36,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_183 = [slice_120, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_103, combine_183), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__33 = paddle._C_ops.softmax(reshape__76, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_109 = paddle._C_ops.transpose(reshape_68, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_67 = paddle.matmul(softmax__33, transpose_109, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_110 = paddle._C_ops.transpose(matmul_67, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_121 = paddle._C_ops.slice(transpose_110, [2], constant_78, constant_82, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(transpose_8, [1], constant_78, constant_82, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_184 = [slice_121, slice_122]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_184, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_68 = paddle.matmul(concat_16, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__52 = paddle._C_ops.add(matmul_68, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_16 = paddle._C_ops.split_with_num(add__52, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_123 = split_with_num_16[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__16 = paddle._C_ops.sigmoid(slice_123)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_124 = split_with_num_16[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__16 = paddle._C_ops.multiply(slice_124, sigmoid__16)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_69 = paddle.matmul(multiply__16, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__53 = paddle._C_ops.add(matmul_69, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__34 = paddle._C_ops.softmax(add__53, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_16 = paddle._C_ops.argmax(softmax__34, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_18 = paddle._C_ops.cast(argmax_16, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__16 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__15, cast_18, constant_82, constant_86, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_18 = paddle._C_ops.embedding(set_value_with_tensor__16, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_57 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_125 = paddle._C_ops.slice(shape_57, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_58 = paddle._C_ops.shape(embedding_18)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_126 = paddle._C_ops.slice(shape_58, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_185 = [parameter_159, slice_126, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_55 = paddle._C_ops.memcpy_h2d(slice_126, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_186 = [parameter_159, memcpy_h2d_55, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_37 = paddle._C_ops.stack(combine_186, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_37 = paddle._C_ops.full_with_tensor(full_1, stack_37, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_187 = [parameter_159, slice_126, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_56 = paddle._C_ops.memcpy_h2d(slice_126, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_188 = [parameter_159, memcpy_h2d_56, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_38 = paddle._C_ops.stack(combine_188, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_38 = paddle._C_ops.full_with_tensor(full_1, stack_38, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_111 = paddle._C_ops.transpose(embedding_18, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_189 = [full_with_tensor_37, full_with_tensor_38]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__72, rnn__73, rnn__74, rnn__75 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_111, combine_189, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_112 = paddle._C_ops.transpose(rnn__72, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_113 = paddle._C_ops.transpose(transpose_112, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_190 = [slice_125, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_190), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_191 = [slice_125, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_72, reshape_73 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_191), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_114 = paddle._C_ops.transpose(transpose_113, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_70 = paddle.matmul(transpose_114, reshape_70, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__19 = paddle._C_ops.scale(matmul_70, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_59 = paddle._C_ops.shape(scale__19)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_127 = paddle._C_ops.slice(shape_59, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_192 = [slice_127, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__19, combine_192), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_38 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_57 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_18 = full_38 < memcpy_h2d_57

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_39 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_54 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_55 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_56 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32) <- (xb, xi32, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11137 = 0
        while less_than_18:
            less_than_18, assign_value_55, full_38, full_39, reshape__78, assign_value_56, assign_value_54, = self.pd_op_while_11137_0_0(full_0, arange_0, feed_2, constant_87, parameter_242, constant_88, constant_89, parameter_243, parameter_244, cast_1, less_than_18, assign_value_55, full_38, full_39, reshape__78, assign_value_56, assign_value_54)
            while_loop_counter_11137 += 1
            if while_loop_counter_11137 > kWhileLoopLimit:
                break
            
        while_108, while_109, while_110, while_111, while_112, while_113, = assign_value_55, full_38, full_39, reshape__78, assign_value_56, assign_value_54,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_193 = [slice_127, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_110, combine_193), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__35 = paddle._C_ops.softmax(reshape__80, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_115 = paddle._C_ops.transpose(reshape_72, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_71 = paddle.matmul(softmax__35, transpose_115, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_116 = paddle._C_ops.transpose(matmul_71, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_128 = paddle._C_ops.slice(transpose_116, [2], constant_82, constant_86, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_129 = paddle._C_ops.slice(transpose_8, [1], constant_82, constant_86, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_194 = [slice_128, slice_129]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_194, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_72 = paddle.matmul(concat_17, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__54 = paddle._C_ops.add(matmul_72, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_17 = paddle._C_ops.split_with_num(add__54, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_130 = split_with_num_17[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__17 = paddle._C_ops.sigmoid(slice_130)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_131 = split_with_num_17[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__17 = paddle._C_ops.multiply(slice_131, sigmoid__17)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_73 = paddle.matmul(multiply__17, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__55 = paddle._C_ops.add(matmul_73, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__36 = paddle._C_ops.softmax(add__55, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_17 = paddle._C_ops.argmax(softmax__36, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_19 = paddle._C_ops.cast(argmax_17, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__17 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__16, cast_19, constant_86, constant_90, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_19 = paddle._C_ops.embedding(set_value_with_tensor__17, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_60 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_132 = paddle._C_ops.slice(shape_60, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_61 = paddle._C_ops.shape(embedding_19)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_133 = paddle._C_ops.slice(shape_61, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_195 = [parameter_159, slice_133, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_58 = paddle._C_ops.memcpy_h2d(slice_133, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_196 = [parameter_159, memcpy_h2d_58, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_39 = paddle._C_ops.stack(combine_196, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_39 = paddle._C_ops.full_with_tensor(full_1, stack_39, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_197 = [parameter_159, slice_133, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_59 = paddle._C_ops.memcpy_h2d(slice_133, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_198 = [parameter_159, memcpy_h2d_59, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_40 = paddle._C_ops.stack(combine_198, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_40 = paddle._C_ops.full_with_tensor(full_1, stack_40, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_117 = paddle._C_ops.transpose(embedding_19, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_199 = [full_with_tensor_39, full_with_tensor_40]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__76, rnn__77, rnn__78, rnn__79 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_117, combine_199, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_118 = paddle._C_ops.transpose(rnn__76, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_119 = paddle._C_ops.transpose(transpose_118, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_200 = [slice_132, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_74, reshape_75 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_200), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_201 = [slice_132, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_76, reshape_77 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_201), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_120 = paddle._C_ops.transpose(transpose_119, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_74 = paddle.matmul(transpose_120, reshape_74, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__20 = paddle._C_ops.scale(matmul_74, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_62 = paddle._C_ops.shape(scale__20)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(shape_62, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_202 = [slice_134, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__20, combine_202), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_40 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_60 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_19 = full_40 < memcpy_h2d_60

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_41 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_57 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_58 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_59 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, -1x40x6x40xf32, xi64, xi64, xf32, xi32) <- (xb, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xi64, xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11232 = 0
        while less_than_19:
            less_than_19, reshape__82, full_41, full_40, assign_value_59, assign_value_57, assign_value_58, = self.pd_op_while_11232_0_0(full_0, arange_0, feed_2, constant_91, parameter_245, constant_92, constant_93, parameter_246, parameter_247, cast_1, less_than_19, reshape__82, full_41, full_40, assign_value_59, assign_value_57, assign_value_58)
            while_loop_counter_11232 += 1
            if while_loop_counter_11232 > kWhileLoopLimit:
                break
            
        while_114, while_115, while_116, while_117, while_118, while_119, = reshape__82, full_41, full_40, assign_value_59, assign_value_57, assign_value_58,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_203 = [slice_134, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_115, combine_203), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__37 = paddle._C_ops.softmax(reshape__84, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_121 = paddle._C_ops.transpose(reshape_76, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_75 = paddle.matmul(softmax__37, transpose_121, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_122 = paddle._C_ops.transpose(matmul_75, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_135 = paddle._C_ops.slice(transpose_122, [2], constant_86, constant_90, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_136 = paddle._C_ops.slice(transpose_8, [1], constant_86, constant_90, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_204 = [slice_135, slice_136]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_204, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_76 = paddle.matmul(concat_18, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__56 = paddle._C_ops.add(matmul_76, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_18 = paddle._C_ops.split_with_num(add__56, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_137 = split_with_num_18[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__18 = paddle._C_ops.sigmoid(slice_137)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_138 = split_with_num_18[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__18 = paddle._C_ops.multiply(slice_138, sigmoid__18)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_77 = paddle.matmul(multiply__18, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__57 = paddle._C_ops.add(matmul_77, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__38 = paddle._C_ops.softmax(add__57, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_18 = paddle._C_ops.argmax(softmax__38, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_20 = paddle._C_ops.cast(argmax_18, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__18 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__17, cast_20, constant_90, constant_94, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_20 = paddle._C_ops.embedding(set_value_with_tensor__18, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_63 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_139 = paddle._C_ops.slice(shape_63, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_64 = paddle._C_ops.shape(embedding_20)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_140 = paddle._C_ops.slice(shape_64, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_205 = [parameter_159, slice_140, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_61 = paddle._C_ops.memcpy_h2d(slice_140, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_206 = [parameter_159, memcpy_h2d_61, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_41 = paddle._C_ops.stack(combine_206, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_41 = paddle._C_ops.full_with_tensor(full_1, stack_41, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_207 = [parameter_159, slice_140, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_62 = paddle._C_ops.memcpy_h2d(slice_140, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_208 = [parameter_159, memcpy_h2d_62, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_42 = paddle._C_ops.stack(combine_208, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_42 = paddle._C_ops.full_with_tensor(full_1, stack_42, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_123 = paddle._C_ops.transpose(embedding_20, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_209 = [full_with_tensor_41, full_with_tensor_42]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__80, rnn__81, rnn__82, rnn__83 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_123, combine_209, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_124 = paddle._C_ops.transpose(rnn__80, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_125 = paddle._C_ops.transpose(transpose_124, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_210 = [slice_139, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_78, reshape_79 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_210), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_211 = [slice_139, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_80, reshape_81 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_211), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_126 = paddle._C_ops.transpose(transpose_125, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_78 = paddle.matmul(transpose_126, reshape_78, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__21 = paddle._C_ops.scale(matmul_78, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_65 = paddle._C_ops.shape(scale__21)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_141 = paddle._C_ops.slice(shape_65, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_212 = [slice_141, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__21, combine_212), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_42 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_63 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_20 = full_42 < memcpy_h2d_63

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_43 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_60 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_61 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_62 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, xf32, -1x40x6x40xf32, xi64, -1x40x6x40xf32) <- (xb, xi32, xi64, xf32, -1x40x6x40xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11327 = 0
        while less_than_20:
            less_than_20, assign_value_61, assign_value_62, assign_value_60, reshape__86, full_42, full_43, = self.pd_op_while_11327_0_0(full_0, arange_0, feed_2, constant_95, parameter_248, constant_96, constant_97, parameter_249, parameter_250, cast_1, less_than_20, assign_value_61, assign_value_62, assign_value_60, reshape__86, full_42, full_43)
            while_loop_counter_11327 += 1
            if while_loop_counter_11327 > kWhileLoopLimit:
                break
            
        while_120, while_121, while_122, while_123, while_124, while_125, = assign_value_61, assign_value_62, assign_value_60, reshape__86, full_42, full_43,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_213 = [slice_141, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_125, combine_213), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__39 = paddle._C_ops.softmax(reshape__88, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_127 = paddle._C_ops.transpose(reshape_80, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_79 = paddle.matmul(softmax__39, transpose_127, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_128 = paddle._C_ops.transpose(matmul_79, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_142 = paddle._C_ops.slice(transpose_128, [2], constant_90, constant_94, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_143 = paddle._C_ops.slice(transpose_8, [1], constant_90, constant_94, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_214 = [slice_142, slice_143]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_214, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_80 = paddle.matmul(concat_19, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__58 = paddle._C_ops.add(matmul_80, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_19 = paddle._C_ops.split_with_num(add__58, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_144 = split_with_num_19[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__19 = paddle._C_ops.sigmoid(slice_144)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_145 = split_with_num_19[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__19 = paddle._C_ops.multiply(slice_145, sigmoid__19)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_81 = paddle.matmul(multiply__19, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__59 = paddle._C_ops.add(matmul_81, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__40 = paddle._C_ops.softmax(add__59, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_19 = paddle._C_ops.argmax(softmax__40, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_21 = paddle._C_ops.cast(argmax_19, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__19 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__18, cast_21, constant_94, constant_98, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_21 = paddle._C_ops.embedding(set_value_with_tensor__19, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_66 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_146 = paddle._C_ops.slice(shape_66, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_67 = paddle._C_ops.shape(embedding_21)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_147 = paddle._C_ops.slice(shape_67, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_215 = [parameter_159, slice_147, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_64 = paddle._C_ops.memcpy_h2d(slice_147, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_216 = [parameter_159, memcpy_h2d_64, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_43 = paddle._C_ops.stack(combine_216, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_43 = paddle._C_ops.full_with_tensor(full_1, stack_43, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_217 = [parameter_159, slice_147, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_65 = paddle._C_ops.memcpy_h2d(slice_147, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_218 = [parameter_159, memcpy_h2d_65, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_44 = paddle._C_ops.stack(combine_218, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_44 = paddle._C_ops.full_with_tensor(full_1, stack_44, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_129 = paddle._C_ops.transpose(embedding_21, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_219 = [full_with_tensor_43, full_with_tensor_44]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__84, rnn__85, rnn__86, rnn__87 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_129, combine_219, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_130 = paddle._C_ops.transpose(rnn__84, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_131 = paddle._C_ops.transpose(transpose_130, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_220 = [slice_146, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_82, reshape_83 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_220), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_221 = [slice_146, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_84, reshape_85 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_221), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_132 = paddle._C_ops.transpose(transpose_131, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_82 = paddle.matmul(transpose_132, reshape_82, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__22 = paddle._C_ops.scale(matmul_82, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_68 = paddle._C_ops.shape(scale__22)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_148 = paddle._C_ops.slice(shape_68, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_222 = [slice_148, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__22, combine_222), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_44 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_66 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_21 = full_44 < memcpy_h2d_66

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_45 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_63 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_64 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_65 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32) <- (xb, xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11422 = 0
        while less_than_21:
            less_than_21, full_44, assign_value_64, full_45, reshape__90, assign_value_65, assign_value_63, = self.pd_op_while_11422_0_0(full_0, arange_0, feed_2, constant_99, parameter_251, constant_100, constant_101, parameter_252, parameter_253, cast_1, less_than_21, full_44, assign_value_64, full_45, reshape__90, assign_value_65, assign_value_63)
            while_loop_counter_11422 += 1
            if while_loop_counter_11422 > kWhileLoopLimit:
                break
            
        while_126, while_127, while_128, while_129, while_130, while_131, = full_44, assign_value_64, full_45, reshape__90, assign_value_65, assign_value_63,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_223 = [slice_148, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_128, combine_223), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__41 = paddle._C_ops.softmax(reshape__92, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_133 = paddle._C_ops.transpose(reshape_84, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_83 = paddle.matmul(softmax__41, transpose_133, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_134 = paddle._C_ops.transpose(matmul_83, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_149 = paddle._C_ops.slice(transpose_134, [2], constant_94, constant_98, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_150 = paddle._C_ops.slice(transpose_8, [1], constant_94, constant_98, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_224 = [slice_149, slice_150]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_224, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_84 = paddle.matmul(concat_20, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__60 = paddle._C_ops.add(matmul_84, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_20 = paddle._C_ops.split_with_num(add__60, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_151 = split_with_num_20[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__20 = paddle._C_ops.sigmoid(slice_151)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_152 = split_with_num_20[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__20 = paddle._C_ops.multiply(slice_152, sigmoid__20)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_85 = paddle.matmul(multiply__20, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__61 = paddle._C_ops.add(matmul_85, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__42 = paddle._C_ops.softmax(add__61, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_20 = paddle._C_ops.argmax(softmax__42, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_22 = paddle._C_ops.cast(argmax_20, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__20 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__19, cast_22, constant_98, constant_102, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_22 = paddle._C_ops.embedding(set_value_with_tensor__20, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_69 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_153 = paddle._C_ops.slice(shape_69, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_70 = paddle._C_ops.shape(embedding_22)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_154 = paddle._C_ops.slice(shape_70, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_225 = [parameter_159, slice_154, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_67 = paddle._C_ops.memcpy_h2d(slice_154, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_226 = [parameter_159, memcpy_h2d_67, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_45 = paddle._C_ops.stack(combine_226, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_45 = paddle._C_ops.full_with_tensor(full_1, stack_45, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_227 = [parameter_159, slice_154, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_68 = paddle._C_ops.memcpy_h2d(slice_154, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_228 = [parameter_159, memcpy_h2d_68, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_46 = paddle._C_ops.stack(combine_228, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_46 = paddle._C_ops.full_with_tensor(full_1, stack_46, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_135 = paddle._C_ops.transpose(embedding_22, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_229 = [full_with_tensor_45, full_with_tensor_46]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__88, rnn__89, rnn__90, rnn__91 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_135, combine_229, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_136 = paddle._C_ops.transpose(rnn__88, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_137 = paddle._C_ops.transpose(transpose_136, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_230 = [slice_153, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_86, reshape_87 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_230), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_231 = [slice_153, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_88, reshape_89 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_231), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_138 = paddle._C_ops.transpose(transpose_137, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_86 = paddle.matmul(transpose_138, reshape_86, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__23 = paddle._C_ops.scale(matmul_86, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_71 = paddle._C_ops.shape(scale__23)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_155 = paddle._C_ops.slice(shape_71, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_232 = [slice_155, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__23, combine_232), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_46 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_69 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_22 = full_46 < memcpy_h2d_69

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_47 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_66 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_67 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_68 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64) <- (xb, xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11517 = 0
        while less_than_22:
            less_than_22, assign_value_68, assign_value_67, reshape__94, full_47, assign_value_66, full_46, = self.pd_op_while_11517_0_0(full_0, arange_0, feed_2, constant_103, parameter_254, constant_104, constant_105, parameter_255, parameter_256, cast_1, less_than_22, assign_value_68, assign_value_67, reshape__94, full_47, assign_value_66, full_46)
            while_loop_counter_11517 += 1
            if while_loop_counter_11517 > kWhileLoopLimit:
                break
            
        while_132, while_133, while_134, while_135, while_136, while_137, = assign_value_68, assign_value_67, reshape__94, full_47, assign_value_66, full_46,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_233 = [slice_155, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_135, combine_233), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__43 = paddle._C_ops.softmax(reshape__96, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_139 = paddle._C_ops.transpose(reshape_88, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_87 = paddle.matmul(softmax__43, transpose_139, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_140 = paddle._C_ops.transpose(matmul_87, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_156 = paddle._C_ops.slice(transpose_140, [2], constant_98, constant_102, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_157 = paddle._C_ops.slice(transpose_8, [1], constant_98, constant_102, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_234 = [slice_156, slice_157]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_234, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_88 = paddle.matmul(concat_21, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__62 = paddle._C_ops.add(matmul_88, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_21 = paddle._C_ops.split_with_num(add__62, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_158 = split_with_num_21[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__21 = paddle._C_ops.sigmoid(slice_158)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_159 = split_with_num_21[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__21 = paddle._C_ops.multiply(slice_159, sigmoid__21)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_89 = paddle.matmul(multiply__21, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__63 = paddle._C_ops.add(matmul_89, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__44 = paddle._C_ops.softmax(add__63, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_21 = paddle._C_ops.argmax(softmax__44, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_23 = paddle._C_ops.cast(argmax_21, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__21 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__20, cast_23, constant_102, constant_106, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_23 = paddle._C_ops.embedding(set_value_with_tensor__21, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_72 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_160 = paddle._C_ops.slice(shape_72, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_73 = paddle._C_ops.shape(embedding_23)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_161 = paddle._C_ops.slice(shape_73, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_235 = [parameter_159, slice_161, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_70 = paddle._C_ops.memcpy_h2d(slice_161, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_236 = [parameter_159, memcpy_h2d_70, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_47 = paddle._C_ops.stack(combine_236, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_47 = paddle._C_ops.full_with_tensor(full_1, stack_47, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_237 = [parameter_159, slice_161, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_71 = paddle._C_ops.memcpy_h2d(slice_161, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_238 = [parameter_159, memcpy_h2d_71, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_48 = paddle._C_ops.stack(combine_238, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_48 = paddle._C_ops.full_with_tensor(full_1, stack_48, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_141 = paddle._C_ops.transpose(embedding_23, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_239 = [full_with_tensor_47, full_with_tensor_48]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__92, rnn__93, rnn__94, rnn__95 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_141, combine_239, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_142 = paddle._C_ops.transpose(rnn__92, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_143 = paddle._C_ops.transpose(transpose_142, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_240 = [slice_160, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_90, reshape_91 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_240), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_241 = [slice_160, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_92, reshape_93 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_241), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_144 = paddle._C_ops.transpose(transpose_143, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_90 = paddle.matmul(transpose_144, reshape_90, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__24 = paddle._C_ops.scale(matmul_90, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_74 = paddle._C_ops.shape(scale__24)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_162 = paddle._C_ops.slice(shape_74, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_242 = [slice_162, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__24, combine_242), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_48 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_72 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_23 = full_48 < memcpy_h2d_72

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_49 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_69 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_70 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_71 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32) <- (xb, xi64, xi32, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11612 = 0
        while less_than_23:
            less_than_23, full_48, assign_value_70, reshape__98, assign_value_69, assign_value_71, full_49, = self.pd_op_while_11612_0_0(full_0, arange_0, feed_2, constant_107, parameter_257, constant_108, constant_109, parameter_258, parameter_259, cast_1, less_than_23, full_48, assign_value_70, reshape__98, assign_value_69, assign_value_71, full_49)
            while_loop_counter_11612 += 1
            if while_loop_counter_11612 > kWhileLoopLimit:
                break
            
        while_138, while_139, while_140, while_141, while_142, while_143, = full_48, assign_value_70, reshape__98, assign_value_69, assign_value_71, full_49,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_243 = [slice_162, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_143, combine_243), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__45 = paddle._C_ops.softmax(reshape__100, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_145 = paddle._C_ops.transpose(reshape_92, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_91 = paddle.matmul(softmax__45, transpose_145, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_146 = paddle._C_ops.transpose(matmul_91, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_163 = paddle._C_ops.slice(transpose_146, [2], constant_102, constant_106, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_164 = paddle._C_ops.slice(transpose_8, [1], constant_102, constant_106, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_244 = [slice_163, slice_164]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_244, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_92 = paddle.matmul(concat_22, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__64 = paddle._C_ops.add(matmul_92, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_22 = paddle._C_ops.split_with_num(add__64, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_165 = split_with_num_22[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__22 = paddle._C_ops.sigmoid(slice_165)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_166 = split_with_num_22[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__22 = paddle._C_ops.multiply(slice_166, sigmoid__22)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_93 = paddle.matmul(multiply__22, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__65 = paddle._C_ops.add(matmul_93, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__46 = paddle._C_ops.softmax(add__65, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_22 = paddle._C_ops.argmax(softmax__46, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_24 = paddle._C_ops.cast(argmax_22, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__22 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__21, cast_24, constant_106, constant_110, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_24 = paddle._C_ops.embedding(set_value_with_tensor__22, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_75 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_167 = paddle._C_ops.slice(shape_75, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_76 = paddle._C_ops.shape(embedding_24)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_168 = paddle._C_ops.slice(shape_76, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_245 = [parameter_159, slice_168, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_73 = paddle._C_ops.memcpy_h2d(slice_168, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_246 = [parameter_159, memcpy_h2d_73, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_49 = paddle._C_ops.stack(combine_246, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_49 = paddle._C_ops.full_with_tensor(full_1, stack_49, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_247 = [parameter_159, slice_168, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_74 = paddle._C_ops.memcpy_h2d(slice_168, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_248 = [parameter_159, memcpy_h2d_74, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_50 = paddle._C_ops.stack(combine_248, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_50 = paddle._C_ops.full_with_tensor(full_1, stack_50, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_147 = paddle._C_ops.transpose(embedding_24, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_249 = [full_with_tensor_49, full_with_tensor_50]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__96, rnn__97, rnn__98, rnn__99 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_147, combine_249, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_148 = paddle._C_ops.transpose(rnn__96, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_149 = paddle._C_ops.transpose(transpose_148, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_250 = [slice_167, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_94, reshape_95 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_250), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_251 = [slice_167, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_96, reshape_97 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_251), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_150 = paddle._C_ops.transpose(transpose_149, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_94 = paddle.matmul(transpose_150, reshape_94, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__25 = paddle._C_ops.scale(matmul_94, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_77 = paddle._C_ops.shape(scale__25)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_169 = paddle._C_ops.slice(shape_77, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_252 = [slice_169, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__25, combine_252), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_50 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_75 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_24 = full_50 < memcpy_h2d_75

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_51 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_72 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_73 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_74 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xf32, -1x40x6x40xf32, xi32, xi64, xi64) <- (xb, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi32, xi64, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11707 = 0
        while less_than_24:
            less_than_24, full_51, assign_value_72, reshape__102, assign_value_73, assign_value_74, full_50, = self.pd_op_while_11707_0_0(full_0, arange_0, feed_2, constant_111, parameter_260, constant_112, constant_113, parameter_261, parameter_262, cast_1, less_than_24, full_51, assign_value_72, reshape__102, assign_value_73, assign_value_74, full_50)
            while_loop_counter_11707 += 1
            if while_loop_counter_11707 > kWhileLoopLimit:
                break
            
        while_144, while_145, while_146, while_147, while_148, while_149, = full_51, assign_value_72, reshape__102, assign_value_73, assign_value_74, full_50,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_253 = [slice_169, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_144, combine_253), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__47 = paddle._C_ops.softmax(reshape__104, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_151 = paddle._C_ops.transpose(reshape_96, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_95 = paddle.matmul(softmax__47, transpose_151, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_152 = paddle._C_ops.transpose(matmul_95, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_170 = paddle._C_ops.slice(transpose_152, [2], constant_106, constant_110, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_171 = paddle._C_ops.slice(transpose_8, [1], constant_106, constant_110, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_254 = [slice_170, slice_171]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_254, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_96 = paddle.matmul(concat_23, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__66 = paddle._C_ops.add(matmul_96, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_23 = paddle._C_ops.split_with_num(add__66, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_172 = split_with_num_23[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__23 = paddle._C_ops.sigmoid(slice_172)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_173 = split_with_num_23[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__23 = paddle._C_ops.multiply(slice_173, sigmoid__23)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_97 = paddle.matmul(multiply__23, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__67 = paddle._C_ops.add(matmul_97, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__48 = paddle._C_ops.softmax(add__67, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_23 = paddle._C_ops.argmax(softmax__48, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_25 = paddle._C_ops.cast(argmax_23, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__23 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__22, cast_25, constant_110, constant_114, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_25 = paddle._C_ops.embedding(set_value_with_tensor__23, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_78 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_174 = paddle._C_ops.slice(shape_78, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_79 = paddle._C_ops.shape(embedding_25)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_175 = paddle._C_ops.slice(shape_79, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_255 = [parameter_159, slice_175, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_76 = paddle._C_ops.memcpy_h2d(slice_175, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_256 = [parameter_159, memcpy_h2d_76, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_51 = paddle._C_ops.stack(combine_256, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_51 = paddle._C_ops.full_with_tensor(full_1, stack_51, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_257 = [parameter_159, slice_175, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_77 = paddle._C_ops.memcpy_h2d(slice_175, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_258 = [parameter_159, memcpy_h2d_77, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_52 = paddle._C_ops.stack(combine_258, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_52 = paddle._C_ops.full_with_tensor(full_1, stack_52, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_153 = paddle._C_ops.transpose(embedding_25, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_259 = [full_with_tensor_51, full_with_tensor_52]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__100, rnn__101, rnn__102, rnn__103 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_153, combine_259, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_154 = paddle._C_ops.transpose(rnn__100, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_155 = paddle._C_ops.transpose(transpose_154, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_260 = [slice_174, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_98, reshape_99 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_260), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_261 = [slice_174, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_100, reshape_101 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_261), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_156 = paddle._C_ops.transpose(transpose_155, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_98 = paddle.matmul(transpose_156, reshape_98, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__26 = paddle._C_ops.scale(matmul_98, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_80 = paddle._C_ops.shape(scale__26)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_176 = paddle._C_ops.slice(shape_80, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_262 = [slice_176, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__26, combine_262), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_52 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_78 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_25 = full_52 < memcpy_h2d_78

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_53 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_75 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_76 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_77 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, -1x40x6x40xf32, xf32, xi32, xi64) <- (xb, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xf32, xi32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11802 = 0
        while less_than_25:
            less_than_25, full_53, assign_value_77, reshape__106, assign_value_75, assign_value_76, full_52, = self.pd_op_while_11802_0_0(full_0, arange_0, feed_2, constant_115, parameter_263, constant_116, constant_117, parameter_264, parameter_265, cast_1, less_than_25, full_53, assign_value_77, reshape__106, assign_value_75, assign_value_76, full_52)
            while_loop_counter_11802 += 1
            if while_loop_counter_11802 > kWhileLoopLimit:
                break
            
        while_150, while_151, while_152, while_153, while_154, while_155, = full_53, assign_value_77, reshape__106, assign_value_75, assign_value_76, full_52,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_263 = [slice_176, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_150, combine_263), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__49 = paddle._C_ops.softmax(reshape__108, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_157 = paddle._C_ops.transpose(reshape_100, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_99 = paddle.matmul(softmax__49, transpose_157, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_158 = paddle._C_ops.transpose(matmul_99, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_177 = paddle._C_ops.slice(transpose_158, [2], constant_110, constant_114, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_178 = paddle._C_ops.slice(transpose_8, [1], constant_110, constant_114, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_264 = [slice_177, slice_178]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_264, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_100 = paddle.matmul(concat_24, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__68 = paddle._C_ops.add(matmul_100, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_24 = paddle._C_ops.split_with_num(add__68, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_179 = split_with_num_24[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__24 = paddle._C_ops.sigmoid(slice_179)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_180 = split_with_num_24[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__24 = paddle._C_ops.multiply(slice_180, sigmoid__24)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_101 = paddle.matmul(multiply__24, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__69 = paddle._C_ops.add(matmul_101, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__50 = paddle._C_ops.softmax(add__69, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_24 = paddle._C_ops.argmax(softmax__50, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_26 = paddle._C_ops.cast(argmax_24, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__24 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__23, cast_26, constant_114, constant_118, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_26 = paddle._C_ops.embedding(set_value_with_tensor__24, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_81 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_181 = paddle._C_ops.slice(shape_81, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_82 = paddle._C_ops.shape(embedding_26)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_182 = paddle._C_ops.slice(shape_82, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_265 = [parameter_159, slice_182, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_79 = paddle._C_ops.memcpy_h2d(slice_182, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_266 = [parameter_159, memcpy_h2d_79, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_53 = paddle._C_ops.stack(combine_266, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_53 = paddle._C_ops.full_with_tensor(full_1, stack_53, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_267 = [parameter_159, slice_182, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_80 = paddle._C_ops.memcpy_h2d(slice_182, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_268 = [parameter_159, memcpy_h2d_80, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_54 = paddle._C_ops.stack(combine_268, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_54 = paddle._C_ops.full_with_tensor(full_1, stack_54, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_159 = paddle._C_ops.transpose(embedding_26, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_269 = [full_with_tensor_53, full_with_tensor_54]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__104, rnn__105, rnn__106, rnn__107 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_159, combine_269, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_160 = paddle._C_ops.transpose(rnn__104, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_161 = paddle._C_ops.transpose(transpose_160, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_270 = [slice_181, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_102, reshape_103 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_270), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_271 = [slice_181, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_104, reshape_105 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_271), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_162 = paddle._C_ops.transpose(transpose_161, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_102 = paddle.matmul(transpose_162, reshape_102, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__27 = paddle._C_ops.scale(matmul_102, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_83 = paddle._C_ops.shape(scale__27)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_183 = paddle._C_ops.slice(shape_83, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_272 = [slice_183, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__27, combine_272), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_54 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_81 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_26 = full_54 < memcpy_h2d_81

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_55 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_78 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_79 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_80 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64) <- (xb, xi64, xi32, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11897 = 0
        while less_than_26:
            less_than_26, full_54, assign_value_79, assign_value_78, reshape__110, full_55, assign_value_80, = self.pd_op_while_11897_0_0(full_0, arange_0, feed_2, constant_119, parameter_266, constant_120, constant_121, parameter_267, parameter_268, cast_1, less_than_26, full_54, assign_value_79, assign_value_78, reshape__110, full_55, assign_value_80)
            while_loop_counter_11897 += 1
            if while_loop_counter_11897 > kWhileLoopLimit:
                break
            
        while_156, while_157, while_158, while_159, while_160, while_161, = full_54, assign_value_79, assign_value_78, reshape__110, full_55, assign_value_80,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_273 = [slice_183, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_160, combine_273), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__51 = paddle._C_ops.softmax(reshape__112, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_163 = paddle._C_ops.transpose(reshape_104, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_103 = paddle.matmul(softmax__51, transpose_163, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_164 = paddle._C_ops.transpose(matmul_103, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_184 = paddle._C_ops.slice(transpose_164, [2], constant_114, constant_118, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_185 = paddle._C_ops.slice(transpose_8, [1], constant_114, constant_118, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_274 = [slice_184, slice_185]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_274, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_104 = paddle.matmul(concat_25, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__70 = paddle._C_ops.add(matmul_104, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_25 = paddle._C_ops.split_with_num(add__70, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_186 = split_with_num_25[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__25 = paddle._C_ops.sigmoid(slice_186)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_187 = split_with_num_25[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__25 = paddle._C_ops.multiply(slice_187, sigmoid__25)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_105 = paddle.matmul(multiply__25, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__71 = paddle._C_ops.add(matmul_105, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__52 = paddle._C_ops.softmax(add__71, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_25 = paddle._C_ops.argmax(softmax__52, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_27 = paddle._C_ops.cast(argmax_25, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__25 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__24, cast_27, constant_118, constant_122, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_27 = paddle._C_ops.embedding(set_value_with_tensor__25, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_84 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_188 = paddle._C_ops.slice(shape_84, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_85 = paddle._C_ops.shape(embedding_27)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_189 = paddle._C_ops.slice(shape_85, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_275 = [parameter_159, slice_189, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_82 = paddle._C_ops.memcpy_h2d(slice_189, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_276 = [parameter_159, memcpy_h2d_82, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_55 = paddle._C_ops.stack(combine_276, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_55 = paddle._C_ops.full_with_tensor(full_1, stack_55, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_277 = [parameter_159, slice_189, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_83 = paddle._C_ops.memcpy_h2d(slice_189, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_278 = [parameter_159, memcpy_h2d_83, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_56 = paddle._C_ops.stack(combine_278, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_56 = paddle._C_ops.full_with_tensor(full_1, stack_56, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_165 = paddle._C_ops.transpose(embedding_27, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_279 = [full_with_tensor_55, full_with_tensor_56]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__108, rnn__109, rnn__110, rnn__111 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_165, combine_279, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_166 = paddle._C_ops.transpose(rnn__108, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_167 = paddle._C_ops.transpose(transpose_166, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_280 = [slice_188, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_106, reshape_107 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_280), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_281 = [slice_188, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_108, reshape_109 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_281), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_168 = paddle._C_ops.transpose(transpose_167, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_106 = paddle.matmul(transpose_168, reshape_106, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__28 = paddle._C_ops.scale(matmul_106, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_86 = paddle._C_ops.shape(scale__28)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_190 = paddle._C_ops.slice(shape_86, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_282 = [slice_190, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__28, combine_282), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_56 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_84 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_27 = full_56 < memcpy_h2d_84

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_57 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_81 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_82 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_83 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, -1x40x6x40xf32, xi64, xf32, xi32) <- (xb, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xi64, xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11992 = 0
        while less_than_27:
            less_than_27, full_57, full_56, reshape__114, assign_value_83, assign_value_81, assign_value_82, = self.pd_op_while_11992_0_0(full_0, arange_0, feed_2, constant_123, parameter_269, constant_124, constant_125, parameter_270, parameter_271, cast_1, less_than_27, full_57, full_56, reshape__114, assign_value_83, assign_value_81, assign_value_82)
            while_loop_counter_11992 += 1
            if while_loop_counter_11992 > kWhileLoopLimit:
                break
            
        while_162, while_163, while_164, while_165, while_166, while_167, = full_57, full_56, reshape__114, assign_value_83, assign_value_81, assign_value_82,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_283 = [slice_190, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_162, combine_283), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__53 = paddle._C_ops.softmax(reshape__116, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_169 = paddle._C_ops.transpose(reshape_108, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_107 = paddle.matmul(softmax__53, transpose_169, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_170 = paddle._C_ops.transpose(matmul_107, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_191 = paddle._C_ops.slice(transpose_170, [2], constant_118, constant_122, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_192 = paddle._C_ops.slice(transpose_8, [1], constant_118, constant_122, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_284 = [slice_191, slice_192]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_284, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_108 = paddle.matmul(concat_26, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__72 = paddle._C_ops.add(matmul_108, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_26 = paddle._C_ops.split_with_num(add__72, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_193 = split_with_num_26[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__26 = paddle._C_ops.sigmoid(slice_193)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_194 = split_with_num_26[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__26 = paddle._C_ops.multiply(slice_194, sigmoid__26)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_109 = paddle.matmul(multiply__26, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__73 = paddle._C_ops.add(matmul_109, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__54 = paddle._C_ops.softmax(add__73, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_26 = paddle._C_ops.argmax(softmax__54, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_28 = paddle._C_ops.cast(argmax_26, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__26 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__25, cast_28, constant_122, constant_126, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_28 = paddle._C_ops.embedding(set_value_with_tensor__26, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_87 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_195 = paddle._C_ops.slice(shape_87, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_88 = paddle._C_ops.shape(embedding_28)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_196 = paddle._C_ops.slice(shape_88, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_285 = [parameter_159, slice_196, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_85 = paddle._C_ops.memcpy_h2d(slice_196, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_286 = [parameter_159, memcpy_h2d_85, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_57 = paddle._C_ops.stack(combine_286, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_57 = paddle._C_ops.full_with_tensor(full_1, stack_57, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_287 = [parameter_159, slice_196, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_86 = paddle._C_ops.memcpy_h2d(slice_196, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_288 = [parameter_159, memcpy_h2d_86, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_58 = paddle._C_ops.stack(combine_288, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_58 = paddle._C_ops.full_with_tensor(full_1, stack_58, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_171 = paddle._C_ops.transpose(embedding_28, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_289 = [full_with_tensor_57, full_with_tensor_58]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__112, rnn__113, rnn__114, rnn__115 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_171, combine_289, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_172 = paddle._C_ops.transpose(rnn__112, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_173 = paddle._C_ops.transpose(transpose_172, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_290 = [slice_195, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_110, reshape_111 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_290), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_291 = [slice_195, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_112, reshape_113 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_291), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_174 = paddle._C_ops.transpose(transpose_173, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_110 = paddle.matmul(transpose_174, reshape_110, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__29 = paddle._C_ops.scale(matmul_110, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_89 = paddle._C_ops.shape(scale__29)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_197 = paddle._C_ops.slice(shape_89, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_292 = [slice_197, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__29, combine_292), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_58 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_87 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_28 = full_58 < memcpy_h2d_87

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_59 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_84 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_85 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_86 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, xi32, xi64, -1x40x6x40xf32) <- (xb, xi64, -1x40x6x40xf32, xf32, xi32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12087 = 0
        while less_than_28:
            less_than_28, assign_value_86, reshape__118, assign_value_84, assign_value_85, full_58, full_59, = self.pd_op_while_12087_0_0(full_0, arange_0, feed_2, constant_127, parameter_272, constant_128, constant_129, parameter_273, parameter_274, cast_1, less_than_28, assign_value_86, reshape__118, assign_value_84, assign_value_85, full_58, full_59)
            while_loop_counter_12087 += 1
            if while_loop_counter_12087 > kWhileLoopLimit:
                break
            
        while_168, while_169, while_170, while_171, while_172, while_173, = assign_value_86, reshape__118, assign_value_84, assign_value_85, full_58, full_59,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_293 = [slice_197, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_173, combine_293), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__55 = paddle._C_ops.softmax(reshape__120, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_175 = paddle._C_ops.transpose(reshape_112, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_111 = paddle.matmul(softmax__55, transpose_175, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_176 = paddle._C_ops.transpose(matmul_111, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_198 = paddle._C_ops.slice(transpose_176, [2], constant_122, constant_126, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_199 = paddle._C_ops.slice(transpose_8, [1], constant_122, constant_126, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_294 = [slice_198, slice_199]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_294, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_112 = paddle.matmul(concat_27, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__74 = paddle._C_ops.add(matmul_112, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_27 = paddle._C_ops.split_with_num(add__74, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_200 = split_with_num_27[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__27 = paddle._C_ops.sigmoid(slice_200)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_201 = split_with_num_27[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__27 = paddle._C_ops.multiply(slice_201, sigmoid__27)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_113 = paddle.matmul(multiply__27, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__75 = paddle._C_ops.add(matmul_113, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__56 = paddle._C_ops.softmax(add__75, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_27 = paddle._C_ops.argmax(softmax__56, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_29 = paddle._C_ops.cast(argmax_27, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__27 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__26, cast_29, constant_126, constant_130, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_29 = paddle._C_ops.embedding(set_value_with_tensor__27, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_90 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_202 = paddle._C_ops.slice(shape_90, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_91 = paddle._C_ops.shape(embedding_29)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_203 = paddle._C_ops.slice(shape_91, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_295 = [parameter_159, slice_203, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_88 = paddle._C_ops.memcpy_h2d(slice_203, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_296 = [parameter_159, memcpy_h2d_88, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_59 = paddle._C_ops.stack(combine_296, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_59 = paddle._C_ops.full_with_tensor(full_1, stack_59, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_297 = [parameter_159, slice_203, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_89 = paddle._C_ops.memcpy_h2d(slice_203, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_298 = [parameter_159, memcpy_h2d_89, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_60 = paddle._C_ops.stack(combine_298, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_60 = paddle._C_ops.full_with_tensor(full_1, stack_60, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_177 = paddle._C_ops.transpose(embedding_29, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_299 = [full_with_tensor_59, full_with_tensor_60]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__116, rnn__117, rnn__118, rnn__119 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_177, combine_299, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_178 = paddle._C_ops.transpose(rnn__116, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_179 = paddle._C_ops.transpose(transpose_178, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_300 = [slice_202, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_114, reshape_115 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_300), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_301 = [slice_202, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_116, reshape_117 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_301), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_180 = paddle._C_ops.transpose(transpose_179, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_114 = paddle.matmul(transpose_180, reshape_114, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__30 = paddle._C_ops.scale(matmul_114, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_92 = paddle._C_ops.shape(scale__30)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_204 = paddle._C_ops.slice(shape_92, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_302 = [slice_204, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__30, combine_302), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_60 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_90 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_29 = full_60 < memcpy_h2d_90

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_61 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_87 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_88 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_89 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xf32, xi64, -1x40x6x40xf32, xi32, -1x40x6x40xf32, xi64) <- (xb, xf32, xi64, -1x40x6x40xf32, xi32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12182 = 0
        while less_than_29:
            less_than_29, assign_value_87, assign_value_89, full_61, assign_value_88, reshape__122, full_60, = self.pd_op_while_12182_0_0(full_0, arange_0, feed_2, constant_131, parameter_275, constant_132, constant_133, parameter_276, parameter_277, cast_1, less_than_29, assign_value_87, assign_value_89, full_61, assign_value_88, reshape__122, full_60)
            while_loop_counter_12182 += 1
            if while_loop_counter_12182 > kWhileLoopLimit:
                break
            
        while_174, while_175, while_176, while_177, while_178, while_179, = assign_value_87, assign_value_89, full_61, assign_value_88, reshape__122, full_60,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_303 = [slice_204, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_176, combine_303), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__57 = paddle._C_ops.softmax(reshape__124, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_181 = paddle._C_ops.transpose(reshape_116, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_115 = paddle.matmul(softmax__57, transpose_181, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_182 = paddle._C_ops.transpose(matmul_115, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_205 = paddle._C_ops.slice(transpose_182, [2], constant_126, constant_130, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_206 = paddle._C_ops.slice(transpose_8, [1], constant_126, constant_130, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_304 = [slice_205, slice_206]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_304, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_116 = paddle.matmul(concat_28, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__76 = paddle._C_ops.add(matmul_116, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_28 = paddle._C_ops.split_with_num(add__76, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_207 = split_with_num_28[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__28 = paddle._C_ops.sigmoid(slice_207)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_208 = split_with_num_28[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__28 = paddle._C_ops.multiply(slice_208, sigmoid__28)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_117 = paddle.matmul(multiply__28, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__77 = paddle._C_ops.add(matmul_117, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__58 = paddle._C_ops.softmax(add__77, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_28 = paddle._C_ops.argmax(softmax__58, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_30 = paddle._C_ops.cast(argmax_28, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__28 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__27, cast_30, constant_130, constant_134, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_30 = paddle._C_ops.embedding(set_value_with_tensor__28, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_93 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_209 = paddle._C_ops.slice(shape_93, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_94 = paddle._C_ops.shape(embedding_30)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_210 = paddle._C_ops.slice(shape_94, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_305 = [parameter_159, slice_210, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_91 = paddle._C_ops.memcpy_h2d(slice_210, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_306 = [parameter_159, memcpy_h2d_91, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_61 = paddle._C_ops.stack(combine_306, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_61 = paddle._C_ops.full_with_tensor(full_1, stack_61, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_307 = [parameter_159, slice_210, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_92 = paddle._C_ops.memcpy_h2d(slice_210, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_308 = [parameter_159, memcpy_h2d_92, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_62 = paddle._C_ops.stack(combine_308, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_62 = paddle._C_ops.full_with_tensor(full_1, stack_62, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_183 = paddle._C_ops.transpose(embedding_30, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_309 = [full_with_tensor_61, full_with_tensor_62]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__120, rnn__121, rnn__122, rnn__123 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_183, combine_309, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_184 = paddle._C_ops.transpose(rnn__120, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_185 = paddle._C_ops.transpose(transpose_184, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_310 = [slice_209, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_118, reshape_119 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_310), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_311 = [slice_209, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_120, reshape_121 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_311), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_186 = paddle._C_ops.transpose(transpose_185, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_118 = paddle.matmul(transpose_186, reshape_118, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__31 = paddle._C_ops.scale(matmul_118, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_95 = paddle._C_ops.shape(scale__31)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_211 = paddle._C_ops.slice(shape_95, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_312 = [slice_211, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__31, combine_312), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_62 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_93 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_30 = full_62 < memcpy_h2d_93

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_63 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_90 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_91 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_92 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, -1x40x6x40xf32, xi64, xi64, xf32, -1x40x6x40xf32) <- (xb, xi32, -1x40x6x40xf32, xi64, xi64, xf32, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12277 = 0
        while less_than_30:
            less_than_30, assign_value_91, full_63, assign_value_92, full_62, assign_value_90, reshape__126, = self.pd_op_while_12277_0_0(full_0, arange_0, feed_2, constant_135, parameter_278, constant_136, constant_137, parameter_279, parameter_280, cast_1, less_than_30, assign_value_91, full_63, assign_value_92, full_62, assign_value_90, reshape__126)
            while_loop_counter_12277 += 1
            if while_loop_counter_12277 > kWhileLoopLimit:
                break
            
        while_180, while_181, while_182, while_183, while_184, while_185, = assign_value_91, full_63, assign_value_92, full_62, assign_value_90, reshape__126,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_313 = [slice_211, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_181, combine_313), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__59 = paddle._C_ops.softmax(reshape__128, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_187 = paddle._C_ops.transpose(reshape_120, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_119 = paddle.matmul(softmax__59, transpose_187, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_188 = paddle._C_ops.transpose(matmul_119, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_212 = paddle._C_ops.slice(transpose_188, [2], constant_130, constant_134, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_213 = paddle._C_ops.slice(transpose_8, [1], constant_130, constant_134, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_314 = [slice_212, slice_213]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_314, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_120 = paddle.matmul(concat_29, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__78 = paddle._C_ops.add(matmul_120, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_29 = paddle._C_ops.split_with_num(add__78, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_214 = split_with_num_29[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__29 = paddle._C_ops.sigmoid(slice_214)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_215 = split_with_num_29[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__29 = paddle._C_ops.multiply(slice_215, sigmoid__29)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_121 = paddle.matmul(multiply__29, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__79 = paddle._C_ops.add(matmul_121, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__60 = paddle._C_ops.softmax(add__79, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_29 = paddle._C_ops.argmax(softmax__60, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_31 = paddle._C_ops.cast(argmax_29, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__29 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__28, cast_31, constant_134, constant_138, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_31 = paddle._C_ops.embedding(set_value_with_tensor__29, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_96 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_216 = paddle._C_ops.slice(shape_96, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_97 = paddle._C_ops.shape(embedding_31)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_217 = paddle._C_ops.slice(shape_97, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_315 = [parameter_159, slice_217, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_94 = paddle._C_ops.memcpy_h2d(slice_217, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_316 = [parameter_159, memcpy_h2d_94, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_63 = paddle._C_ops.stack(combine_316, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_63 = paddle._C_ops.full_with_tensor(full_1, stack_63, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_317 = [parameter_159, slice_217, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_95 = paddle._C_ops.memcpy_h2d(slice_217, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_318 = [parameter_159, memcpy_h2d_95, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_64 = paddle._C_ops.stack(combine_318, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_64 = paddle._C_ops.full_with_tensor(full_1, stack_64, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_189 = paddle._C_ops.transpose(embedding_31, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_319 = [full_with_tensor_63, full_with_tensor_64]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__124, rnn__125, rnn__126, rnn__127 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_189, combine_319, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_190 = paddle._C_ops.transpose(rnn__124, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_191 = paddle._C_ops.transpose(transpose_190, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_320 = [slice_216, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_122, reshape_123 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_320), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_321 = [slice_216, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_124, reshape_125 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_321), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_192 = paddle._C_ops.transpose(transpose_191, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_122 = paddle.matmul(transpose_192, reshape_122, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__32 = paddle._C_ops.scale(matmul_122, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_98 = paddle._C_ops.shape(scale__32)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_218 = paddle._C_ops.slice(shape_98, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_322 = [slice_218, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__32, combine_322), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_64 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_96 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_31 = full_64 < memcpy_h2d_96

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_65 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_93 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_94 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_95 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xf32, xi64) <- (xb, xi32, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12372 = 0
        while less_than_31:
            less_than_31, assign_value_94, full_65, assign_value_95, reshape__130, assign_value_93, full_64, = self.pd_op_while_12372_0_0(full_0, arange_0, feed_2, constant_139, parameter_281, constant_140, constant_141, parameter_282, parameter_283, cast_1, less_than_31, assign_value_94, full_65, assign_value_95, reshape__130, assign_value_93, full_64)
            while_loop_counter_12372 += 1
            if while_loop_counter_12372 > kWhileLoopLimit:
                break
            
        while_186, while_187, while_188, while_189, while_190, while_191, = assign_value_94, full_65, assign_value_95, reshape__130, assign_value_93, full_64,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_323 = [slice_218, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_187, combine_323), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__61 = paddle._C_ops.softmax(reshape__132, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_193 = paddle._C_ops.transpose(reshape_124, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_123 = paddle.matmul(softmax__61, transpose_193, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_194 = paddle._C_ops.transpose(matmul_123, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_219 = paddle._C_ops.slice(transpose_194, [2], constant_134, constant_138, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_220 = paddle._C_ops.slice(transpose_8, [1], constant_134, constant_138, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_324 = [slice_219, slice_220]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_324, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_124 = paddle.matmul(concat_30, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__80 = paddle._C_ops.add(matmul_124, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_30 = paddle._C_ops.split_with_num(add__80, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_221 = split_with_num_30[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__30 = paddle._C_ops.sigmoid(slice_221)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_222 = split_with_num_30[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__30 = paddle._C_ops.multiply(slice_222, sigmoid__30)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_125 = paddle.matmul(multiply__30, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__81 = paddle._C_ops.add(matmul_125, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__62 = paddle._C_ops.softmax(add__81, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_30 = paddle._C_ops.argmax(softmax__62, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_32 = paddle._C_ops.cast(argmax_30, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__30 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__29, cast_32, constant_138, constant_142, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_32 = paddle._C_ops.embedding(set_value_with_tensor__30, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_99 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_223 = paddle._C_ops.slice(shape_99, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_100 = paddle._C_ops.shape(embedding_32)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_224 = paddle._C_ops.slice(shape_100, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_325 = [parameter_159, slice_224, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_97 = paddle._C_ops.memcpy_h2d(slice_224, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_326 = [parameter_159, memcpy_h2d_97, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_65 = paddle._C_ops.stack(combine_326, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_65 = paddle._C_ops.full_with_tensor(full_1, stack_65, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_327 = [parameter_159, slice_224, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_98 = paddle._C_ops.memcpy_h2d(slice_224, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_328 = [parameter_159, memcpy_h2d_98, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_66 = paddle._C_ops.stack(combine_328, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_66 = paddle._C_ops.full_with_tensor(full_1, stack_66, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_195 = paddle._C_ops.transpose(embedding_32, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_329 = [full_with_tensor_65, full_with_tensor_66]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__128, rnn__129, rnn__130, rnn__131 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_195, combine_329, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_196 = paddle._C_ops.transpose(rnn__128, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_197 = paddle._C_ops.transpose(transpose_196, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_330 = [slice_223, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_126, reshape_127 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_330), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_331 = [slice_223, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_128, reshape_129 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_331), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_198 = paddle._C_ops.transpose(transpose_197, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_126 = paddle.matmul(transpose_198, reshape_126, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__33 = paddle._C_ops.scale(matmul_126, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_101 = paddle._C_ops.shape(scale__33)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_225 = paddle._C_ops.slice(shape_101, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_332 = [slice_225, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__33, combine_332), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_66 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_99 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_32 = full_66 < memcpy_h2d_99

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_67 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_96 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_97 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_98 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, xf32, xi64, -1x40x6x40xf32, xi32) <- (xb, -1x40x6x40xf32, xi64, xf32, xi64, -1x40x6x40xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12467 = 0
        while less_than_32:
            less_than_32, reshape__134, full_66, assign_value_96, assign_value_98, full_67, assign_value_97, = self.pd_op_while_12467_0_0(full_0, arange_0, feed_2, constant_143, parameter_284, constant_144, constant_145, parameter_285, parameter_286, cast_1, less_than_32, reshape__134, full_66, assign_value_96, assign_value_98, full_67, assign_value_97)
            while_loop_counter_12467 += 1
            if while_loop_counter_12467 > kWhileLoopLimit:
                break
            
        while_192, while_193, while_194, while_195, while_196, while_197, = reshape__134, full_66, assign_value_96, assign_value_98, full_67, assign_value_97,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_333 = [slice_225, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_196, combine_333), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__63 = paddle._C_ops.softmax(reshape__136, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_199 = paddle._C_ops.transpose(reshape_128, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_127 = paddle.matmul(softmax__63, transpose_199, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_200 = paddle._C_ops.transpose(matmul_127, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_226 = paddle._C_ops.slice(transpose_200, [2], constant_138, constant_142, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_227 = paddle._C_ops.slice(transpose_8, [1], constant_138, constant_142, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_334 = [slice_226, slice_227]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_334, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_128 = paddle.matmul(concat_31, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__82 = paddle._C_ops.add(matmul_128, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_31 = paddle._C_ops.split_with_num(add__82, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_228 = split_with_num_31[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__31 = paddle._C_ops.sigmoid(slice_228)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_229 = split_with_num_31[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__31 = paddle._C_ops.multiply(slice_229, sigmoid__31)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_129 = paddle.matmul(multiply__31, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__83 = paddle._C_ops.add(matmul_129, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__64 = paddle._C_ops.softmax(add__83, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_31 = paddle._C_ops.argmax(softmax__64, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_33 = paddle._C_ops.cast(argmax_31, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__31 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__30, cast_33, constant_142, constant_146, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_33 = paddle._C_ops.embedding(set_value_with_tensor__31, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_102 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_230 = paddle._C_ops.slice(shape_102, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_103 = paddle._C_ops.shape(embedding_33)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_231 = paddle._C_ops.slice(shape_103, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_335 = [parameter_159, slice_231, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_100 = paddle._C_ops.memcpy_h2d(slice_231, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_336 = [parameter_159, memcpy_h2d_100, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_67 = paddle._C_ops.stack(combine_336, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_67 = paddle._C_ops.full_with_tensor(full_1, stack_67, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_337 = [parameter_159, slice_231, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_101 = paddle._C_ops.memcpy_h2d(slice_231, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_338 = [parameter_159, memcpy_h2d_101, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_68 = paddle._C_ops.stack(combine_338, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_68 = paddle._C_ops.full_with_tensor(full_1, stack_68, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_201 = paddle._C_ops.transpose(embedding_33, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_339 = [full_with_tensor_67, full_with_tensor_68]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__132, rnn__133, rnn__134, rnn__135 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_201, combine_339, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_202 = paddle._C_ops.transpose(rnn__132, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_203 = paddle._C_ops.transpose(transpose_202, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_340 = [slice_230, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_130, reshape_131 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_340), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_341 = [slice_230, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_132, reshape_133 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_341), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_204 = paddle._C_ops.transpose(transpose_203, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_130 = paddle.matmul(transpose_204, reshape_130, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__34 = paddle._C_ops.scale(matmul_130, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_104 = paddle._C_ops.shape(scale__34)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_232 = paddle._C_ops.slice(shape_104, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_342 = [slice_232, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__34, combine_342), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_68 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_102 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_33 = full_68 < memcpy_h2d_102

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_69 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_99 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_100 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_101 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi32) <- (xb, xi64, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12562 = 0
        while less_than_33:
            less_than_33, assign_value_101, full_68, reshape__138, full_69, assign_value_99, assign_value_100, = self.pd_op_while_12562_0_0(full_0, arange_0, feed_2, constant_147, parameter_287, constant_148, constant_149, parameter_288, parameter_289, cast_1, less_than_33, assign_value_101, full_68, reshape__138, full_69, assign_value_99, assign_value_100)
            while_loop_counter_12562 += 1
            if while_loop_counter_12562 > kWhileLoopLimit:
                break
            
        while_198, while_199, while_200, while_201, while_202, while_203, = assign_value_101, full_68, reshape__138, full_69, assign_value_99, assign_value_100,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_343 = [slice_232, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_201, combine_343), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__65 = paddle._C_ops.softmax(reshape__140, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_205 = paddle._C_ops.transpose(reshape_132, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_131 = paddle.matmul(softmax__65, transpose_205, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_206 = paddle._C_ops.transpose(matmul_131, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_233 = paddle._C_ops.slice(transpose_206, [2], constant_142, constant_146, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_234 = paddle._C_ops.slice(transpose_8, [1], constant_142, constant_146, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_344 = [slice_233, slice_234]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_344, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_132 = paddle.matmul(concat_32, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__84 = paddle._C_ops.add(matmul_132, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_32 = paddle._C_ops.split_with_num(add__84, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_235 = split_with_num_32[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__32 = paddle._C_ops.sigmoid(slice_235)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_236 = split_with_num_32[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__32 = paddle._C_ops.multiply(slice_236, sigmoid__32)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_133 = paddle.matmul(multiply__32, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__85 = paddle._C_ops.add(matmul_133, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__66 = paddle._C_ops.softmax(add__85, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_32 = paddle._C_ops.argmax(softmax__66, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_34 = paddle._C_ops.cast(argmax_32, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__32 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__31, cast_34, constant_146, constant_150, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_34 = paddle._C_ops.embedding(set_value_with_tensor__32, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_105 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_237 = paddle._C_ops.slice(shape_105, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_106 = paddle._C_ops.shape(embedding_34)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_238 = paddle._C_ops.slice(shape_106, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_345 = [parameter_159, slice_238, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_103 = paddle._C_ops.memcpy_h2d(slice_238, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_346 = [parameter_159, memcpy_h2d_103, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_69 = paddle._C_ops.stack(combine_346, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_69 = paddle._C_ops.full_with_tensor(full_1, stack_69, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_347 = [parameter_159, slice_238, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_104 = paddle._C_ops.memcpy_h2d(slice_238, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_348 = [parameter_159, memcpy_h2d_104, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_70 = paddle._C_ops.stack(combine_348, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_70 = paddle._C_ops.full_with_tensor(full_1, stack_70, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_207 = paddle._C_ops.transpose(embedding_34, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_349 = [full_with_tensor_69, full_with_tensor_70]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__136, rnn__137, rnn__138, rnn__139 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_207, combine_349, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_208 = paddle._C_ops.transpose(rnn__136, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_209 = paddle._C_ops.transpose(transpose_208, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_350 = [slice_237, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_134, reshape_135 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_350), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_351 = [slice_237, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_136, reshape_137 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_351), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_210 = paddle._C_ops.transpose(transpose_209, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_134 = paddle.matmul(transpose_210, reshape_134, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__35 = paddle._C_ops.scale(matmul_134, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_107 = paddle._C_ops.shape(scale__35)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_239 = paddle._C_ops.slice(shape_107, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_352 = [slice_239, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__35, combine_352), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_70 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_105 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_34 = full_70 < memcpy_h2d_105

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_71 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_102 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_103 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_104 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xi32) <- (xb, xi64, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12657 = 0
        while less_than_34:
            less_than_34, assign_value_104, assign_value_102, reshape__142, full_71, full_70, assign_value_103, = self.pd_op_while_12657_0_0(full_0, arange_0, feed_2, constant_151, parameter_290, constant_152, constant_153, parameter_291, parameter_292, cast_1, less_than_34, assign_value_104, assign_value_102, reshape__142, full_71, full_70, assign_value_103)
            while_loop_counter_12657 += 1
            if while_loop_counter_12657 > kWhileLoopLimit:
                break
            
        while_204, while_205, while_206, while_207, while_208, while_209, = assign_value_104, assign_value_102, reshape__142, full_71, full_70, assign_value_103,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_353 = [slice_239, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_207, combine_353), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__67 = paddle._C_ops.softmax(reshape__144, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_211 = paddle._C_ops.transpose(reshape_136, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_135 = paddle.matmul(softmax__67, transpose_211, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_212 = paddle._C_ops.transpose(matmul_135, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_240 = paddle._C_ops.slice(transpose_212, [2], constant_146, constant_150, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_241 = paddle._C_ops.slice(transpose_8, [1], constant_146, constant_150, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_354 = [slice_240, slice_241]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_354, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_136 = paddle.matmul(concat_33, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__86 = paddle._C_ops.add(matmul_136, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_33 = paddle._C_ops.split_with_num(add__86, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_242 = split_with_num_33[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__33 = paddle._C_ops.sigmoid(slice_242)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_243 = split_with_num_33[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__33 = paddle._C_ops.multiply(slice_243, sigmoid__33)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_137 = paddle.matmul(multiply__33, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__87 = paddle._C_ops.add(matmul_137, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__68 = paddle._C_ops.softmax(add__87, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_33 = paddle._C_ops.argmax(softmax__68, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_35 = paddle._C_ops.cast(argmax_33, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__33 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__32, cast_35, constant_150, constant_154, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_35 = paddle._C_ops.embedding(set_value_with_tensor__33, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_108 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_244 = paddle._C_ops.slice(shape_108, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_109 = paddle._C_ops.shape(embedding_35)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_245 = paddle._C_ops.slice(shape_109, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_355 = [parameter_159, slice_245, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_106 = paddle._C_ops.memcpy_h2d(slice_245, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_356 = [parameter_159, memcpy_h2d_106, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_71 = paddle._C_ops.stack(combine_356, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_71 = paddle._C_ops.full_with_tensor(full_1, stack_71, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_357 = [parameter_159, slice_245, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_107 = paddle._C_ops.memcpy_h2d(slice_245, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_358 = [parameter_159, memcpy_h2d_107, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_72 = paddle._C_ops.stack(combine_358, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_72 = paddle._C_ops.full_with_tensor(full_1, stack_72, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_213 = paddle._C_ops.transpose(embedding_35, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_359 = [full_with_tensor_71, full_with_tensor_72]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__140, rnn__141, rnn__142, rnn__143 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_213, combine_359, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_214 = paddle._C_ops.transpose(rnn__140, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_215 = paddle._C_ops.transpose(transpose_214, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_360 = [slice_244, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_138, reshape_139 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_360), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_361 = [slice_244, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_140, reshape_141 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_361), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_216 = paddle._C_ops.transpose(transpose_215, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_138 = paddle.matmul(transpose_216, reshape_138, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__36 = paddle._C_ops.scale(matmul_138, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_110 = paddle._C_ops.shape(scale__36)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_246 = paddle._C_ops.slice(shape_110, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_362 = [slice_246, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__36, combine_362), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_72 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_108 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_35 = full_72 < memcpy_h2d_108

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_73 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_105 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_106 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_107 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32, xi32, xi64) <- (xb, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32, xi32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12752 = 0
        while less_than_35:
            less_than_35, full_73, assign_value_105, assign_value_107, reshape__146, assign_value_106, full_72, = self.pd_op_while_12752_0_0(full_0, arange_0, feed_2, constant_155, parameter_293, constant_156, constant_157, parameter_294, parameter_295, cast_1, less_than_35, full_73, assign_value_105, assign_value_107, reshape__146, assign_value_106, full_72)
            while_loop_counter_12752 += 1
            if while_loop_counter_12752 > kWhileLoopLimit:
                break
            
        while_210, while_211, while_212, while_213, while_214, while_215, = full_73, assign_value_105, assign_value_107, reshape__146, assign_value_106, full_72,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_363 = [slice_246, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_210, combine_363), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__69 = paddle._C_ops.softmax(reshape__148, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_217 = paddle._C_ops.transpose(reshape_140, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_139 = paddle.matmul(softmax__69, transpose_217, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_218 = paddle._C_ops.transpose(matmul_139, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_247 = paddle._C_ops.slice(transpose_218, [2], constant_150, constant_154, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_248 = paddle._C_ops.slice(transpose_8, [1], constant_150, constant_154, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_364 = [slice_247, slice_248]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_364, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_140 = paddle.matmul(concat_34, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__88 = paddle._C_ops.add(matmul_140, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_34 = paddle._C_ops.split_with_num(add__88, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_249 = split_with_num_34[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__34 = paddle._C_ops.sigmoid(slice_249)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_250 = split_with_num_34[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__34 = paddle._C_ops.multiply(slice_250, sigmoid__34)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_141 = paddle.matmul(multiply__34, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__89 = paddle._C_ops.add(matmul_141, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__70 = paddle._C_ops.softmax(add__89, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_34 = paddle._C_ops.argmax(softmax__70, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_36 = paddle._C_ops.cast(argmax_34, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__34 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__33, cast_36, constant_154, constant_158, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_36 = paddle._C_ops.embedding(set_value_with_tensor__34, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_111 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_251 = paddle._C_ops.slice(shape_111, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_112 = paddle._C_ops.shape(embedding_36)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_252 = paddle._C_ops.slice(shape_112, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_365 = [parameter_159, slice_252, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_109 = paddle._C_ops.memcpy_h2d(slice_252, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_366 = [parameter_159, memcpy_h2d_109, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_73 = paddle._C_ops.stack(combine_366, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_73 = paddle._C_ops.full_with_tensor(full_1, stack_73, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_367 = [parameter_159, slice_252, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_110 = paddle._C_ops.memcpy_h2d(slice_252, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_368 = [parameter_159, memcpy_h2d_110, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_74 = paddle._C_ops.stack(combine_368, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_74 = paddle._C_ops.full_with_tensor(full_1, stack_74, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_219 = paddle._C_ops.transpose(embedding_36, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_369 = [full_with_tensor_73, full_with_tensor_74]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__144, rnn__145, rnn__146, rnn__147 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_219, combine_369, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_220 = paddle._C_ops.transpose(rnn__144, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_221 = paddle._C_ops.transpose(transpose_220, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_370 = [slice_251, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_142, reshape_143 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_370), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_371 = [slice_251, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_144, reshape_145 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_371), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_222 = paddle._C_ops.transpose(transpose_221, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_142 = paddle.matmul(transpose_222, reshape_142, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__37 = paddle._C_ops.scale(matmul_142, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_113 = paddle._C_ops.shape(scale__37)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_253 = paddle._C_ops.slice(shape_113, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_372 = [slice_253, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__37, combine_372), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_74 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_111 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_36 = full_74 < memcpy_h2d_111

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_75 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_108 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_109 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_110 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xf32, xi64, xi64, -1x40x6x40xf32, xi32) <- (xb, -1x40x6x40xf32, xf32, xi64, xi64, -1x40x6x40xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12847 = 0
        while less_than_36:
            less_than_36, reshape__150, assign_value_108, full_74, assign_value_110, full_75, assign_value_109, = self.pd_op_while_12847_0_0(full_0, arange_0, feed_2, constant_159, parameter_296, constant_160, constant_161, parameter_297, parameter_298, cast_1, less_than_36, reshape__150, assign_value_108, full_74, assign_value_110, full_75, assign_value_109)
            while_loop_counter_12847 += 1
            if while_loop_counter_12847 > kWhileLoopLimit:
                break
            
        while_216, while_217, while_218, while_219, while_220, while_221, = reshape__150, assign_value_108, full_74, assign_value_110, full_75, assign_value_109,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_373 = [slice_253, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_220, combine_373), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__71 = paddle._C_ops.softmax(reshape__152, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_223 = paddle._C_ops.transpose(reshape_144, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_143 = paddle.matmul(softmax__71, transpose_223, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_224 = paddle._C_ops.transpose(matmul_143, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_254 = paddle._C_ops.slice(transpose_224, [2], constant_154, constant_158, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_255 = paddle._C_ops.slice(transpose_8, [1], constant_154, constant_158, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_374 = [slice_254, slice_255]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_374, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_144 = paddle.matmul(concat_35, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__90 = paddle._C_ops.add(matmul_144, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_35 = paddle._C_ops.split_with_num(add__90, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_256 = split_with_num_35[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__35 = paddle._C_ops.sigmoid(slice_256)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_257 = split_with_num_35[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__35 = paddle._C_ops.multiply(slice_257, sigmoid__35)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_145 = paddle.matmul(multiply__35, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__91 = paddle._C_ops.add(matmul_145, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__72 = paddle._C_ops.softmax(add__91, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_35 = paddle._C_ops.argmax(softmax__72, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_37 = paddle._C_ops.cast(argmax_35, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__35 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__34, cast_37, constant_158, constant_162, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_37 = paddle._C_ops.embedding(set_value_with_tensor__35, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_114 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_258 = paddle._C_ops.slice(shape_114, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_115 = paddle._C_ops.shape(embedding_37)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_259 = paddle._C_ops.slice(shape_115, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_375 = [parameter_159, slice_259, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_112 = paddle._C_ops.memcpy_h2d(slice_259, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_376 = [parameter_159, memcpy_h2d_112, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_75 = paddle._C_ops.stack(combine_376, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_75 = paddle._C_ops.full_with_tensor(full_1, stack_75, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_377 = [parameter_159, slice_259, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_113 = paddle._C_ops.memcpy_h2d(slice_259, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_378 = [parameter_159, memcpy_h2d_113, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_76 = paddle._C_ops.stack(combine_378, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_76 = paddle._C_ops.full_with_tensor(full_1, stack_76, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_225 = paddle._C_ops.transpose(embedding_37, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_379 = [full_with_tensor_75, full_with_tensor_76]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__148, rnn__149, rnn__150, rnn__151 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_225, combine_379, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_226 = paddle._C_ops.transpose(rnn__148, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_227 = paddle._C_ops.transpose(transpose_226, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_380 = [slice_258, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_146, reshape_147 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_380), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_381 = [slice_258, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_148, reshape_149 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_381), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_228 = paddle._C_ops.transpose(transpose_227, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_146 = paddle.matmul(transpose_228, reshape_146, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__38 = paddle._C_ops.scale(matmul_146, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_116 = paddle._C_ops.shape(scale__38)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_260 = paddle._C_ops.slice(shape_116, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_382 = [slice_260, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__38, combine_382), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_76 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_114 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_37 = full_76 < memcpy_h2d_114

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_77 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_111 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_112 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_113 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32) <- (xb, xi32, xi64, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12942 = 0
        while less_than_37:
            less_than_37, assign_value_112, full_76, full_77, assign_value_111, assign_value_113, reshape__154, = self.pd_op_while_12942_0_0(full_0, arange_0, feed_2, constant_163, parameter_299, constant_164, constant_165, parameter_300, parameter_301, cast_1, less_than_37, assign_value_112, full_76, full_77, assign_value_111, assign_value_113, reshape__154)
            while_loop_counter_12942 += 1
            if while_loop_counter_12942 > kWhileLoopLimit:
                break
            
        while_222, while_223, while_224, while_225, while_226, while_227, = assign_value_112, full_76, full_77, assign_value_111, assign_value_113, reshape__154,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_383 = [slice_260, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_224, combine_383), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__73 = paddle._C_ops.softmax(reshape__156, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_229 = paddle._C_ops.transpose(reshape_148, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_147 = paddle.matmul(softmax__73, transpose_229, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_230 = paddle._C_ops.transpose(matmul_147, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_261 = paddle._C_ops.slice(transpose_230, [2], constant_158, constant_162, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_262 = paddle._C_ops.slice(transpose_8, [1], constant_158, constant_162, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_384 = [slice_261, slice_262]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_384, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_148 = paddle.matmul(concat_36, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__92 = paddle._C_ops.add(matmul_148, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_36 = paddle._C_ops.split_with_num(add__92, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_263 = split_with_num_36[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__36 = paddle._C_ops.sigmoid(slice_263)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_264 = split_with_num_36[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__36 = paddle._C_ops.multiply(slice_264, sigmoid__36)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_149 = paddle.matmul(multiply__36, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__93 = paddle._C_ops.add(matmul_149, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__74 = paddle._C_ops.softmax(add__93, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_36 = paddle._C_ops.argmax(softmax__74, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_38 = paddle._C_ops.cast(argmax_36, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__36 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__35, cast_38, constant_162, constant_166, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_38 = paddle._C_ops.embedding(set_value_with_tensor__36, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_117 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_265 = paddle._C_ops.slice(shape_117, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_118 = paddle._C_ops.shape(embedding_38)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_266 = paddle._C_ops.slice(shape_118, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_385 = [parameter_159, slice_266, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_115 = paddle._C_ops.memcpy_h2d(slice_266, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_386 = [parameter_159, memcpy_h2d_115, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_77 = paddle._C_ops.stack(combine_386, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_77 = paddle._C_ops.full_with_tensor(full_1, stack_77, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_387 = [parameter_159, slice_266, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_116 = paddle._C_ops.memcpy_h2d(slice_266, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_388 = [parameter_159, memcpy_h2d_116, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_78 = paddle._C_ops.stack(combine_388, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_78 = paddle._C_ops.full_with_tensor(full_1, stack_78, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_231 = paddle._C_ops.transpose(embedding_38, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_389 = [full_with_tensor_77, full_with_tensor_78]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__152, rnn__153, rnn__154, rnn__155 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_231, combine_389, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_232 = paddle._C_ops.transpose(rnn__152, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_233 = paddle._C_ops.transpose(transpose_232, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_390 = [slice_265, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_150, reshape_151 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_390), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_391 = [slice_265, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_152, reshape_153 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_391), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_234 = paddle._C_ops.transpose(transpose_233, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_150 = paddle.matmul(transpose_234, reshape_150, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__39 = paddle._C_ops.scale(matmul_150, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_119 = paddle._C_ops.shape(scale__39)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_267 = paddle._C_ops.slice(shape_119, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_392 = [slice_267, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__39, combine_392), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_78 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_117 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_38 = full_78 < memcpy_h2d_117

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_79 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_114 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_115 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_116 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xi32, xf32) <- (xb, xi64, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xi32, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_13037 = 0
        while less_than_38:
            less_than_38, full_78, reshape__158, assign_value_116, full_79, assign_value_115, assign_value_114, = self.pd_op_while_13037_0_0(full_0, arange_0, feed_2, constant_167, parameter_302, constant_168, constant_169, parameter_303, parameter_304, cast_1, less_than_38, full_78, reshape__158, assign_value_116, full_79, assign_value_115, assign_value_114)
            while_loop_counter_13037 += 1
            if while_loop_counter_13037 > kWhileLoopLimit:
                break
            
        while_228, while_229, while_230, while_231, while_232, while_233, = full_78, reshape__158, assign_value_116, full_79, assign_value_115, assign_value_114,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_393 = [slice_267, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_231, combine_393), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__75 = paddle._C_ops.softmax(reshape__160, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_235 = paddle._C_ops.transpose(reshape_152, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_151 = paddle.matmul(softmax__75, transpose_235, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_236 = paddle._C_ops.transpose(matmul_151, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_268 = paddle._C_ops.slice(transpose_236, [2], constant_162, constant_166, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_269 = paddle._C_ops.slice(transpose_8, [1], constant_162, constant_166, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_394 = [slice_268, slice_269]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_394, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_152 = paddle.matmul(concat_37, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__94 = paddle._C_ops.add(matmul_152, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_37 = paddle._C_ops.split_with_num(add__94, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_270 = split_with_num_37[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__37 = paddle._C_ops.sigmoid(slice_270)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_271 = split_with_num_37[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__37 = paddle._C_ops.multiply(slice_271, sigmoid__37)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_153 = paddle.matmul(multiply__37, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__95 = paddle._C_ops.add(matmul_153, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__76 = paddle._C_ops.softmax(add__95, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_37 = paddle._C_ops.argmax(softmax__76, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_39 = paddle._C_ops.cast(argmax_37, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__37 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__36, cast_39, constant_166, constant_170, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_39 = paddle._C_ops.embedding(set_value_with_tensor__37, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_120 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_272 = paddle._C_ops.slice(shape_120, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_121 = paddle._C_ops.shape(embedding_39)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_273 = paddle._C_ops.slice(shape_121, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_395 = [parameter_159, slice_273, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_118 = paddle._C_ops.memcpy_h2d(slice_273, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_396 = [parameter_159, memcpy_h2d_118, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_79 = paddle._C_ops.stack(combine_396, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_79 = paddle._C_ops.full_with_tensor(full_1, stack_79, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_397 = [parameter_159, slice_273, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_119 = paddle._C_ops.memcpy_h2d(slice_273, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_398 = [parameter_159, memcpy_h2d_119, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_80 = paddle._C_ops.stack(combine_398, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_80 = paddle._C_ops.full_with_tensor(full_1, stack_80, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_237 = paddle._C_ops.transpose(embedding_39, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_399 = [full_with_tensor_79, full_with_tensor_80]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__156, rnn__157, rnn__158, rnn__159 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_237, combine_399, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_238 = paddle._C_ops.transpose(rnn__156, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_239 = paddle._C_ops.transpose(transpose_238, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_400 = [slice_272, constant_7, constant_9]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_154, reshape_155 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_400), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_401 = [slice_272, constant_10, constant_9]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_156, reshape_157 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_401), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_240 = paddle._C_ops.transpose(transpose_239, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_154 = paddle.matmul(transpose_240, reshape_154, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__40 = paddle._C_ops.scale(matmul_154, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_122 = paddle._C_ops.shape(scale__40)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_274 = paddle._C_ops.slice(shape_122, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_402 = [slice_274, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__40, combine_402), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_80 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_120 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_39 = full_80 < memcpy_h2d_120

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_81 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_117 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_118 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_119 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64) <- (xb, xi32, xi64, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_13132 = 0
        while less_than_39:
            less_than_39, assign_value_118, assign_value_119, assign_value_117, reshape__162, full_81, full_80, = self.pd_op_while_13132_0_0(full_0, arange_0, feed_2, constant_171, parameter_305, constant_172, constant_173, parameter_306, parameter_307, cast_1, less_than_39, assign_value_118, assign_value_119, assign_value_117, reshape__162, full_81, full_80)
            while_loop_counter_13132 += 1
            if while_loop_counter_13132 > kWhileLoopLimit:
                break
            
        while_234, while_235, while_236, while_237, while_238, while_239, = assign_value_118, assign_value_119, assign_value_117, reshape__162, full_81, full_80,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_403 = [slice_274, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_238, combine_403), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__77 = paddle._C_ops.softmax(reshape__164, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_241 = paddle._C_ops.transpose(reshape_156, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_155 = paddle.matmul(softmax__77, transpose_241, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_242 = paddle._C_ops.transpose(matmul_155, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_275 = paddle._C_ops.slice(transpose_242, [2], constant_166, constant_170, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_276 = paddle._C_ops.slice(transpose_8, [1], constant_166, constant_170, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_404 = [slice_275, slice_276]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_404, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_156 = paddle.matmul(concat_38, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__96 = paddle._C_ops.add(matmul_156, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_38 = paddle._C_ops.split_with_num(add__96, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_277 = split_with_num_38[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__38 = paddle._C_ops.sigmoid(slice_277)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_278 = split_with_num_38[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__38 = paddle._C_ops.multiply(slice_278, sigmoid__38)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_157 = paddle.matmul(multiply__38, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__97 = paddle._C_ops.add(matmul_157, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__78 = paddle._C_ops.softmax(add__97, -1)

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_38 = paddle._C_ops.argmax(softmax__78, constant_21, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_40 = paddle._C_ops.cast(argmax_38, paddle.int64)

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__38 = paddle._C_ops.set_value_with_tensor(set_value_with_tensor__37, cast_40, constant_170, constant_174, constant_3, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_40 = paddle._C_ops.embedding(set_value_with_tensor__38, parameter_178, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_123 = paddle._C_ops.shape(add__17)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_279 = paddle._C_ops.slice(shape_123, [0], constant_2, constant_3, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_124 = paddle._C_ops.shape(embedding_40)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_280 = paddle._C_ops.slice(shape_124, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_405 = [parameter_159, slice_280, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_121 = paddle._C_ops.memcpy_h2d(slice_280, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_406 = [parameter_159, memcpy_h2d_121, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_81 = paddle._C_ops.stack(combine_406, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_81 = paddle._C_ops.full_with_tensor(full_1, stack_81, paddle.float32)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_407 = [parameter_159, slice_280, parameter_160]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_122 = paddle._C_ops.memcpy_h2d(slice_280, 1)

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_408 = [parameter_159, memcpy_h2d_122, parameter_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_82 = paddle._C_ops.stack(combine_408, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_82 = paddle._C_ops.full_with_tensor(full_1, stack_82, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_243 = paddle._C_ops.transpose(embedding_40, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_409 = [full_with_tensor_81, full_with_tensor_82]

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__160, rnn__161, rnn__162, rnn__163 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_243, combine_409, combine_19, None, parameter_169, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_244 = paddle._C_ops.transpose(rnn__160, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_245 = paddle._C_ops.transpose(transpose_244, [0, 2, 1])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_410 = [slice_279, constant_7, constant_9]

        # pd_op.reshape_: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, combine_410), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_411 = [slice_279, constant_10, constant_9]

        # pd_op.reshape_: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, combine_411), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_246 = paddle._C_ops.transpose(transpose_245, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_158 = paddle.matmul(transpose_246, reshape__166, transpose_x=False, transpose_y=False)

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__41 = paddle._C_ops.scale(matmul_158, full_0, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_125 = paddle._C_ops.shape(scale__41)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_281 = paddle._C_ops.slice(shape_125, [0], constant_2, constant_3, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_412 = [slice_281, constant_6, constant_8, constant_6]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape(scale__41, combine_412), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (xi64) <- ()
        full_82 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_123 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_40 = full_82 < memcpy_h2d_123

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_83 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_120 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_121 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_122 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32) <- (xb, xi64, xi32, xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is not None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_13227 = 0
        while less_than_40:
            less_than_40, full_82, assign_value_121, assign_value_122, full_83, assign_value_120, reshape__170, = self.pd_op_while_13227_0_0(full_0, arange_0, feed_2, constant_175, parameter_308, constant_176, constant_177, parameter_309, parameter_310, cast_1, less_than_40, full_82, assign_value_121, assign_value_122, full_83, assign_value_120, reshape__170)
            while_loop_counter_13227 += 1
            if while_loop_counter_13227 > kWhileLoopLimit:
                break
            
        while_240, while_241, while_242, while_243, while_244, while_245, = full_82, assign_value_121, assign_value_122, full_83, assign_value_120, reshape__170,

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_413 = [slice_281, constant_6, constant_9]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__172, reshape__173 = (lambda x, f: f(x))(paddle._C_ops.reshape(while_243, combine_413), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__79 = paddle._C_ops.softmax(reshape__172, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_247 = paddle._C_ops.transpose(reshape__168, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_159 = paddle.matmul(softmax__79, transpose_247, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_248 = paddle._C_ops.transpose(matmul_159, [0, 2, 1])

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_282 = paddle._C_ops.slice(transpose_248, [2], constant_170, constant_174, [1], [2])

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_283 = paddle._C_ops.slice(transpose_8, [1], constant_170, constant_174, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_414 = [slice_282, slice_283]

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_39 = paddle._C_ops.concat(combine_414, constant_19)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_160 = paddle.matmul(concat_39, parameter_190, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__98 = paddle._C_ops.add(matmul_160, parameter_191)

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_39 = paddle._C_ops.split_with_num(add__98, 2, constant_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_284 = split_with_num_39[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__39 = paddle._C_ops.sigmoid(slice_284)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_285 = split_with_num_39[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__39 = paddle._C_ops.multiply(slice_285, sigmoid__39)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_161 = paddle.matmul(multiply__39, parameter_192, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__99 = paddle._C_ops.add(matmul_161, parameter_193)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__80 = paddle._C_ops.softmax(add__99, -1)

        # builtin.combine: ([-1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32]) <- (-1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32)
        combine_415 = [softmax__2, softmax__4, softmax__6, softmax__8, softmax__10, softmax__12, softmax__14, softmax__16, softmax__18, softmax__20, softmax__22, softmax__24, softmax__26, softmax__28, softmax__30, softmax__32, softmax__34, softmax__36, softmax__38, softmax__40, softmax__42, softmax__44, softmax__46, softmax__48, softmax__50, softmax__52, softmax__54, softmax__56, softmax__58, softmax__60, softmax__62, softmax__64, softmax__66, softmax__68, softmax__70, softmax__72, softmax__74, softmax__76, softmax__78, softmax__80]

        # pd_op.stack: (-1x40x92xf32) <- ([-1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32])
        stack_83 = paddle._C_ops.stack(combine_415, 1)

        # pd_op.scale_: (-1x40x92xf32) <- (-1x40x92xf32, 1xf32)
        scale__42 = paddle._C_ops.scale(stack_83, full_0, float('0'), True)
        return scale__42



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

    def forward(self, parameter_310, parameter_309, constant_177, constant_176, parameter_308, constant_175, parameter_307, parameter_306, constant_173, constant_172, parameter_305, constant_171, parameter_304, parameter_303, constant_169, constant_168, parameter_302, constant_167, parameter_301, parameter_300, constant_165, constant_164, parameter_299, constant_163, parameter_298, parameter_297, constant_161, constant_160, parameter_296, constant_159, parameter_295, parameter_294, constant_157, constant_156, parameter_293, constant_155, parameter_292, parameter_291, constant_153, constant_152, parameter_290, constant_151, parameter_289, parameter_288, constant_149, constant_148, parameter_287, constant_147, parameter_286, parameter_285, constant_145, constant_144, parameter_284, constant_143, parameter_283, parameter_282, constant_141, constant_140, parameter_281, constant_139, parameter_280, parameter_279, constant_137, constant_136, parameter_278, constant_135, parameter_277, parameter_276, constant_133, constant_132, parameter_275, constant_131, parameter_274, parameter_273, constant_129, constant_128, parameter_272, constant_127, parameter_271, parameter_270, constant_125, constant_124, parameter_269, constant_123, parameter_268, parameter_267, constant_121, constant_120, parameter_266, constant_119, parameter_265, parameter_264, constant_117, constant_116, parameter_263, constant_115, parameter_262, parameter_261, constant_113, constant_112, parameter_260, constant_111, parameter_259, parameter_258, constant_109, constant_108, parameter_257, constant_107, parameter_256, parameter_255, constant_105, constant_104, parameter_254, constant_103, parameter_253, parameter_252, constant_101, constant_100, parameter_251, constant_99, parameter_250, parameter_249, constant_97, constant_96, parameter_248, constant_95, parameter_247, parameter_246, constant_93, constant_92, parameter_245, constant_91, parameter_244, parameter_243, constant_89, constant_88, parameter_242, constant_87, parameter_241, parameter_240, constant_85, constant_84, parameter_239, constant_83, parameter_238, parameter_237, constant_81, constant_80, parameter_236, constant_79, parameter_235, parameter_234, constant_77, constant_76, parameter_233, constant_75, parameter_232, parameter_231, constant_73, constant_72, parameter_230, constant_71, parameter_229, parameter_228, constant_69, constant_68, parameter_227, constant_67, parameter_226, parameter_225, constant_65, constant_64, parameter_224, constant_63, parameter_223, parameter_222, constant_61, constant_60, parameter_221, constant_59, parameter_220, parameter_219, constant_57, constant_56, parameter_218, constant_55, parameter_217, parameter_216, constant_53, constant_52, parameter_215, constant_51, parameter_214, parameter_213, constant_49, constant_48, parameter_212, constant_47, parameter_211, parameter_210, constant_45, constant_44, parameter_209, constant_43, parameter_208, parameter_207, constant_41, constant_40, parameter_206, constant_39, parameter_205, parameter_204, constant_37, constant_36, parameter_203, constant_35, parameter_202, parameter_201, constant_33, constant_32, parameter_200, constant_31, parameter_199, parameter_198, constant_29, constant_28, parameter_197, constant_27, parameter_196, parameter_195, constant_25, constant_24, parameter_194, constant_23, parameter_189, parameter_188, constant_18, constant_17, parameter_187, constant_16, parameter_177, parameter_176, constant_15, constant_14, parameter_175, constant_13, constant_174, constant_170, constant_166, constant_162, constant_158, constant_154, constant_150, constant_146, constant_142, constant_138, constant_134, constant_130, constant_126, constant_122, constant_118, constant_114, constant_110, constant_106, constant_102, constant_98, constant_94, constant_90, constant_86, constant_82, constant_78, constant_74, constant_70, constant_66, constant_62, constant_58, constant_54, constant_50, constant_46, constant_42, constant_38, constant_34, constant_30, constant_26, constant_22, constant_21, constant_20, constant_19, constant_12, constant_11, constant_10, constant_9, parameter_173, parameter_171, constant_8, parameter_169, parameter_160, parameter_159, constant_7, constant_6, constant_5, constant_4, parameter_158, constant_3, constant_2, parameter_157, parameter_151, parameter_115, constant_1, parameter_54, parameter_28, constant_0, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_32, parameter_29, parameter_31, parameter_30, parameter_33, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_170, parameter_172, parameter_174, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_190, parameter_191, parameter_192, parameter_193, feed_1, feed_2, feed_0):
        return self.builtin_module_8767_0_0(parameter_310, parameter_309, constant_177, constant_176, parameter_308, constant_175, parameter_307, parameter_306, constant_173, constant_172, parameter_305, constant_171, parameter_304, parameter_303, constant_169, constant_168, parameter_302, constant_167, parameter_301, parameter_300, constant_165, constant_164, parameter_299, constant_163, parameter_298, parameter_297, constant_161, constant_160, parameter_296, constant_159, parameter_295, parameter_294, constant_157, constant_156, parameter_293, constant_155, parameter_292, parameter_291, constant_153, constant_152, parameter_290, constant_151, parameter_289, parameter_288, constant_149, constant_148, parameter_287, constant_147, parameter_286, parameter_285, constant_145, constant_144, parameter_284, constant_143, parameter_283, parameter_282, constant_141, constant_140, parameter_281, constant_139, parameter_280, parameter_279, constant_137, constant_136, parameter_278, constant_135, parameter_277, parameter_276, constant_133, constant_132, parameter_275, constant_131, parameter_274, parameter_273, constant_129, constant_128, parameter_272, constant_127, parameter_271, parameter_270, constant_125, constant_124, parameter_269, constant_123, parameter_268, parameter_267, constant_121, constant_120, parameter_266, constant_119, parameter_265, parameter_264, constant_117, constant_116, parameter_263, constant_115, parameter_262, parameter_261, constant_113, constant_112, parameter_260, constant_111, parameter_259, parameter_258, constant_109, constant_108, parameter_257, constant_107, parameter_256, parameter_255, constant_105, constant_104, parameter_254, constant_103, parameter_253, parameter_252, constant_101, constant_100, parameter_251, constant_99, parameter_250, parameter_249, constant_97, constant_96, parameter_248, constant_95, parameter_247, parameter_246, constant_93, constant_92, parameter_245, constant_91, parameter_244, parameter_243, constant_89, constant_88, parameter_242, constant_87, parameter_241, parameter_240, constant_85, constant_84, parameter_239, constant_83, parameter_238, parameter_237, constant_81, constant_80, parameter_236, constant_79, parameter_235, parameter_234, constant_77, constant_76, parameter_233, constant_75, parameter_232, parameter_231, constant_73, constant_72, parameter_230, constant_71, parameter_229, parameter_228, constant_69, constant_68, parameter_227, constant_67, parameter_226, parameter_225, constant_65, constant_64, parameter_224, constant_63, parameter_223, parameter_222, constant_61, constant_60, parameter_221, constant_59, parameter_220, parameter_219, constant_57, constant_56, parameter_218, constant_55, parameter_217, parameter_216, constant_53, constant_52, parameter_215, constant_51, parameter_214, parameter_213, constant_49, constant_48, parameter_212, constant_47, parameter_211, parameter_210, constant_45, constant_44, parameter_209, constant_43, parameter_208, parameter_207, constant_41, constant_40, parameter_206, constant_39, parameter_205, parameter_204, constant_37, constant_36, parameter_203, constant_35, parameter_202, parameter_201, constant_33, constant_32, parameter_200, constant_31, parameter_199, parameter_198, constant_29, constant_28, parameter_197, constant_27, parameter_196, parameter_195, constant_25, constant_24, parameter_194, constant_23, parameter_189, parameter_188, constant_18, constant_17, parameter_187, constant_16, parameter_177, parameter_176, constant_15, constant_14, parameter_175, constant_13, constant_174, constant_170, constant_166, constant_162, constant_158, constant_154, constant_150, constant_146, constant_142, constant_138, constant_134, constant_130, constant_126, constant_122, constant_118, constant_114, constant_110, constant_106, constant_102, constant_98, constant_94, constant_90, constant_86, constant_82, constant_78, constant_74, constant_70, constant_66, constant_62, constant_58, constant_54, constant_50, constant_46, constant_42, constant_38, constant_34, constant_30, constant_26, constant_22, constant_21, constant_20, constant_19, constant_12, constant_11, constant_10, constant_9, parameter_173, parameter_171, constant_8, parameter_169, parameter_160, parameter_159, constant_7, constant_6, constant_5, constant_4, parameter_158, constant_3, constant_2, parameter_157, parameter_151, parameter_115, constant_1, parameter_54, parameter_28, constant_0, parameter_7, parameter_1, parameter_0, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_32, parameter_29, parameter_31, parameter_30, parameter_33, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_170, parameter_172, parameter_174, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_190, parameter_191, parameter_192, parameter_193, feed_1, feed_2, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_8767_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_310
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_177
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_176
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_308
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_175
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_173
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_172
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_305
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_171
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_169
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_168
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_302
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_167
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_165
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_164
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_299
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_163
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_161
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_160
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_296
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_159
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_157
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_156
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_293
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_155
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_153
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_152
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_290
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_151
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_149
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_148
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_287
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_147
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_145
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_144
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_284
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_143
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_141
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_140
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_281
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_139
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_137
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_136
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_278
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_135
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_133
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_132
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_275
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_131
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_129
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_128
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_272
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_127
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_125
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_124
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_269
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_123
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_121
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_120
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_266
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_119
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_117
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_116
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_263
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_115
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_113
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_112
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_260
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_111
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_109
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_108
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_257
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_107
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_105
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_104
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_254
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_103
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_101
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_100
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_251
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_99
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_97
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_96
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_248
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_95
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_93
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_92
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_245
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_91
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_89
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_88
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_242
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_87
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_85
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_84
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_239
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_83
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_81
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_80
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_236
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_79
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_77
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_76
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_233
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_75
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_73
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_72
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_230
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_71
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_69
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_68
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_227
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_67
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_65
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_64
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_224
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_63
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_61
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_60
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_221
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_59
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_57
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_56
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_218
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_55
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_53
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_52
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_215
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_51
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_49
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_48
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_212
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_47
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_45
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_44
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_209
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_43
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_41
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_40
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_206
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_39
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_37
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_36
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_203
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_35
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_33
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_32
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_200
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_31
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_29
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_28
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_197
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_27
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_25
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_24
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_194
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_23
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_18
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_17
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_187
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_16
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            # constant_15
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_14
            paddle.to_tensor([-2147483648], dtype='int32').reshape([1]),
            # parameter_175
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_13
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_174
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            # constant_170
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            # constant_166
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            # constant_162
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            # constant_158
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            # constant_154
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            # constant_150
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            # constant_146
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            # constant_142
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            # constant_138
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            # constant_134
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            # constant_130
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            # constant_126
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            # constant_122
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            # constant_118
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            # constant_114
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            # constant_110
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            # constant_106
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            # constant_102
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            # constant_98
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            # constant_94
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            # constant_90
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            # constant_86
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            # constant_82
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            # constant_78
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            # constant_74
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            # constant_70
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            # constant_66
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            # constant_62
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            # constant_58
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            # constant_54
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            # constant_50
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            # constant_46
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            # constant_42
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            # constant_38
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            # constant_34
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            # constant_30
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            # constant_26
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            # constant_22
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_21
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_20
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_19
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            # constant_12
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_11
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # constant_10
            paddle.to_tensor([512], dtype='int32').reshape([1]),
            # constant_9
            paddle.to_tensor([240], dtype='int32').reshape([1]),
            # parameter_173
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_8
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            # parameter_169
            paddle.cast(paddle.randint(low=0, high=3, shape=[], dtype='int64'), 'uint8'),
            # parameter_160
            paddle.to_tensor([128], dtype='int32').reshape([]),
            # parameter_159
            paddle.to_tensor([2], dtype='int32').reshape([]),
            # constant_7
            paddle.to_tensor([128], dtype='int32').reshape([1]),
            # constant_6
            paddle.to_tensor([40], dtype='int32').reshape([1]),
            # constant_5
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_4
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.to_tensor([40], dtype='int32').reshape([]),
            # constant_3
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_2
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # parameter_157
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_1
            paddle.to_tensor([2, 1], dtype='int64').reshape([2]),
            # parameter_54
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # constant_0
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            # parameter_7
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([41, 128], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([93, 128], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([512, 92], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([92], dtype='float32', min=0, max=0.5),
            # feed_1
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([1, 40]),
            # feed_2
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 48, 160], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_310
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_177
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_176
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_308
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_175
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_173
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_172
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_305
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_171
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_169
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_168
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_302
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_167
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_165
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_164
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_299
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_163
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_161
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_160
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_296
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_159
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_157
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_156
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_293
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_155
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_153
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_152
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_290
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_151
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_149
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_148
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_287
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_147
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_145
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_144
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_284
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_143
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_141
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_140
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_281
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_139
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_137
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_136
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_278
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_135
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_133
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_132
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_275
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_131
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_129
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_128
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_272
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_127
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_125
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_124
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_269
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_123
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_121
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_120
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_266
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_119
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_117
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_116
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_263
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_115
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_113
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_112
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_260
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_111
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_109
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_108
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_257
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_107
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_105
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_104
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_254
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_103
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_101
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_100
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_251
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_99
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_97
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_96
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_248
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_95
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_93
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_92
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_245
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_91
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_89
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_88
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_242
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_87
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_85
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_84
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_239
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_83
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_81
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_80
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_236
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_79
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_77
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_76
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_233
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_75
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_73
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_72
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_230
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_71
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_69
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_68
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_227
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_67
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_65
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_64
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_224
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_63
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_61
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_60
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_221
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_59
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_57
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_56
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_218
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_55
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_53
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_52
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_215
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_51
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_49
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_48
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_212
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_47
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_45
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_44
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_209
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_43
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_41
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_40
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_206
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_39
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_37
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_36
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_203
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_35
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_33
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_32
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_200
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_31
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_29
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_28
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_197
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_27
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_25
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_24
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_194
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_23
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_18
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_17
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_187
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[1, 40, 6, 40], dtype='float32'),
            # constant_15
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_175
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_174
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_170
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_166
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_162
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_158
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_154
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_150
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_146
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_142
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_138
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_134
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_130
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_126
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_122
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_118
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_114
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_110
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_106
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_102
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_98
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_94
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_90
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_86
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_82
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_78
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_74
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_70
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_66
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_62
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_58
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_54
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_50
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_46
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_42
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_38
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_34
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_30
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_26
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_22
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_21
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_20
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_19
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_173
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_169
            paddle.static.InputSpec(shape=[], dtype='uint8'),
            # parameter_160
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # parameter_159
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_157
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            # constant_1
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_54
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            # constant_0
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_7
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
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
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[41, 128], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[93, 128], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[512, 92], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[92], dtype='float32'),
            # feed_1
            paddle.static.InputSpec(shape=[None, 40], dtype='int64'),
            # feed_2
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 48, 160], dtype='float32'),
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