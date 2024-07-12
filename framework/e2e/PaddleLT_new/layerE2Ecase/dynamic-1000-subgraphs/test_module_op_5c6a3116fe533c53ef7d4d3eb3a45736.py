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
        return True, f"last stage failed. stderr: {stderr}"
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
    return [150][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, ):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 152, paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (152xi64) <- (1xf32, 1xf32, 1xf32)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype=paddle.int64)

        # pd_op.cast: (152xf32) <- (152xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.scale: (152xf32) <- (152xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 8, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (152xf32) <- (152xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_3, 0, True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 100, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (100xi64) <- (1xf32, 1xf32, 1xf32)
        arange_1 = paddle.arange(full_0, full_4, full_2, dtype=paddle.int64)

        # pd_op.cast: (100xf32) <- (100xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (100xf32) <- (100xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_1, full_2, 0.5, True)

        # pd_op.scale: (100xf32) <- (100xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_3, 0, True)

        # builtin.combine: ([100xf32, 152xf32]) <- (100xf32, 152xf32)
        combine_0 = [scale_3, scale_1]

        # pd_op.meshgrid: ([100x152xf32, 100x152xf32]) <- ([100xf32, 152xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (100x152xf32, 100x152xf32) <- ([100x152xf32, 100x152xf32])
        split_0, split_1, = meshgrid_0

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_2, -32, True)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_2, -32, True)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(split_1, full_2, 32, True)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(split_0, full_2, 32, True)

        # builtin.combine: ([100x152xf32, 100x152xf32, 100x152xf32, 100x152xf32]) <- (100x152xf32, 100x152xf32, 100x152xf32, 100x152xf32)
        combine_1 = [scale_4, scale_5, scale_6, scale_7]

        # pd_op.stack: (100x152x4xf32) <- ([100x152xf32, 100x152xf32, 100x152xf32, 100x152xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # pd_op.cast: (100x152x4xf32) <- (100x152x4xf32)
        cast_2 = paddle._C_ops.cast(stack_0, paddle.float32)

        # builtin.combine: ([100x152xf32, 100x152xf32]) <- (100x152xf32, 100x152xf32)
        combine_2 = [split_1, split_0]

        # pd_op.stack: (100x152x2xf32) <- ([100x152xf32, 100x152xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.cast: (100x152x2xf32) <- (100x152x2xf32)
        cast_3 = paddle._C_ops.cast(stack_1, paddle.float32)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (15200x4xf32, 0x100x152x4xi64) <- (100x152x4xf32, 2xi64)
        reshape_0, reshape_1 = paddle.reshape(cast_2, full_int_array_0), None

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (15200x2xf32, 0x100x152x2xi64) <- (100x152x2xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(cast_3, full_int_array_1), None

        # pd_op.full: (15200x1xf32) <- ()
        full_5 = paddle._C_ops.full([15200, 1], 8, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], 76, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (76xi64) <- (1xf32, 1xf32, 1xf32)
        arange_2 = paddle.arange(full_0, full_6, full_2, dtype=paddle.int64)

        # pd_op.cast: (76xf32) <- (76xi64)
        cast_4 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (76xf32) <- (76xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_4, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], 16, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (76xf32) <- (76xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_7, 0, True)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], 50, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (50xi64) <- (1xf32, 1xf32, 1xf32)
        arange_3 = paddle.arange(full_0, full_8, full_2, dtype=paddle.int64)

        # pd_op.cast: (50xf32) <- (50xi64)
        cast_5 = paddle._C_ops.cast(arange_3, paddle.float32)

        # pd_op.scale: (50xf32) <- (50xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(cast_5, full_2, 0.5, True)

        # pd_op.scale: (50xf32) <- (50xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(scale_10, full_7, 0, True)

        # builtin.combine: ([50xf32, 76xf32]) <- (50xf32, 76xf32)
        combine_3 = [scale_11, scale_9]

        # pd_op.meshgrid: ([50x76xf32, 50x76xf32]) <- ([50xf32, 76xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (50x76xf32, 50x76xf32) <- ([50x76xf32, 50x76xf32])
        split_2, split_3, = meshgrid_1

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(split_3, full_2, -64, True)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(split_2, full_2, -64, True)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_3, full_2, 64, True)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_2, full_2, 64, True)

        # builtin.combine: ([50x76xf32, 50x76xf32, 50x76xf32, 50x76xf32]) <- (50x76xf32, 50x76xf32, 50x76xf32, 50x76xf32)
        combine_4 = [scale_12, scale_13, scale_14, scale_15]

        # pd_op.stack: (50x76x4xf32) <- ([50x76xf32, 50x76xf32, 50x76xf32, 50x76xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)

        # pd_op.cast: (50x76x4xf32) <- (50x76x4xf32)
        cast_6 = paddle._C_ops.cast(stack_2, paddle.float32)

        # builtin.combine: ([50x76xf32, 50x76xf32]) <- (50x76xf32, 50x76xf32)
        combine_5 = [split_3, split_2]

        # pd_op.stack: (50x76x2xf32) <- ([50x76xf32, 50x76xf32])
        stack_3 = paddle._C_ops.stack(combine_5, -1)

        # pd_op.cast: (50x76x2xf32) <- (50x76x2xf32)
        cast_7 = paddle._C_ops.cast(stack_3, paddle.float32)

        # pd_op.reshape: (3800x4xf32, 0x50x76x4xi64) <- (50x76x4xf32, 2xi64)
        reshape_4, reshape_5 = paddle.reshape(cast_6, full_int_array_0), None

        # pd_op.reshape: (3800x2xf32, 0x50x76x2xi64) <- (50x76x2xf32, 2xi64)
        reshape_6, reshape_7 = paddle.reshape(cast_7, full_int_array_1), None

        # pd_op.full: (3800x1xf32) <- ()
        full_9 = paddle._C_ops.full([3800, 1], 16, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], 38, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (38xi64) <- (1xf32, 1xf32, 1xf32)
        arange_4 = paddle.arange(full_0, full_10, full_2, dtype=paddle.int64)

        # pd_op.cast: (38xf32) <- (38xi64)
        cast_8 = paddle._C_ops.cast(arange_4, paddle.float32)

        # pd_op.scale: (38xf32) <- (38xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(cast_8, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], 32, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (38xf32) <- (38xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(scale_16, full_11, 0, True)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], 25, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (25xi64) <- (1xf32, 1xf32, 1xf32)
        arange_5 = paddle.arange(full_0, full_12, full_2, dtype=paddle.int64)

        # pd_op.cast: (25xf32) <- (25xi64)
        cast_9 = paddle._C_ops.cast(arange_5, paddle.float32)

        # pd_op.scale: (25xf32) <- (25xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_9, full_2, 0.5, True)

        # pd_op.scale: (25xf32) <- (25xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_11, 0, True)

        # builtin.combine: ([25xf32, 38xf32]) <- (25xf32, 38xf32)
        combine_6 = [scale_19, scale_17]

        # pd_op.meshgrid: ([25x38xf32, 25x38xf32]) <- ([25xf32, 38xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (25x38xf32, 25x38xf32) <- ([25x38xf32, 25x38xf32])
        split_4, split_5, = meshgrid_2

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(split_5, full_2, -128, True)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(split_4, full_2, -128, True)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(split_5, full_2, 128, True)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(split_4, full_2, 128, True)

        # builtin.combine: ([25x38xf32, 25x38xf32, 25x38xf32, 25x38xf32]) <- (25x38xf32, 25x38xf32, 25x38xf32, 25x38xf32)
        combine_7 = [scale_20, scale_21, scale_22, scale_23]

        # pd_op.stack: (25x38x4xf32) <- ([25x38xf32, 25x38xf32, 25x38xf32, 25x38xf32])
        stack_4 = paddle._C_ops.stack(combine_7, -1)

        # pd_op.cast: (25x38x4xf32) <- (25x38x4xf32)
        cast_10 = paddle._C_ops.cast(stack_4, paddle.float32)

        # builtin.combine: ([25x38xf32, 25x38xf32]) <- (25x38xf32, 25x38xf32)
        combine_8 = [split_5, split_4]

        # pd_op.stack: (25x38x2xf32) <- ([25x38xf32, 25x38xf32])
        stack_5 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.cast: (25x38x2xf32) <- (25x38x2xf32)
        cast_11 = paddle._C_ops.cast(stack_5, paddle.float32)

        # pd_op.reshape: (950x4xf32, 0x25x38x4xi64) <- (25x38x4xf32, 2xi64)
        reshape_8, reshape_9 = paddle.reshape(cast_10, full_int_array_0), None

        # pd_op.reshape: (950x2xf32, 0x25x38x2xi64) <- (25x38x2xf32, 2xi64)
        reshape_10, reshape_11 = paddle.reshape(cast_11, full_int_array_1), None

        # pd_op.full: (950x1xf32) <- ()
        full_13 = paddle._C_ops.full([950, 1], 32, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], 19, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (19xi64) <- (1xf32, 1xf32, 1xf32)
        arange_6 = paddle.arange(full_0, full_14, full_2, dtype=paddle.int64)

        # pd_op.cast: (19xf32) <- (19xi64)
        cast_12 = paddle._C_ops.cast(arange_6, paddle.float32)

        # pd_op.scale: (19xf32) <- (19xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(cast_12, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full([1], 64, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (19xf32) <- (19xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(scale_24, full_15, 0, True)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full([1], 13, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (13xi64) <- (1xf32, 1xf32, 1xf32)
        arange_7 = paddle.arange(full_0, full_16, full_2, dtype=paddle.int64)

        # pd_op.cast: (13xf32) <- (13xi64)
        cast_13 = paddle._C_ops.cast(arange_7, paddle.float32)

        # pd_op.scale: (13xf32) <- (13xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(cast_13, full_2, 0.5, True)

        # pd_op.scale: (13xf32) <- (13xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(scale_26, full_15, 0, True)

        # builtin.combine: ([13xf32, 19xf32]) <- (13xf32, 19xf32)
        combine_9 = [scale_27, scale_25]

        # pd_op.meshgrid: ([13x19xf32, 13x19xf32]) <- ([13xf32, 19xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_9)

        # builtin.split: (13x19xf32, 13x19xf32) <- ([13x19xf32, 13x19xf32])
        split_6, split_7, = meshgrid_3

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(split_7, full_2, -256, True)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(split_6, full_2, -256, True)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(split_7, full_2, 256, True)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(split_6, full_2, 256, True)

        # builtin.combine: ([13x19xf32, 13x19xf32, 13x19xf32, 13x19xf32]) <- (13x19xf32, 13x19xf32, 13x19xf32, 13x19xf32)
        combine_10 = [scale_28, scale_29, scale_30, scale_31]

        # pd_op.stack: (13x19x4xf32) <- ([13x19xf32, 13x19xf32, 13x19xf32, 13x19xf32])
        stack_6 = paddle._C_ops.stack(combine_10, -1)

        # pd_op.cast: (13x19x4xf32) <- (13x19x4xf32)
        cast_14 = paddle._C_ops.cast(stack_6, paddle.float32)

        # builtin.combine: ([13x19xf32, 13x19xf32]) <- (13x19xf32, 13x19xf32)
        combine_11 = [split_7, split_6]

        # pd_op.stack: (13x19x2xf32) <- ([13x19xf32, 13x19xf32])
        stack_7 = paddle._C_ops.stack(combine_11, -1)

        # pd_op.cast: (13x19x2xf32) <- (13x19x2xf32)
        cast_15 = paddle._C_ops.cast(stack_7, paddle.float32)

        # pd_op.reshape: (247x4xf32, 0x13x19x4xi64) <- (13x19x4xf32, 2xi64)
        reshape_12, reshape_13 = paddle.reshape(cast_14, full_int_array_0), None

        # pd_op.reshape: (247x2xf32, 0x13x19x2xi64) <- (13x19x2xf32, 2xi64)
        reshape_14, reshape_15 = paddle.reshape(cast_15, full_int_array_1), None

        # pd_op.full: (247x1xf32) <- ()
        full_17 = paddle._C_ops.full([247, 1], 64, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full([1], 10, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (10xi64) <- (1xf32, 1xf32, 1xf32)
        arange_8 = paddle.arange(full_0, full_18, full_2, dtype=paddle.int64)

        # pd_op.cast: (10xf32) <- (10xi64)
        cast_16 = paddle._C_ops.cast(arange_8, paddle.float32)

        # pd_op.scale: (10xf32) <- (10xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(cast_16, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full([1], 128, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (10xf32) <- (10xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(scale_32, full_19, 0, True)

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full([1], 7, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (7xi64) <- (1xf32, 1xf32, 1xf32)
        arange_9 = paddle.arange(full_0, full_20, full_2, dtype=paddle.int64)

        # pd_op.cast: (7xf32) <- (7xi64)
        cast_17 = paddle._C_ops.cast(arange_9, paddle.float32)

        # pd_op.scale: (7xf32) <- (7xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(cast_17, full_2, 0.5, True)

        # pd_op.scale: (7xf32) <- (7xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(scale_34, full_19, 0, True)

        # builtin.combine: ([7xf32, 10xf32]) <- (7xf32, 10xf32)
        combine_12 = [scale_35, scale_33]

        # pd_op.meshgrid: ([7x10xf32, 7x10xf32]) <- ([7xf32, 10xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_12)

        # builtin.split: (7x10xf32, 7x10xf32) <- ([7x10xf32, 7x10xf32])
        split_8, split_9, = meshgrid_4

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(split_9, full_2, -512, True)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(split_8, full_2, -512, True)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(split_9, full_2, 512, True)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(split_8, full_2, 512, True)

        # builtin.combine: ([7x10xf32, 7x10xf32, 7x10xf32, 7x10xf32]) <- (7x10xf32, 7x10xf32, 7x10xf32, 7x10xf32)
        combine_13 = [scale_36, scale_37, scale_38, scale_39]

        # pd_op.stack: (7x10x4xf32) <- ([7x10xf32, 7x10xf32, 7x10xf32, 7x10xf32])
        stack_8 = paddle._C_ops.stack(combine_13, -1)

        # pd_op.cast: (7x10x4xf32) <- (7x10x4xf32)
        cast_18 = paddle._C_ops.cast(stack_8, paddle.float32)

        # builtin.combine: ([7x10xf32, 7x10xf32]) <- (7x10xf32, 7x10xf32)
        combine_14 = [split_9, split_8]

        # pd_op.stack: (7x10x2xf32) <- ([7x10xf32, 7x10xf32])
        stack_9 = paddle._C_ops.stack(combine_14, -1)

        # pd_op.cast: (7x10x2xf32) <- (7x10x2xf32)
        cast_19 = paddle._C_ops.cast(stack_9, paddle.float32)

        # pd_op.reshape: (70x4xf32, 0x7x10x4xi64) <- (7x10x4xf32, 2xi64)
        reshape_16, reshape_17 = paddle.reshape(cast_18, full_int_array_0), None

        # pd_op.reshape: (70x2xf32, 0x7x10x2xi64) <- (7x10x2xf32, 2xi64)
        reshape_18, reshape_19 = paddle.reshape(cast_19, full_int_array_1), None

        # pd_op.full: (70x1xf32) <- ()
        full_21 = paddle._C_ops.full([70, 1], 128, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([15200x4xf32, 3800x4xf32, 950x4xf32, 247x4xf32, 70x4xf32]) <- (15200x4xf32, 3800x4xf32, 950x4xf32, 247x4xf32, 70x4xf32)
        combine_15 = [reshape_0, reshape_4, reshape_8, reshape_12, reshape_16]

        # pd_op.concat: (20267x4xf32) <- ([15200x4xf32, 3800x4xf32, 950x4xf32, 247x4xf32, 70x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_15, full_22)

        # builtin.combine: ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32]) <- (15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32)
        combine_16 = [reshape_2, reshape_6, reshape_10, reshape_14, reshape_18]

        # pd_op.concat: (20267x2xf32) <- ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_16, full_22)

        # builtin.combine: ([15200x1xf32, 3800x1xf32, 950x1xf32, 247x1xf32, 70x1xf32]) <- (15200x1xf32, 3800x1xf32, 950x1xf32, 247x1xf32, 70x1xf32)
        combine_17 = [full_5, full_9, full_13, full_17, full_21]

        # pd_op.concat: (20267x1xf32) <- ([15200x1xf32, 3800x1xf32, 950x1xf32, 247x1xf32, 70x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_17, full_22)

        # pd_op.divide: (20267x2xf32) <- (20267x2xf32, 20267x1xf32)
        divide_0 = concat_1 / concat_2

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_2 = [15200, 3800, 950, 247, 70]

        # pd_op.split: ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32]) <- (20267x2xf32, 5xi64, 1xi32)
        split_10 = paddle.split(divide_0, full_int_array_2, full_22)

        # builtin.split: (15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32) <- ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32])
        split_11, split_12, split_13, split_14, split_15, = split_10
        return divide_0, concat_0, reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, concat_2



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

class Block_builtin_module_0_0_0(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, ):
        args = []
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_full_0,
            self.op_full_1,
            self.op_full_2,
            self.op_arange_0,
            self.op_cast_0,
            self.op_scale_0,
            self.op_full_3,
            self.op_scale_1,
            self.op_full_4,
            self.op_arange_1,
            self.op_cast_1,
            self.op_scale_2,
            self.op_scale_3,
            self.op_combine_0,
            self.op_meshgrid_0,
            self.op_split_0,
            self.op_scale_4,
            self.op_scale_5,
            self.op_scale_6,
            self.op_scale_7,
            self.op_combine_1,
            self.op_stack_0,
            self.op_cast_2,
            self.op_combine_2,
            self.op_stack_1,
            self.op_cast_3,
            self.op_full_int_array_0,
            self.op_reshape_0,
            self.op_full_int_array_1,
            self.op_reshape_1,
            self.op_full_5,
            self.op_full_6,
            self.op_arange_2,
            self.op_cast_4,
            self.op_scale_8,
            self.op_full_7,
            self.op_scale_9,
            self.op_full_8,
            self.op_arange_3,
            self.op_cast_5,
            self.op_scale_10,
            self.op_scale_11,
            self.op_combine_3,
            self.op_meshgrid_1,
            self.op_split_1,
            self.op_scale_12,
            self.op_scale_13,
            self.op_scale_14,
            self.op_scale_15,
            self.op_combine_4,
            self.op_stack_2,
            self.op_cast_6,
            self.op_combine_5,
            self.op_stack_3,
            self.op_cast_7,
            self.op_reshape_2,
            self.op_reshape_3,
            self.op_full_9,
            self.op_full_10,
            self.op_arange_4,
            self.op_cast_8,
            self.op_scale_16,
            self.op_full_11,
            self.op_scale_17,
            self.op_full_12,
            self.op_arange_5,
            self.op_cast_9,
            self.op_scale_18,
            self.op_scale_19,
            self.op_combine_6,
            self.op_meshgrid_2,
            self.op_split_2,
            self.op_scale_20,
            self.op_scale_21,
            self.op_scale_22,
            self.op_scale_23,
            self.op_combine_7,
            self.op_stack_4,
            self.op_cast_10,
            self.op_combine_8,
            self.op_stack_5,
            self.op_cast_11,
            self.op_reshape_4,
            self.op_reshape_5,
            self.op_full_13,
            self.op_full_14,
            self.op_arange_6,
            self.op_cast_12,
            self.op_scale_24,
            self.op_full_15,
            self.op_scale_25,
            self.op_full_16,
            self.op_arange_7,
            self.op_cast_13,
            self.op_scale_26,
            self.op_scale_27,
            self.op_combine_9,
            self.op_meshgrid_3,
            self.op_split_3,
            self.op_scale_28,
            self.op_scale_29,
            self.op_scale_30,
            self.op_scale_31,
            self.op_combine_10,
            self.op_stack_6,
            self.op_cast_14,
            self.op_combine_11,
            self.op_stack_7,
            self.op_cast_15,
            self.op_reshape_6,
            self.op_reshape_7,
            self.op_full_17,
            self.op_full_18,
            self.op_arange_8,
            self.op_cast_16,
            self.op_scale_32,
            self.op_full_19,
            self.op_scale_33,
            self.op_full_20,
            self.op_arange_9,
            self.op_cast_17,
            self.op_scale_34,
            self.op_scale_35,
            self.op_combine_12,
            self.op_meshgrid_4,
            self.op_split_4,
            self.op_scale_36,
            self.op_scale_37,
            self.op_scale_38,
            self.op_scale_39,
            self.op_combine_13,
            self.op_stack_8,
            self.op_cast_18,
            self.op_combine_14,
            self.op_stack_9,
            self.op_cast_19,
            self.op_reshape_8,
            self.op_reshape_9,
            self.op_full_21,
            self.op_full_22,
            self.op_combine_15,
            self.op_concat_0,
            self.op_combine_16,
            self.op_concat_1,
            self.op_combine_17,
            self.op_concat_2,
            self.op_divide_0,
            self.op_full_int_array_2,
            self.op_split_5,
            self.op_split_6,
        ]

    def op_full_0(self, ):
    
        # EarlyReturn(0, 0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        return [full_0]

    def op_full_1(self, full_0):
    
        # EarlyReturn(0, 1)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 152, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_1]

    def op_full_2(self, full_0, full_1):
    
        # EarlyReturn(0, 2)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_1, full_2]

    def op_arange_0(self, full_0, full_1, full_2):
    
        # EarlyReturn(0, 3)

        # pd_op.arange: (152xi64) <- (1xf32, 1xf32, 1xf32)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype=paddle.int64)

        return [full_0, full_2, arange_0]

    def op_cast_0(self, full_0, full_2, arange_0):
    
        # EarlyReturn(0, 4)

        # pd_op.cast: (152xf32) <- (152xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        return [full_0, full_2, cast_0]

    def op_scale_0(self, full_0, full_2, cast_0):
    
        # EarlyReturn(0, 5)

        # pd_op.scale: (152xf32) <- (152xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_2, 0.5, True)

        return [full_0, full_2, scale_0]

    def op_full_3(self, full_0, full_2, scale_0):
    
        # EarlyReturn(0, 6)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 8, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, scale_0, full_3]

    def op_scale_1(self, full_0, full_2, scale_0, full_3):
    
        # EarlyReturn(0, 7)

        # pd_op.scale: (152xf32) <- (152xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_3, 0, True)

        return [full_0, full_2, full_3, scale_1]

    def op_full_4(self, full_0, full_2, full_3, scale_1):
    
        # EarlyReturn(0, 8)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 100, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_3, scale_1, full_4]

    def op_arange_1(self, full_0, full_2, full_3, scale_1, full_4):
    
        # EarlyReturn(0, 9)

        # pd_op.arange: (100xi64) <- (1xf32, 1xf32, 1xf32)
        arange_1 = paddle.arange(full_0, full_4, full_2, dtype=paddle.int64)

        return [full_0, full_2, full_3, scale_1, arange_1]

    def op_cast_1(self, full_0, full_2, full_3, scale_1, arange_1):
    
        # EarlyReturn(0, 10)

        # pd_op.cast: (100xf32) <- (100xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        return [full_0, full_2, full_3, scale_1, cast_1]

    def op_scale_2(self, full_0, full_2, full_3, scale_1, cast_1):
    
        # EarlyReturn(0, 11)

        # pd_op.scale: (100xf32) <- (100xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_1, full_2, 0.5, True)

        return [full_0, full_2, full_3, scale_1, scale_2]

    def op_scale_3(self, full_0, full_2, full_3, scale_1, scale_2):
    
        # EarlyReturn(0, 12)

        # pd_op.scale: (100xf32) <- (100xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_3, 0, True)

        return [full_0, full_2, scale_1, scale_3]

    def op_combine_0(self, full_0, full_2, scale_1, scale_3):
    
        # EarlyReturn(0, 13)

        # builtin.combine: ([100xf32, 152xf32]) <- (100xf32, 152xf32)
        combine_0 = [scale_3, scale_1]

        return [full_0, full_2, combine_0]

    def op_meshgrid_0(self, full_0, full_2, combine_0):
    
        # EarlyReturn(0, 14)

        # pd_op.meshgrid: ([100x152xf32, 100x152xf32]) <- ([100xf32, 152xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        return [full_0, full_2, meshgrid_0]

    def op_split_0(self, full_0, full_2, meshgrid_0):
    
        # EarlyReturn(0, 15)

        # builtin.split: (100x152xf32, 100x152xf32) <- ([100x152xf32, 100x152xf32])
        split_0, split_1, = meshgrid_0

        return [full_0, full_2, split_0, split_1]

    def op_scale_4(self, full_0, full_2, split_0, split_1):
    
        # EarlyReturn(0, 16)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_2, -32, True)

        return [full_0, full_2, split_0, split_1, scale_4]

    def op_scale_5(self, full_0, full_2, split_0, split_1, scale_4):
    
        # EarlyReturn(0, 17)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_2, -32, True)

        return [full_0, full_2, split_0, split_1, scale_4, scale_5]

    def op_scale_6(self, full_0, full_2, split_0, split_1, scale_4, scale_5):
    
        # EarlyReturn(0, 18)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(split_1, full_2, 32, True)

        return [full_0, full_2, split_0, split_1, scale_4, scale_5, scale_6]

    def op_scale_7(self, full_0, full_2, split_0, split_1, scale_4, scale_5, scale_6):
    
        # EarlyReturn(0, 19)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(split_0, full_2, 32, True)

        return [full_0, full_2, split_0, split_1, scale_4, scale_5, scale_6, scale_7]

    def op_combine_1(self, full_0, full_2, split_0, split_1, scale_4, scale_5, scale_6, scale_7):
    
        # EarlyReturn(0, 20)

        # builtin.combine: ([100x152xf32, 100x152xf32, 100x152xf32, 100x152xf32]) <- (100x152xf32, 100x152xf32, 100x152xf32, 100x152xf32)
        combine_1 = [scale_4, scale_5, scale_6, scale_7]

        return [full_0, full_2, split_0, split_1, combine_1]

    def op_stack_0(self, full_0, full_2, split_0, split_1, combine_1):
    
        # EarlyReturn(0, 21)

        # pd_op.stack: (100x152x4xf32) <- ([100x152xf32, 100x152xf32, 100x152xf32, 100x152xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        return [full_0, full_2, split_0, split_1, stack_0]

    def op_cast_2(self, full_0, full_2, split_0, split_1, stack_0):
    
        # EarlyReturn(0, 22)

        # pd_op.cast: (100x152x4xf32) <- (100x152x4xf32)
        cast_2 = paddle._C_ops.cast(stack_0, paddle.float32)

        return [full_0, full_2, split_0, split_1, cast_2]

    def op_combine_2(self, full_0, full_2, split_0, split_1, cast_2):
    
        # EarlyReturn(0, 23)

        # builtin.combine: ([100x152xf32, 100x152xf32]) <- (100x152xf32, 100x152xf32)
        combine_2 = [split_1, split_0]

        return [full_0, full_2, cast_2, combine_2]

    def op_stack_1(self, full_0, full_2, cast_2, combine_2):
    
        # EarlyReturn(0, 24)

        # pd_op.stack: (100x152x2xf32) <- ([100x152xf32, 100x152xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)

        return [full_0, full_2, cast_2, stack_1]

    def op_cast_3(self, full_0, full_2, cast_2, stack_1):
    
        # EarlyReturn(0, 25)

        # pd_op.cast: (100x152x2xf32) <- (100x152x2xf32)
        cast_3 = paddle._C_ops.cast(stack_1, paddle.float32)

        return [full_0, full_2, cast_2, cast_3]

    def op_full_int_array_0(self, full_0, full_2, cast_2, cast_3):
    
        # EarlyReturn(0, 26)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        return [full_0, full_2, cast_2, cast_3, full_int_array_0]

    def op_reshape_0(self, full_0, full_2, cast_2, cast_3, full_int_array_0):
    
        # EarlyReturn(0, 27)

        # pd_op.reshape: (15200x4xf32, 0x100x152x4xi64) <- (100x152x4xf32, 2xi64)
        reshape_0, reshape_1 = paddle.reshape(cast_2, full_int_array_0), None

        return [full_0, full_2, cast_3, full_int_array_0, reshape_0]

    def op_full_int_array_1(self, full_0, full_2, cast_3, full_int_array_0, reshape_0):
    
        # EarlyReturn(0, 28)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        return [full_0, full_2, cast_3, full_int_array_0, reshape_0, full_int_array_1]

    def op_reshape_1(self, full_0, full_2, cast_3, full_int_array_0, reshape_0, full_int_array_1):
    
        # EarlyReturn(0, 29)

        # pd_op.reshape: (15200x2xf32, 0x100x152x2xi64) <- (100x152x2xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(cast_3, full_int_array_1), None

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2]

    def op_full_5(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2):
    
        # EarlyReturn(0, 30)

        # pd_op.full: (15200x1xf32) <- ()
        full_5 = paddle._C_ops.full([15200, 1], 8, paddle.float32, paddle.framework._current_expected_place())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5]

    def op_full_6(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5):
    
        # EarlyReturn(0, 31)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], 76, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_6]

    def op_arange_2(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_6):
    
        # EarlyReturn(0, 32)

        # pd_op.arange: (76xi64) <- (1xf32, 1xf32, 1xf32)
        arange_2 = paddle.arange(full_0, full_6, full_2, dtype=paddle.int64)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, arange_2]

    def op_cast_4(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, arange_2):
    
        # EarlyReturn(0, 33)

        # pd_op.cast: (76xf32) <- (76xi64)
        cast_4 = paddle._C_ops.cast(arange_2, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_4]

    def op_scale_8(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_4):
    
        # EarlyReturn(0, 34)

        # pd_op.scale: (76xf32) <- (76xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_4, full_2, 0.5, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, scale_8]

    def op_full_7(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, scale_8):
    
        # EarlyReturn(0, 35)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], 16, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, scale_8, full_7]

    def op_scale_9(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, scale_8, full_7):
    
        # EarlyReturn(0, 36)

        # pd_op.scale: (76xf32) <- (76xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_7, 0, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9]

    def op_full_8(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9):
    
        # EarlyReturn(0, 37)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], 50, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9, full_8]

    def op_arange_3(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9, full_8):
    
        # EarlyReturn(0, 38)

        # pd_op.arange: (50xi64) <- (1xf32, 1xf32, 1xf32)
        arange_3 = paddle.arange(full_0, full_8, full_2, dtype=paddle.int64)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9, arange_3]

    def op_cast_5(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9, arange_3):
    
        # EarlyReturn(0, 39)

        # pd_op.cast: (50xf32) <- (50xi64)
        cast_5 = paddle._C_ops.cast(arange_3, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9, cast_5]

    def op_scale_10(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9, cast_5):
    
        # EarlyReturn(0, 40)

        # pd_op.scale: (50xf32) <- (50xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(cast_5, full_2, 0.5, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9, scale_10]

    def op_scale_11(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, full_7, scale_9, scale_10):
    
        # EarlyReturn(0, 41)

        # pd_op.scale: (50xf32) <- (50xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(scale_10, full_7, 0, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, scale_9, scale_11]

    def op_combine_3(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, scale_9, scale_11):
    
        # EarlyReturn(0, 42)

        # builtin.combine: ([50xf32, 76xf32]) <- (50xf32, 76xf32)
        combine_3 = [scale_11, scale_9]

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, combine_3]

    def op_meshgrid_1(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, combine_3):
    
        # EarlyReturn(0, 43)

        # pd_op.meshgrid: ([50x76xf32, 50x76xf32]) <- ([50xf32, 76xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, meshgrid_1]

    def op_split_1(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, meshgrid_1):
    
        # EarlyReturn(0, 44)

        # builtin.split: (50x76xf32, 50x76xf32) <- ([50x76xf32, 50x76xf32])
        split_2, split_3, = meshgrid_1

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3]

    def op_scale_12(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3):
    
        # EarlyReturn(0, 45)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(split_3, full_2, -64, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, scale_12]

    def op_scale_13(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, scale_12):
    
        # EarlyReturn(0, 46)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(split_2, full_2, -64, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, scale_12, scale_13]

    def op_scale_14(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, scale_12, scale_13):
    
        # EarlyReturn(0, 47)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_3, full_2, 64, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, scale_12, scale_13, scale_14]

    def op_scale_15(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, scale_12, scale_13, scale_14):
    
        # EarlyReturn(0, 48)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_2, full_2, 64, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, scale_12, scale_13, scale_14, scale_15]

    def op_combine_4(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, scale_12, scale_13, scale_14, scale_15):
    
        # EarlyReturn(0, 49)

        # builtin.combine: ([50x76xf32, 50x76xf32, 50x76xf32, 50x76xf32]) <- (50x76xf32, 50x76xf32, 50x76xf32, 50x76xf32)
        combine_4 = [scale_12, scale_13, scale_14, scale_15]

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, combine_4]

    def op_stack_2(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, combine_4):
    
        # EarlyReturn(0, 50)

        # pd_op.stack: (50x76x4xf32) <- ([50x76xf32, 50x76xf32, 50x76xf32, 50x76xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, stack_2]

    def op_cast_6(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, stack_2):
    
        # EarlyReturn(0, 51)

        # pd_op.cast: (50x76x4xf32) <- (50x76x4xf32)
        cast_6 = paddle._C_ops.cast(stack_2, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, cast_6]

    def op_combine_5(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, split_2, split_3, cast_6):
    
        # EarlyReturn(0, 52)

        # builtin.combine: ([50x76xf32, 50x76xf32]) <- (50x76xf32, 50x76xf32)
        combine_5 = [split_3, split_2]

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_6, combine_5]

    def op_stack_3(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_6, combine_5):
    
        # EarlyReturn(0, 53)

        # pd_op.stack: (50x76x2xf32) <- ([50x76xf32, 50x76xf32])
        stack_3 = paddle._C_ops.stack(combine_5, -1)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_6, stack_3]

    def op_cast_7(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_6, stack_3):
    
        # EarlyReturn(0, 54)

        # pd_op.cast: (50x76x2xf32) <- (50x76x2xf32)
        cast_7 = paddle._C_ops.cast(stack_3, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_6, cast_7]

    def op_reshape_2(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_6, cast_7):
    
        # EarlyReturn(0, 55)

        # pd_op.reshape: (3800x4xf32, 0x50x76x4xi64) <- (50x76x4xf32, 2xi64)
        reshape_4, reshape_5 = paddle.reshape(cast_6, full_int_array_0), None

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_7, reshape_4]

    def op_reshape_3(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, cast_7, reshape_4):
    
        # EarlyReturn(0, 56)

        # pd_op.reshape: (3800x2xf32, 0x50x76x2xi64) <- (50x76x2xf32, 2xi64)
        reshape_6, reshape_7 = paddle.reshape(cast_7, full_int_array_1), None

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6]

    def op_full_9(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6):
    
        # EarlyReturn(0, 57)

        # pd_op.full: (3800x1xf32) <- ()
        full_9 = paddle._C_ops.full([3800, 1], 16, paddle.float32, paddle.framework._current_expected_place())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9]

    def op_full_10(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9):
    
        # EarlyReturn(0, 58)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], 38, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_10]

    def op_arange_4(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_10):
    
        # EarlyReturn(0, 59)

        # pd_op.arange: (38xi64) <- (1xf32, 1xf32, 1xf32)
        arange_4 = paddle.arange(full_0, full_10, full_2, dtype=paddle.int64)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, arange_4]

    def op_cast_8(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, arange_4):
    
        # EarlyReturn(0, 60)

        # pd_op.cast: (38xf32) <- (38xi64)
        cast_8 = paddle._C_ops.cast(arange_4, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_8]

    def op_scale_16(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_8):
    
        # EarlyReturn(0, 61)

        # pd_op.scale: (38xf32) <- (38xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(cast_8, full_2, 0.5, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, scale_16]

    def op_full_11(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, scale_16):
    
        # EarlyReturn(0, 62)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], 32, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, scale_16, full_11]

    def op_scale_17(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, scale_16, full_11):
    
        # EarlyReturn(0, 63)

        # pd_op.scale: (38xf32) <- (38xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(scale_16, full_11, 0, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17]

    def op_full_12(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17):
    
        # EarlyReturn(0, 64)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], 25, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17, full_12]

    def op_arange_5(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17, full_12):
    
        # EarlyReturn(0, 65)

        # pd_op.arange: (25xi64) <- (1xf32, 1xf32, 1xf32)
        arange_5 = paddle.arange(full_0, full_12, full_2, dtype=paddle.int64)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17, arange_5]

    def op_cast_9(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17, arange_5):
    
        # EarlyReturn(0, 66)

        # pd_op.cast: (25xf32) <- (25xi64)
        cast_9 = paddle._C_ops.cast(arange_5, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17, cast_9]

    def op_scale_18(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17, cast_9):
    
        # EarlyReturn(0, 67)

        # pd_op.scale: (25xf32) <- (25xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_9, full_2, 0.5, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17, scale_18]

    def op_scale_19(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, full_11, scale_17, scale_18):
    
        # EarlyReturn(0, 68)

        # pd_op.scale: (25xf32) <- (25xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_11, 0, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, scale_17, scale_19]

    def op_combine_6(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, scale_17, scale_19):
    
        # EarlyReturn(0, 69)

        # builtin.combine: ([25xf32, 38xf32]) <- (25xf32, 38xf32)
        combine_6 = [scale_19, scale_17]

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, combine_6]

    def op_meshgrid_2(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, combine_6):
    
        # EarlyReturn(0, 70)

        # pd_op.meshgrid: ([25x38xf32, 25x38xf32]) <- ([25xf32, 38xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, meshgrid_2]

    def op_split_2(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, meshgrid_2):
    
        # EarlyReturn(0, 71)

        # builtin.split: (25x38xf32, 25x38xf32) <- ([25x38xf32, 25x38xf32])
        split_4, split_5, = meshgrid_2

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5]

    def op_scale_20(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5):
    
        # EarlyReturn(0, 72)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(split_5, full_2, -128, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, scale_20]

    def op_scale_21(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, scale_20):
    
        # EarlyReturn(0, 73)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(split_4, full_2, -128, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, scale_20, scale_21]

    def op_scale_22(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, scale_20, scale_21):
    
        # EarlyReturn(0, 74)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(split_5, full_2, 128, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, scale_20, scale_21, scale_22]

    def op_scale_23(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, scale_20, scale_21, scale_22):
    
        # EarlyReturn(0, 75)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(split_4, full_2, 128, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, scale_20, scale_21, scale_22, scale_23]

    def op_combine_7(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, scale_20, scale_21, scale_22, scale_23):
    
        # EarlyReturn(0, 76)

        # builtin.combine: ([25x38xf32, 25x38xf32, 25x38xf32, 25x38xf32]) <- (25x38xf32, 25x38xf32, 25x38xf32, 25x38xf32)
        combine_7 = [scale_20, scale_21, scale_22, scale_23]

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, combine_7]

    def op_stack_4(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, combine_7):
    
        # EarlyReturn(0, 77)

        # pd_op.stack: (25x38x4xf32) <- ([25x38xf32, 25x38xf32, 25x38xf32, 25x38xf32])
        stack_4 = paddle._C_ops.stack(combine_7, -1)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, stack_4]

    def op_cast_10(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, stack_4):
    
        # EarlyReturn(0, 78)

        # pd_op.cast: (25x38x4xf32) <- (25x38x4xf32)
        cast_10 = paddle._C_ops.cast(stack_4, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, cast_10]

    def op_combine_8(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, split_4, split_5, cast_10):
    
        # EarlyReturn(0, 79)

        # builtin.combine: ([25x38xf32, 25x38xf32]) <- (25x38xf32, 25x38xf32)
        combine_8 = [split_5, split_4]

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_10, combine_8]

    def op_stack_5(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_10, combine_8):
    
        # EarlyReturn(0, 80)

        # pd_op.stack: (25x38x2xf32) <- ([25x38xf32, 25x38xf32])
        stack_5 = paddle._C_ops.stack(combine_8, -1)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_10, stack_5]

    def op_cast_11(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_10, stack_5):
    
        # EarlyReturn(0, 81)

        # pd_op.cast: (25x38x2xf32) <- (25x38x2xf32)
        cast_11 = paddle._C_ops.cast(stack_5, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_10, cast_11]

    def op_reshape_4(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_10, cast_11):
    
        # EarlyReturn(0, 82)

        # pd_op.reshape: (950x4xf32, 0x25x38x4xi64) <- (25x38x4xf32, 2xi64)
        reshape_8, reshape_9 = paddle.reshape(cast_10, full_int_array_0), None

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_11, reshape_8]

    def op_reshape_5(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, cast_11, reshape_8):
    
        # EarlyReturn(0, 83)

        # pd_op.reshape: (950x2xf32, 0x25x38x2xi64) <- (25x38x2xf32, 2xi64)
        reshape_10, reshape_11 = paddle.reshape(cast_11, full_int_array_1), None

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10]

    def op_full_13(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10):
    
        # EarlyReturn(0, 84)

        # pd_op.full: (950x1xf32) <- ()
        full_13 = paddle._C_ops.full([950, 1], 32, paddle.float32, paddle.framework._current_expected_place())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13]

    def op_full_14(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13):
    
        # EarlyReturn(0, 85)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], 19, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_14]

    def op_arange_6(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_14):
    
        # EarlyReturn(0, 86)

        # pd_op.arange: (19xi64) <- (1xf32, 1xf32, 1xf32)
        arange_6 = paddle.arange(full_0, full_14, full_2, dtype=paddle.int64)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, arange_6]

    def op_cast_12(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, arange_6):
    
        # EarlyReturn(0, 87)

        # pd_op.cast: (19xf32) <- (19xi64)
        cast_12 = paddle._C_ops.cast(arange_6, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_12]

    def op_scale_24(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_12):
    
        # EarlyReturn(0, 88)

        # pd_op.scale: (19xf32) <- (19xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(cast_12, full_2, 0.5, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, scale_24]

    def op_full_15(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, scale_24):
    
        # EarlyReturn(0, 89)

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full([1], 64, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, scale_24, full_15]

    def op_scale_25(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, scale_24, full_15):
    
        # EarlyReturn(0, 90)

        # pd_op.scale: (19xf32) <- (19xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(scale_24, full_15, 0, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25]

    def op_full_16(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25):
    
        # EarlyReturn(0, 91)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full([1], 13, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25, full_16]

    def op_arange_7(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25, full_16):
    
        # EarlyReturn(0, 92)

        # pd_op.arange: (13xi64) <- (1xf32, 1xf32, 1xf32)
        arange_7 = paddle.arange(full_0, full_16, full_2, dtype=paddle.int64)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25, arange_7]

    def op_cast_13(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25, arange_7):
    
        # EarlyReturn(0, 93)

        # pd_op.cast: (13xf32) <- (13xi64)
        cast_13 = paddle._C_ops.cast(arange_7, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25, cast_13]

    def op_scale_26(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25, cast_13):
    
        # EarlyReturn(0, 94)

        # pd_op.scale: (13xf32) <- (13xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(cast_13, full_2, 0.5, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25, scale_26]

    def op_scale_27(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, full_15, scale_25, scale_26):
    
        # EarlyReturn(0, 95)

        # pd_op.scale: (13xf32) <- (13xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(scale_26, full_15, 0, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, scale_25, scale_27]

    def op_combine_9(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, scale_25, scale_27):
    
        # EarlyReturn(0, 96)

        # builtin.combine: ([13xf32, 19xf32]) <- (13xf32, 19xf32)
        combine_9 = [scale_27, scale_25]

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, combine_9]

    def op_meshgrid_3(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, combine_9):
    
        # EarlyReturn(0, 97)

        # pd_op.meshgrid: ([13x19xf32, 13x19xf32]) <- ([13xf32, 19xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_9)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, meshgrid_3]

    def op_split_3(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, meshgrid_3):
    
        # EarlyReturn(0, 98)

        # builtin.split: (13x19xf32, 13x19xf32) <- ([13x19xf32, 13x19xf32])
        split_6, split_7, = meshgrid_3

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7]

    def op_scale_28(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7):
    
        # EarlyReturn(0, 99)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(split_7, full_2, -256, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, scale_28]

    def op_scale_29(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, scale_28):
    
        # EarlyReturn(0, 100)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(split_6, full_2, -256, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, scale_28, scale_29]

    def op_scale_30(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, scale_28, scale_29):
    
        # EarlyReturn(0, 101)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(split_7, full_2, 256, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, scale_28, scale_29, scale_30]

    def op_scale_31(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, scale_28, scale_29, scale_30):
    
        # EarlyReturn(0, 102)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(split_6, full_2, 256, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, scale_28, scale_29, scale_30, scale_31]

    def op_combine_10(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, scale_28, scale_29, scale_30, scale_31):
    
        # EarlyReturn(0, 103)

        # builtin.combine: ([13x19xf32, 13x19xf32, 13x19xf32, 13x19xf32]) <- (13x19xf32, 13x19xf32, 13x19xf32, 13x19xf32)
        combine_10 = [scale_28, scale_29, scale_30, scale_31]

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, combine_10]

    def op_stack_6(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, combine_10):
    
        # EarlyReturn(0, 104)

        # pd_op.stack: (13x19x4xf32) <- ([13x19xf32, 13x19xf32, 13x19xf32, 13x19xf32])
        stack_6 = paddle._C_ops.stack(combine_10, -1)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, stack_6]

    def op_cast_14(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, stack_6):
    
        # EarlyReturn(0, 105)

        # pd_op.cast: (13x19x4xf32) <- (13x19x4xf32)
        cast_14 = paddle._C_ops.cast(stack_6, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, cast_14]

    def op_combine_11(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, split_6, split_7, cast_14):
    
        # EarlyReturn(0, 106)

        # builtin.combine: ([13x19xf32, 13x19xf32]) <- (13x19xf32, 13x19xf32)
        combine_11 = [split_7, split_6]

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_14, combine_11]

    def op_stack_7(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_14, combine_11):
    
        # EarlyReturn(0, 107)

        # pd_op.stack: (13x19x2xf32) <- ([13x19xf32, 13x19xf32])
        stack_7 = paddle._C_ops.stack(combine_11, -1)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_14, stack_7]

    def op_cast_15(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_14, stack_7):
    
        # EarlyReturn(0, 108)

        # pd_op.cast: (13x19x2xf32) <- (13x19x2xf32)
        cast_15 = paddle._C_ops.cast(stack_7, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_14, cast_15]

    def op_reshape_6(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_14, cast_15):
    
        # EarlyReturn(0, 109)

        # pd_op.reshape: (247x4xf32, 0x13x19x4xi64) <- (13x19x4xf32, 2xi64)
        reshape_12, reshape_13 = paddle.reshape(cast_14, full_int_array_0), None

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_15, reshape_12]

    def op_reshape_7(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, cast_15, reshape_12):
    
        # EarlyReturn(0, 110)

        # pd_op.reshape: (247x2xf32, 0x13x19x2xi64) <- (13x19x2xf32, 2xi64)
        reshape_14, reshape_15 = paddle.reshape(cast_15, full_int_array_1), None

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14]

    def op_full_17(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14):
    
        # EarlyReturn(0, 111)

        # pd_op.full: (247x1xf32) <- ()
        full_17 = paddle._C_ops.full([247, 1], 64, paddle.float32, paddle.framework._current_expected_place())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17]

    def op_full_18(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17):
    
        # EarlyReturn(0, 112)

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full([1], 10, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_18]

    def op_arange_8(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_18):
    
        # EarlyReturn(0, 113)

        # pd_op.arange: (10xi64) <- (1xf32, 1xf32, 1xf32)
        arange_8 = paddle.arange(full_0, full_18, full_2, dtype=paddle.int64)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, arange_8]

    def op_cast_16(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, arange_8):
    
        # EarlyReturn(0, 114)

        # pd_op.cast: (10xf32) <- (10xi64)
        cast_16 = paddle._C_ops.cast(arange_8, paddle.float32)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_16]

    def op_scale_32(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_16):
    
        # EarlyReturn(0, 115)

        # pd_op.scale: (10xf32) <- (10xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(cast_16, full_2, 0.5, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, scale_32]

    def op_full_19(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, scale_32):
    
        # EarlyReturn(0, 116)

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full([1], 128, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, scale_32, full_19]

    def op_scale_33(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, scale_32, full_19):
    
        # EarlyReturn(0, 117)

        # pd_op.scale: (10xf32) <- (10xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(scale_32, full_19, 0, True)

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33]

    def op_full_20(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33):
    
        # EarlyReturn(0, 118)

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full([1], 7, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33, full_20]

    def op_arange_9(self, full_0, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33, full_20):
    
        # EarlyReturn(0, 119)

        # pd_op.arange: (7xi64) <- (1xf32, 1xf32, 1xf32)
        arange_9 = paddle.arange(full_0, full_20, full_2, dtype=paddle.int64)

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33, arange_9]

    def op_cast_17(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33, arange_9):
    
        # EarlyReturn(0, 120)

        # pd_op.cast: (7xf32) <- (7xi64)
        cast_17 = paddle._C_ops.cast(arange_9, paddle.float32)

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33, cast_17]

    def op_scale_34(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33, cast_17):
    
        # EarlyReturn(0, 121)

        # pd_op.scale: (7xf32) <- (7xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(cast_17, full_2, 0.5, True)

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33, scale_34]

    def op_scale_35(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, full_19, scale_33, scale_34):
    
        # EarlyReturn(0, 122)

        # pd_op.scale: (7xf32) <- (7xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(scale_34, full_19, 0, True)

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, scale_33, scale_35]

    def op_combine_12(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, scale_33, scale_35):
    
        # EarlyReturn(0, 123)

        # builtin.combine: ([7xf32, 10xf32]) <- (7xf32, 10xf32)
        combine_12 = [scale_35, scale_33]

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, combine_12]

    def op_meshgrid_4(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, combine_12):
    
        # EarlyReturn(0, 124)

        # pd_op.meshgrid: ([7x10xf32, 7x10xf32]) <- ([7xf32, 10xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_12)

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, meshgrid_4]

    def op_split_4(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, meshgrid_4):
    
        # EarlyReturn(0, 125)

        # builtin.split: (7x10xf32, 7x10xf32) <- ([7x10xf32, 7x10xf32])
        split_8, split_9, = meshgrid_4

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9]

    def op_scale_36(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9):
    
        # EarlyReturn(0, 126)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(split_9, full_2, -512, True)

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, scale_36]

    def op_scale_37(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, scale_36):
    
        # EarlyReturn(0, 127)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(split_8, full_2, -512, True)

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, scale_36, scale_37]

    def op_scale_38(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, scale_36, scale_37):
    
        # EarlyReturn(0, 128)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(split_9, full_2, 512, True)

        return [full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, scale_36, scale_37, scale_38]

    def op_scale_39(self, full_2, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, scale_36, scale_37, scale_38):
    
        # EarlyReturn(0, 129)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(split_8, full_2, 512, True)

        return [full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, scale_36, scale_37, scale_38, scale_39]

    def op_combine_13(self, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, scale_36, scale_37, scale_38, scale_39):
    
        # EarlyReturn(0, 130)

        # builtin.combine: ([7x10xf32, 7x10xf32, 7x10xf32, 7x10xf32]) <- (7x10xf32, 7x10xf32, 7x10xf32, 7x10xf32)
        combine_13 = [scale_36, scale_37, scale_38, scale_39]

        return [full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, combine_13]

    def op_stack_8(self, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, combine_13):
    
        # EarlyReturn(0, 131)

        # pd_op.stack: (7x10x4xf32) <- ([7x10xf32, 7x10xf32, 7x10xf32, 7x10xf32])
        stack_8 = paddle._C_ops.stack(combine_13, -1)

        return [full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, stack_8]

    def op_cast_18(self, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, stack_8):
    
        # EarlyReturn(0, 132)

        # pd_op.cast: (7x10x4xf32) <- (7x10x4xf32)
        cast_18 = paddle._C_ops.cast(stack_8, paddle.float32)

        return [full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, cast_18]

    def op_combine_14(self, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, split_8, split_9, cast_18):
    
        # EarlyReturn(0, 133)

        # builtin.combine: ([7x10xf32, 7x10xf32]) <- (7x10xf32, 7x10xf32)
        combine_14 = [split_9, split_8]

        return [full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_18, combine_14]

    def op_stack_9(self, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_18, combine_14):
    
        # EarlyReturn(0, 134)

        # pd_op.stack: (7x10x2xf32) <- ([7x10xf32, 7x10xf32])
        stack_9 = paddle._C_ops.stack(combine_14, -1)

        return [full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_18, stack_9]

    def op_cast_19(self, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_18, stack_9):
    
        # EarlyReturn(0, 135)

        # pd_op.cast: (7x10x2xf32) <- (7x10x2xf32)
        cast_19 = paddle._C_ops.cast(stack_9, paddle.float32)

        return [full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_18, cast_19]

    def op_reshape_8(self, full_int_array_0, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_18, cast_19):
    
        # EarlyReturn(0, 136)

        # pd_op.reshape: (70x4xf32, 0x7x10x4xi64) <- (7x10x4xf32, 2xi64)
        reshape_16, reshape_17 = paddle.reshape(cast_18, full_int_array_0), None

        return [reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_19, reshape_16]

    def op_reshape_9(self, reshape_0, full_int_array_1, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, cast_19, reshape_16):
    
        # EarlyReturn(0, 137)

        # pd_op.reshape: (70x2xf32, 0x7x10x2xi64) <- (7x10x2xf32, 2xi64)
        reshape_18, reshape_19 = paddle.reshape(cast_19, full_int_array_1), None

        return [reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18]

    def op_full_21(self, reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18):
    
        # EarlyReturn(0, 138)

        # pd_op.full: (70x1xf32) <- ()
        full_21 = paddle._C_ops.full([70, 1], 128, paddle.float32, paddle.framework._current_expected_place())

        return [reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18, full_21]

    def op_full_22(self, reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18, full_21):
    
        # EarlyReturn(0, 139)

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        return [reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18, full_21, full_22]

    def op_combine_15(self, reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18, full_21, full_22):
    
        # EarlyReturn(0, 140)

        # builtin.combine: ([15200x4xf32, 3800x4xf32, 950x4xf32, 247x4xf32, 70x4xf32]) <- (15200x4xf32, 3800x4xf32, 950x4xf32, 247x4xf32, 70x4xf32)
        combine_15 = [reshape_0, reshape_4, reshape_8, reshape_12, reshape_16]

        return [reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18, full_21, full_22, combine_15]

    def op_concat_0(self, reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18, full_21, full_22, combine_15):
    
        # EarlyReturn(0, 141)

        # pd_op.concat: (20267x4xf32) <- ([15200x4xf32, 3800x4xf32, 950x4xf32, 247x4xf32, 70x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_15, full_22)

        return [reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18, full_21, full_22, concat_0]

    def op_combine_16(self, reshape_0, reshape_2, full_5, reshape_4, reshape_6, full_9, reshape_8, reshape_10, full_13, reshape_12, reshape_14, full_17, reshape_16, reshape_18, full_21, full_22, concat_0):
    
        # EarlyReturn(0, 142)

        # builtin.combine: ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32]) <- (15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32)
        combine_16 = [reshape_2, reshape_6, reshape_10, reshape_14, reshape_18]

        return [reshape_0, full_5, reshape_4, full_9, reshape_8, full_13, reshape_12, full_17, reshape_16, full_21, full_22, concat_0, combine_16]

    def op_concat_1(self, reshape_0, full_5, reshape_4, full_9, reshape_8, full_13, reshape_12, full_17, reshape_16, full_21, full_22, concat_0, combine_16):
    
        # EarlyReturn(0, 143)

        # pd_op.concat: (20267x2xf32) <- ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_16, full_22)

        return [reshape_0, full_5, reshape_4, full_9, reshape_8, full_13, reshape_12, full_17, reshape_16, full_21, full_22, concat_0, concat_1]

    def op_combine_17(self, reshape_0, full_5, reshape_4, full_9, reshape_8, full_13, reshape_12, full_17, reshape_16, full_21, full_22, concat_0, concat_1):
    
        # EarlyReturn(0, 144)

        # builtin.combine: ([15200x1xf32, 3800x1xf32, 950x1xf32, 247x1xf32, 70x1xf32]) <- (15200x1xf32, 3800x1xf32, 950x1xf32, 247x1xf32, 70x1xf32)
        combine_17 = [full_5, full_9, full_13, full_17, full_21]

        return [reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, full_22, concat_0, concat_1, combine_17]

    def op_concat_2(self, reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, full_22, concat_0, concat_1, combine_17):
    
        # EarlyReturn(0, 145)

        # pd_op.concat: (20267x1xf32) <- ([15200x1xf32, 3800x1xf32, 950x1xf32, 247x1xf32, 70x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_17, full_22)

        return [reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, full_22, concat_0, concat_1, concat_2]

    def op_divide_0(self, reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, full_22, concat_0, concat_1, concat_2):
    
        # EarlyReturn(0, 146)

        # pd_op.divide: (20267x2xf32) <- (20267x2xf32, 20267x1xf32)
        divide_0 = concat_1 / concat_2

        return [reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, full_22, concat_0, concat_2, divide_0]

    def op_full_int_array_2(self, reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, full_22, concat_0, concat_2, divide_0):
    
        # EarlyReturn(0, 147)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_2 = [15200, 3800, 950, 247, 70]

        return [reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, full_22, concat_0, concat_2, divide_0, full_int_array_2]

    def op_split_5(self, reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, full_22, concat_0, concat_2, divide_0, full_int_array_2):
    
        # EarlyReturn(0, 148)

        # pd_op.split: ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32]) <- (20267x2xf32, 5xi64, 1xi32)
        split_10 = paddle.split(divide_0, full_int_array_2, full_22)

        return [reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, concat_0, concat_2, divide_0, split_10]

    def op_split_6(self, reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, concat_0, concat_2, divide_0, split_10):
    
        # EarlyReturn(0, 149)

        # builtin.split: (15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32) <- ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32])
        split_11, split_12, split_13, split_14, split_15, = split_10

        return [divide_0, concat_0, reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, concat_2]

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = Block_builtin_module_0_0_0()
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
                # program paniced.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()