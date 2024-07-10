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
    return [208][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, parameter_0, data_0, data_2, data_1):

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [22, 49, 8, 16]

        # pd_op.reshape: (22x49x8x16xf32, 0x22x49x128xi64) <- (22x49x128xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(data_0, full_int_array_0), None

        # pd_op.transpose: (22x8x49x16xf32) <- (22x49x8x16xf32)
        transpose_0 = paddle.transpose(reshape_0, perm=[0, 2, 1, 3])

        # pd_op.transpose: (196x8xf32) <- (8x196xf32)
        transpose_1 = paddle.transpose(parameter_0, perm=[1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_1, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_18 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_20 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_25 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_26 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_27 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_28 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_29 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_30 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_31 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_32 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_33 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_34 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_35 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_36 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_37 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_38 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_39 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_40 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_41 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_42 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_43 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_44 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_45 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_46 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_47 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_48 = full_0

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(transpose_1, slice_0, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(data_1, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(transpose_1, slice_1, assign_48)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(data_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(transpose_1, slice_2, assign_47)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(data_1, [0], full_int_array_4, full_int_array_5, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(transpose_1, slice_3, assign_46)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [5]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(data_1, [0], full_int_array_5, full_int_array_6, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(transpose_1, slice_4, assign_45)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [6]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(data_1, [0], full_int_array_6, full_int_array_7, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(transpose_1, slice_5, assign_44)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [7]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(data_1, [0], full_int_array_7, full_int_array_8, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(transpose_1, slice_6, assign_43)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [8]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(data_1, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(transpose_1, slice_7, assign_42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [9]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(data_1, [0], full_int_array_9, full_int_array_10, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_8 = paddle._C_ops.gather(transpose_1, slice_8, assign_41)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [10]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(data_1, [0], full_int_array_10, full_int_array_11, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_9 = paddle._C_ops.gather(transpose_1, slice_9, assign_40)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [11]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(data_1, [0], full_int_array_11, full_int_array_12, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_10 = paddle._C_ops.gather(transpose_1, slice_10, assign_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [12]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(data_1, [0], full_int_array_12, full_int_array_13, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_11 = paddle._C_ops.gather(transpose_1, slice_11, assign_38)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [13]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(data_1, [0], full_int_array_13, full_int_array_14, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_12 = paddle._C_ops.gather(transpose_1, slice_12, assign_37)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [14]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(data_1, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_13 = paddle._C_ops.gather(transpose_1, slice_13, assign_36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [15]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(data_1, [0], full_int_array_15, full_int_array_16, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_14 = paddle._C_ops.gather(transpose_1, slice_14, assign_35)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [16]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(data_1, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_15 = paddle._C_ops.gather(transpose_1, slice_15, assign_34)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [17]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(data_1, [0], full_int_array_17, full_int_array_18, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_16 = paddle._C_ops.gather(transpose_1, slice_16, assign_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [18]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(data_1, [0], full_int_array_18, full_int_array_19, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_17 = paddle._C_ops.gather(transpose_1, slice_17, assign_32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [19]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(data_1, [0], full_int_array_19, full_int_array_20, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_18 = paddle._C_ops.gather(transpose_1, slice_18, assign_31)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [20]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(data_1, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_19 = paddle._C_ops.gather(transpose_1, slice_19, assign_30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [21]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(data_1, [0], full_int_array_21, full_int_array_22, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_20 = paddle._C_ops.gather(transpose_1, slice_20, assign_29)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [22]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(data_1, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_21 = paddle._C_ops.gather(transpose_1, slice_21, assign_28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [23]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(data_1, [0], full_int_array_23, full_int_array_24, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_22 = paddle._C_ops.gather(transpose_1, slice_22, assign_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [24]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(data_1, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_23 = paddle._C_ops.gather(transpose_1, slice_23, assign_26)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [25]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(data_1, [0], full_int_array_25, full_int_array_26, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_24 = paddle._C_ops.gather(transpose_1, slice_24, assign_25)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [26]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(data_1, [0], full_int_array_26, full_int_array_27, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_25 = paddle._C_ops.gather(transpose_1, slice_25, assign_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [27]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(data_1, [0], full_int_array_27, full_int_array_28, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_26 = paddle._C_ops.gather(transpose_1, slice_26, assign_23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [28]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(data_1, [0], full_int_array_28, full_int_array_29, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_27 = paddle._C_ops.gather(transpose_1, slice_27, assign_22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [29]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(data_1, [0], full_int_array_29, full_int_array_30, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_28 = paddle._C_ops.gather(transpose_1, slice_28, assign_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [30]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(data_1, [0], full_int_array_30, full_int_array_31, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_29 = paddle._C_ops.gather(transpose_1, slice_29, assign_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [31]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(data_1, [0], full_int_array_31, full_int_array_32, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_30 = paddle._C_ops.gather(transpose_1, slice_30, assign_19)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [32]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(data_1, [0], full_int_array_32, full_int_array_33, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_31 = paddle._C_ops.gather(transpose_1, slice_31, assign_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [33]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(data_1, [0], full_int_array_33, full_int_array_34, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_32 = paddle._C_ops.gather(transpose_1, slice_32, assign_17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [34]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(data_1, [0], full_int_array_34, full_int_array_35, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_33 = paddle._C_ops.gather(transpose_1, slice_33, assign_16)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [35]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(data_1, [0], full_int_array_35, full_int_array_36, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_34 = paddle._C_ops.gather(transpose_1, slice_34, assign_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [36]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(data_1, [0], full_int_array_36, full_int_array_37, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_35 = paddle._C_ops.gather(transpose_1, slice_35, assign_14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [37]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(data_1, [0], full_int_array_37, full_int_array_38, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_36 = paddle._C_ops.gather(transpose_1, slice_36, assign_13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [38]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(data_1, [0], full_int_array_38, full_int_array_39, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_37 = paddle._C_ops.gather(transpose_1, slice_37, assign_12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [39]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(data_1, [0], full_int_array_39, full_int_array_40, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_38 = paddle._C_ops.gather(transpose_1, slice_38, assign_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [40]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(data_1, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_39 = paddle._C_ops.gather(transpose_1, slice_39, assign_10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [41]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(data_1, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_40 = paddle._C_ops.gather(transpose_1, slice_40, assign_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [42]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(data_1, [0], full_int_array_42, full_int_array_43, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_41 = paddle._C_ops.gather(transpose_1, slice_41, assign_8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [43]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(data_1, [0], full_int_array_43, full_int_array_44, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_42 = paddle._C_ops.gather(transpose_1, slice_42, assign_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [44]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(data_1, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_43 = paddle._C_ops.gather(transpose_1, slice_43, assign_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [45]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(data_1, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_44 = paddle._C_ops.gather(transpose_1, slice_44, assign_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [46]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(data_1, [0], full_int_array_46, full_int_array_47, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_45 = paddle._C_ops.gather(transpose_1, slice_45, assign_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [47]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(data_1, [0], full_int_array_47, full_int_array_48, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_46 = paddle._C_ops.gather(transpose_1, slice_46, assign_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [48]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(data_1, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_47 = paddle._C_ops.gather(transpose_1, slice_47, assign_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [49]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(data_1, [0], full_int_array_49, full_int_array_50, [1], [0])

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_48 = paddle._C_ops.gather(transpose_1, slice_48, assign_1)

        # builtin.combine: ([196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32]) <- (196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32)
        combine_0 = [gather_0, gather_1, gather_2, gather_3, gather_4, gather_5, gather_6, gather_7, gather_8, gather_9, gather_10, gather_11, gather_12, gather_13, gather_14, gather_15, gather_16, gather_17, gather_18, gather_19, gather_20, gather_21, gather_22, gather_23, gather_24, gather_25, gather_26, gather_27, gather_28, gather_29, gather_30, gather_31, gather_32, gather_33, gather_34, gather_35, gather_36, gather_37, gather_38, gather_39, gather_40, gather_41, gather_42, gather_43, gather_44, gather_45, gather_46, gather_47, gather_48]

        # pd_op.concat: (9604x8xf32) <- ([196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, assign_0)

        # pd_op.transpose: (8x9604xf32) <- (9604x8xf32)
        transpose_2 = paddle.transpose(concat_0, perm=[1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_51 = [0, 49, 196]

        # pd_op.reshape: (8x49x196xf32, 0x8x9604xi64) <- (8x9604xf32, 3xi64)
        reshape_2, reshape_3 = paddle.reshape(transpose_2, full_int_array_51), None

        # pd_op.transpose: (22x8x16x196xf32) <- (22x8x196x16xf32)
        transpose_3 = paddle.transpose(data_2, perm=[0, 1, 3, 2])
        return reshape_1, transpose_1, slice_0, full_0, gather_0, slice_1, assign_48, gather_1, slice_2, assign_47, gather_2, slice_3, assign_46, gather_3, slice_4, assign_45, gather_4, slice_5, assign_44, gather_5, slice_6, assign_43, gather_6, slice_7, assign_42, gather_7, slice_8, assign_41, gather_8, slice_9, assign_40, gather_9, slice_10, assign_39, gather_10, slice_11, assign_38, gather_11, slice_12, assign_37, gather_12, slice_13, assign_36, gather_13, slice_14, assign_35, gather_14, slice_15, assign_34, gather_15, slice_16, assign_33, gather_16, slice_17, assign_32, gather_17, slice_18, assign_31, gather_18, slice_19, assign_30, gather_19, slice_20, assign_29, gather_20, slice_21, assign_28, gather_21, slice_22, assign_27, gather_22, slice_23, assign_26, gather_23, slice_24, assign_25, gather_24, slice_25, assign_24, gather_25, slice_26, assign_23, gather_26, slice_27, assign_22, gather_27, slice_28, assign_21, gather_28, slice_29, assign_20, gather_29, slice_30, assign_19, gather_30, slice_31, assign_18, gather_31, slice_32, assign_17, gather_32, slice_33, assign_16, gather_33, slice_34, assign_15, gather_34, slice_35, assign_14, gather_35, slice_36, assign_13, gather_36, slice_37, assign_12, gather_37, slice_38, assign_11, gather_38, slice_39, assign_10, gather_39, slice_40, assign_9, gather_40, slice_41, assign_8, gather_41, slice_42, assign_7, gather_42, slice_43, assign_6, gather_43, slice_44, assign_5, gather_44, slice_45, assign_4, gather_45, slice_46, assign_3, gather_46, slice_47, assign_2, gather_47, slice_48, assign_1, gather_48, assign_0, reshape_3, transpose_0, transpose_3, reshape_2



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

    def forward(self, parameter_0, data_0, data_2, data_1):
        args = [parameter_0, data_0, data_2, data_1]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_full_int_array_0,
            self.op_reshape_0,
            self.op_transpose_0,
            self.op_transpose_1,
            self.op_full_int_array_1,
            self.op_full_int_array_2,
            self.op_slice_0,
            self.op_full_0,
            self.op_assign_0,
            self.op_assign_1,
            self.op_assign_2,
            self.op_assign_3,
            self.op_assign_4,
            self.op_assign_5,
            self.op_assign_6,
            self.op_assign_7,
            self.op_assign_8,
            self.op_assign_9,
            self.op_assign_10,
            self.op_assign_11,
            self.op_assign_12,
            self.op_assign_13,
            self.op_assign_14,
            self.op_assign_15,
            self.op_assign_16,
            self.op_assign_17,
            self.op_assign_18,
            self.op_assign_19,
            self.op_assign_20,
            self.op_assign_21,
            self.op_assign_22,
            self.op_assign_23,
            self.op_assign_24,
            self.op_assign_25,
            self.op_assign_26,
            self.op_assign_27,
            self.op_assign_28,
            self.op_assign_29,
            self.op_assign_30,
            self.op_assign_31,
            self.op_assign_32,
            self.op_assign_33,
            self.op_assign_34,
            self.op_assign_35,
            self.op_assign_36,
            self.op_assign_37,
            self.op_assign_38,
            self.op_assign_39,
            self.op_assign_40,
            self.op_assign_41,
            self.op_assign_42,
            self.op_assign_43,
            self.op_assign_44,
            self.op_assign_45,
            self.op_assign_46,
            self.op_assign_47,
            self.op_assign_48,
            self.op_gather_0,
            self.op_full_int_array_3,
            self.op_slice_1,
            self.op_gather_1,
            self.op_full_int_array_4,
            self.op_slice_2,
            self.op_gather_2,
            self.op_full_int_array_5,
            self.op_slice_3,
            self.op_gather_3,
            self.op_full_int_array_6,
            self.op_slice_4,
            self.op_gather_4,
            self.op_full_int_array_7,
            self.op_slice_5,
            self.op_gather_5,
            self.op_full_int_array_8,
            self.op_slice_6,
            self.op_gather_6,
            self.op_full_int_array_9,
            self.op_slice_7,
            self.op_gather_7,
            self.op_full_int_array_10,
            self.op_slice_8,
            self.op_gather_8,
            self.op_full_int_array_11,
            self.op_slice_9,
            self.op_gather_9,
            self.op_full_int_array_12,
            self.op_slice_10,
            self.op_gather_10,
            self.op_full_int_array_13,
            self.op_slice_11,
            self.op_gather_11,
            self.op_full_int_array_14,
            self.op_slice_12,
            self.op_gather_12,
            self.op_full_int_array_15,
            self.op_slice_13,
            self.op_gather_13,
            self.op_full_int_array_16,
            self.op_slice_14,
            self.op_gather_14,
            self.op_full_int_array_17,
            self.op_slice_15,
            self.op_gather_15,
            self.op_full_int_array_18,
            self.op_slice_16,
            self.op_gather_16,
            self.op_full_int_array_19,
            self.op_slice_17,
            self.op_gather_17,
            self.op_full_int_array_20,
            self.op_slice_18,
            self.op_gather_18,
            self.op_full_int_array_21,
            self.op_slice_19,
            self.op_gather_19,
            self.op_full_int_array_22,
            self.op_slice_20,
            self.op_gather_20,
            self.op_full_int_array_23,
            self.op_slice_21,
            self.op_gather_21,
            self.op_full_int_array_24,
            self.op_slice_22,
            self.op_gather_22,
            self.op_full_int_array_25,
            self.op_slice_23,
            self.op_gather_23,
            self.op_full_int_array_26,
            self.op_slice_24,
            self.op_gather_24,
            self.op_full_int_array_27,
            self.op_slice_25,
            self.op_gather_25,
            self.op_full_int_array_28,
            self.op_slice_26,
            self.op_gather_26,
            self.op_full_int_array_29,
            self.op_slice_27,
            self.op_gather_27,
            self.op_full_int_array_30,
            self.op_slice_28,
            self.op_gather_28,
            self.op_full_int_array_31,
            self.op_slice_29,
            self.op_gather_29,
            self.op_full_int_array_32,
            self.op_slice_30,
            self.op_gather_30,
            self.op_full_int_array_33,
            self.op_slice_31,
            self.op_gather_31,
            self.op_full_int_array_34,
            self.op_slice_32,
            self.op_gather_32,
            self.op_full_int_array_35,
            self.op_slice_33,
            self.op_gather_33,
            self.op_full_int_array_36,
            self.op_slice_34,
            self.op_gather_34,
            self.op_full_int_array_37,
            self.op_slice_35,
            self.op_gather_35,
            self.op_full_int_array_38,
            self.op_slice_36,
            self.op_gather_36,
            self.op_full_int_array_39,
            self.op_slice_37,
            self.op_gather_37,
            self.op_full_int_array_40,
            self.op_slice_38,
            self.op_gather_38,
            self.op_full_int_array_41,
            self.op_slice_39,
            self.op_gather_39,
            self.op_full_int_array_42,
            self.op_slice_40,
            self.op_gather_40,
            self.op_full_int_array_43,
            self.op_slice_41,
            self.op_gather_41,
            self.op_full_int_array_44,
            self.op_slice_42,
            self.op_gather_42,
            self.op_full_int_array_45,
            self.op_slice_43,
            self.op_gather_43,
            self.op_full_int_array_46,
            self.op_slice_44,
            self.op_gather_44,
            self.op_full_int_array_47,
            self.op_slice_45,
            self.op_gather_45,
            self.op_full_int_array_48,
            self.op_slice_46,
            self.op_gather_46,
            self.op_full_int_array_49,
            self.op_slice_47,
            self.op_gather_47,
            self.op_full_int_array_50,
            self.op_slice_48,
            self.op_gather_48,
            self.op_combine_0,
            self.op_concat_0,
            self.op_transpose_2,
            self.op_full_int_array_51,
            self.op_reshape_1,
            self.op_transpose_3,
        ]

    def op_full_int_array_0(self, parameter_0, data_0, data_2, data_1):
    
        # EarlyReturn(0, 0)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [22, 49, 8, 16]

        return [parameter_0, data_0, data_2, data_1, full_int_array_0]

    def op_reshape_0(self, parameter_0, data_0, data_2, data_1, full_int_array_0):
    
        # EarlyReturn(0, 1)

        # pd_op.reshape: (22x49x8x16xf32, 0x22x49x128xi64) <- (22x49x128xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(data_0, full_int_array_0), None

        return [parameter_0, data_2, data_1, reshape_0, reshape_1]

    def op_transpose_0(self, parameter_0, data_2, data_1, reshape_0, reshape_1):
    
        # EarlyReturn(0, 2)

        # pd_op.transpose: (22x8x49x16xf32) <- (22x49x8x16xf32)
        transpose_0 = paddle.transpose(reshape_0, perm=[0, 2, 1, 3])

        return [parameter_0, data_2, data_1, reshape_1, transpose_0]

    def op_transpose_1(self, parameter_0, data_2, data_1, reshape_1, transpose_0):
    
        # EarlyReturn(0, 3)

        # pd_op.transpose: (196x8xf32) <- (8x196xf32)
        transpose_1 = paddle.transpose(parameter_0, perm=[1, 0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1]

    def op_full_int_array_1(self, data_2, data_1, reshape_1, transpose_0, transpose_1):
    
        # EarlyReturn(0, 4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_1]

    def op_full_int_array_2(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_1):
    
        # EarlyReturn(0, 5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_1, full_int_array_2]

    def op_slice_0(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_1, full_int_array_2):
    
        # EarlyReturn(0, 6)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_1, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0]

    def op_full_0(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0):
    
        # EarlyReturn(0, 7)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0]

    def op_assign_0(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0):
    
        # EarlyReturn(0, 8)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0]

    def op_assign_1(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0):
    
        # EarlyReturn(0, 9)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1]

    def op_assign_2(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1):
    
        # EarlyReturn(0, 10)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2]

    def op_assign_3(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2):
    
        # EarlyReturn(0, 11)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3]

    def op_assign_4(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3):
    
        # EarlyReturn(0, 12)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4]

    def op_assign_5(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4):
    
        # EarlyReturn(0, 13)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5]

    def op_assign_6(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5):
    
        # EarlyReturn(0, 14)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6]

    def op_assign_7(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6):
    
        # EarlyReturn(0, 15)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7]

    def op_assign_8(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7):
    
        # EarlyReturn(0, 16)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8]

    def op_assign_9(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8):
    
        # EarlyReturn(0, 17)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9]

    def op_assign_10(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9):
    
        # EarlyReturn(0, 18)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10]

    def op_assign_11(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10):
    
        # EarlyReturn(0, 19)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11]

    def op_assign_12(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11):
    
        # EarlyReturn(0, 20)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12]

    def op_assign_13(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12):
    
        # EarlyReturn(0, 21)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13]

    def op_assign_14(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13):
    
        # EarlyReturn(0, 22)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14]

    def op_assign_15(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14):
    
        # EarlyReturn(0, 23)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15]

    def op_assign_16(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15):
    
        # EarlyReturn(0, 24)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16]

    def op_assign_17(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16):
    
        # EarlyReturn(0, 25)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17]

    def op_assign_18(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17):
    
        # EarlyReturn(0, 26)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_18 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18]

    def op_assign_19(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18):
    
        # EarlyReturn(0, 27)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19]

    def op_assign_20(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19):
    
        # EarlyReturn(0, 28)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_20 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20]

    def op_assign_21(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20):
    
        # EarlyReturn(0, 29)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21]

    def op_assign_22(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21):
    
        # EarlyReturn(0, 30)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22]

    def op_assign_23(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22):
    
        # EarlyReturn(0, 31)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23]

    def op_assign_24(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23):
    
        # EarlyReturn(0, 32)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24]

    def op_assign_25(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24):
    
        # EarlyReturn(0, 33)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_25 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25]

    def op_assign_26(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25):
    
        # EarlyReturn(0, 34)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_26 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26]

    def op_assign_27(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26):
    
        # EarlyReturn(0, 35)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_27 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27]

    def op_assign_28(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27):
    
        # EarlyReturn(0, 36)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_28 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28]

    def op_assign_29(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28):
    
        # EarlyReturn(0, 37)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_29 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29]

    def op_assign_30(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29):
    
        # EarlyReturn(0, 38)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_30 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30]

    def op_assign_31(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30):
    
        # EarlyReturn(0, 39)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_31 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31]

    def op_assign_32(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31):
    
        # EarlyReturn(0, 40)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_32 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32]

    def op_assign_33(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32):
    
        # EarlyReturn(0, 41)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_33 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33]

    def op_assign_34(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33):
    
        # EarlyReturn(0, 42)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_34 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34]

    def op_assign_35(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34):
    
        # EarlyReturn(0, 43)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_35 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35]

    def op_assign_36(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35):
    
        # EarlyReturn(0, 44)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_36 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36]

    def op_assign_37(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36):
    
        # EarlyReturn(0, 45)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_37 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37]

    def op_assign_38(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37):
    
        # EarlyReturn(0, 46)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_38 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38]

    def op_assign_39(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38):
    
        # EarlyReturn(0, 47)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_39 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39]

    def op_assign_40(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39):
    
        # EarlyReturn(0, 48)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_40 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40]

    def op_assign_41(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40):
    
        # EarlyReturn(0, 49)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_41 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41]

    def op_assign_42(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41):
    
        # EarlyReturn(0, 50)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_42 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42]

    def op_assign_43(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42):
    
        # EarlyReturn(0, 51)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_43 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43]

    def op_assign_44(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43):
    
        # EarlyReturn(0, 52)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_44 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44]

    def op_assign_45(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44):
    
        # EarlyReturn(0, 53)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_45 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45]

    def op_assign_46(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45):
    
        # EarlyReturn(0, 54)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_46 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46]

    def op_assign_47(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46):
    
        # EarlyReturn(0, 55)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_47 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47]

    def op_assign_48(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47):
    
        # EarlyReturn(0, 56)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_48 = full_0

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48]

    def op_gather_0(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48):
    
        # EarlyReturn(0, 57)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(transpose_1, slice_0, full_0)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0]

    def op_full_int_array_3(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0):
    
        # EarlyReturn(0, 58)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, full_int_array_3]

    def op_slice_1(self, data_2, data_1, reshape_1, transpose_0, transpose_1, full_int_array_2, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, full_int_array_3):
    
        # EarlyReturn(0, 59)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(data_1, [0], full_int_array_2, full_int_array_3, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, full_int_array_3, slice_1]

    def op_gather_1(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, full_int_array_3, slice_1):
    
        # EarlyReturn(0, 60)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(transpose_1, slice_1, assign_48)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, full_int_array_3, slice_1, gather_1]

    def op_full_int_array_4(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, full_int_array_3, slice_1, gather_1):
    
        # EarlyReturn(0, 61)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, full_int_array_3, slice_1, gather_1, full_int_array_4]

    def op_slice_2(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, full_int_array_3, slice_1, gather_1, full_int_array_4):
    
        # EarlyReturn(0, 62)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(data_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, full_int_array_4, slice_2]

    def op_gather_2(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, full_int_array_4, slice_2):
    
        # EarlyReturn(0, 63)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(transpose_1, slice_2, assign_47)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, full_int_array_4, slice_2, gather_2]

    def op_full_int_array_5(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, full_int_array_4, slice_2, gather_2):
    
        # EarlyReturn(0, 64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, full_int_array_4, slice_2, gather_2, full_int_array_5]

    def op_slice_3(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, full_int_array_4, slice_2, gather_2, full_int_array_5):
    
        # EarlyReturn(0, 65)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(data_1, [0], full_int_array_4, full_int_array_5, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, full_int_array_5, slice_3]

    def op_gather_3(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, full_int_array_5, slice_3):
    
        # EarlyReturn(0, 66)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(transpose_1, slice_3, assign_46)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, full_int_array_5, slice_3, gather_3]

    def op_full_int_array_6(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, full_int_array_5, slice_3, gather_3):
    
        # EarlyReturn(0, 67)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [5]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, full_int_array_5, slice_3, gather_3, full_int_array_6]

    def op_slice_4(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, full_int_array_5, slice_3, gather_3, full_int_array_6):
    
        # EarlyReturn(0, 68)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(data_1, [0], full_int_array_5, full_int_array_6, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, full_int_array_6, slice_4]

    def op_gather_4(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, full_int_array_6, slice_4):
    
        # EarlyReturn(0, 69)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(transpose_1, slice_4, assign_45)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, full_int_array_6, slice_4, gather_4]

    def op_full_int_array_7(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, full_int_array_6, slice_4, gather_4):
    
        # EarlyReturn(0, 70)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [6]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, full_int_array_6, slice_4, gather_4, full_int_array_7]

    def op_slice_5(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, full_int_array_6, slice_4, gather_4, full_int_array_7):
    
        # EarlyReturn(0, 71)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(data_1, [0], full_int_array_6, full_int_array_7, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, full_int_array_7, slice_5]

    def op_gather_5(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, full_int_array_7, slice_5):
    
        # EarlyReturn(0, 72)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(transpose_1, slice_5, assign_44)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, full_int_array_7, slice_5, gather_5]

    def op_full_int_array_8(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, full_int_array_7, slice_5, gather_5):
    
        # EarlyReturn(0, 73)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [7]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, full_int_array_7, slice_5, gather_5, full_int_array_8]

    def op_slice_6(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, full_int_array_7, slice_5, gather_5, full_int_array_8):
    
        # EarlyReturn(0, 74)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(data_1, [0], full_int_array_7, full_int_array_8, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, full_int_array_8, slice_6]

    def op_gather_6(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, full_int_array_8, slice_6):
    
        # EarlyReturn(0, 75)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(transpose_1, slice_6, assign_43)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, full_int_array_8, slice_6, gather_6]

    def op_full_int_array_9(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, full_int_array_8, slice_6, gather_6):
    
        # EarlyReturn(0, 76)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [8]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, full_int_array_8, slice_6, gather_6, full_int_array_9]

    def op_slice_7(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, full_int_array_8, slice_6, gather_6, full_int_array_9):
    
        # EarlyReturn(0, 77)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(data_1, [0], full_int_array_8, full_int_array_9, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, full_int_array_9, slice_7]

    def op_gather_7(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, full_int_array_9, slice_7):
    
        # EarlyReturn(0, 78)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(transpose_1, slice_7, assign_42)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, full_int_array_9, slice_7, gather_7]

    def op_full_int_array_10(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, full_int_array_9, slice_7, gather_7):
    
        # EarlyReturn(0, 79)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [9]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, full_int_array_9, slice_7, gather_7, full_int_array_10]

    def op_slice_8(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, full_int_array_9, slice_7, gather_7, full_int_array_10):
    
        # EarlyReturn(0, 80)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(data_1, [0], full_int_array_9, full_int_array_10, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, full_int_array_10, slice_8]

    def op_gather_8(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, full_int_array_10, slice_8):
    
        # EarlyReturn(0, 81)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_8 = paddle._C_ops.gather(transpose_1, slice_8, assign_41)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, full_int_array_10, slice_8, gather_8]

    def op_full_int_array_11(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, full_int_array_10, slice_8, gather_8):
    
        # EarlyReturn(0, 82)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [10]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, full_int_array_10, slice_8, gather_8, full_int_array_11]

    def op_slice_9(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, full_int_array_10, slice_8, gather_8, full_int_array_11):
    
        # EarlyReturn(0, 83)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(data_1, [0], full_int_array_10, full_int_array_11, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, full_int_array_11, slice_9]

    def op_gather_9(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, full_int_array_11, slice_9):
    
        # EarlyReturn(0, 84)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_9 = paddle._C_ops.gather(transpose_1, slice_9, assign_40)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, full_int_array_11, slice_9, gather_9]

    def op_full_int_array_12(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, full_int_array_11, slice_9, gather_9):
    
        # EarlyReturn(0, 85)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [11]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, full_int_array_11, slice_9, gather_9, full_int_array_12]

    def op_slice_10(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, full_int_array_11, slice_9, gather_9, full_int_array_12):
    
        # EarlyReturn(0, 86)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(data_1, [0], full_int_array_11, full_int_array_12, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, full_int_array_12, slice_10]

    def op_gather_10(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, full_int_array_12, slice_10):
    
        # EarlyReturn(0, 87)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_10 = paddle._C_ops.gather(transpose_1, slice_10, assign_39)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, full_int_array_12, slice_10, gather_10]

    def op_full_int_array_13(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, full_int_array_12, slice_10, gather_10):
    
        # EarlyReturn(0, 88)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [12]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, full_int_array_12, slice_10, gather_10, full_int_array_13]

    def op_slice_11(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, full_int_array_12, slice_10, gather_10, full_int_array_13):
    
        # EarlyReturn(0, 89)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(data_1, [0], full_int_array_12, full_int_array_13, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, full_int_array_13, slice_11]

    def op_gather_11(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, full_int_array_13, slice_11):
    
        # EarlyReturn(0, 90)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_11 = paddle._C_ops.gather(transpose_1, slice_11, assign_38)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, full_int_array_13, slice_11, gather_11]

    def op_full_int_array_14(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, full_int_array_13, slice_11, gather_11):
    
        # EarlyReturn(0, 91)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [13]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, full_int_array_13, slice_11, gather_11, full_int_array_14]

    def op_slice_12(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, full_int_array_13, slice_11, gather_11, full_int_array_14):
    
        # EarlyReturn(0, 92)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(data_1, [0], full_int_array_13, full_int_array_14, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, full_int_array_14, slice_12]

    def op_gather_12(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, full_int_array_14, slice_12):
    
        # EarlyReturn(0, 93)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_12 = paddle._C_ops.gather(transpose_1, slice_12, assign_37)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, full_int_array_14, slice_12, gather_12]

    def op_full_int_array_15(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, full_int_array_14, slice_12, gather_12):
    
        # EarlyReturn(0, 94)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [14]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, full_int_array_14, slice_12, gather_12, full_int_array_15]

    def op_slice_13(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, full_int_array_14, slice_12, gather_12, full_int_array_15):
    
        # EarlyReturn(0, 95)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(data_1, [0], full_int_array_14, full_int_array_15, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, full_int_array_15, slice_13]

    def op_gather_13(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, full_int_array_15, slice_13):
    
        # EarlyReturn(0, 96)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_13 = paddle._C_ops.gather(transpose_1, slice_13, assign_36)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, full_int_array_15, slice_13, gather_13]

    def op_full_int_array_16(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, full_int_array_15, slice_13, gather_13):
    
        # EarlyReturn(0, 97)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [15]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, full_int_array_15, slice_13, gather_13, full_int_array_16]

    def op_slice_14(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, full_int_array_15, slice_13, gather_13, full_int_array_16):
    
        # EarlyReturn(0, 98)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(data_1, [0], full_int_array_15, full_int_array_16, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, full_int_array_16, slice_14]

    def op_gather_14(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, full_int_array_16, slice_14):
    
        # EarlyReturn(0, 99)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_14 = paddle._C_ops.gather(transpose_1, slice_14, assign_35)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, full_int_array_16, slice_14, gather_14]

    def op_full_int_array_17(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, full_int_array_16, slice_14, gather_14):
    
        # EarlyReturn(0, 100)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [16]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, full_int_array_16, slice_14, gather_14, full_int_array_17]

    def op_slice_15(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, full_int_array_16, slice_14, gather_14, full_int_array_17):
    
        # EarlyReturn(0, 101)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(data_1, [0], full_int_array_16, full_int_array_17, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, full_int_array_17, slice_15]

    def op_gather_15(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, full_int_array_17, slice_15):
    
        # EarlyReturn(0, 102)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_15 = paddle._C_ops.gather(transpose_1, slice_15, assign_34)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, full_int_array_17, slice_15, gather_15]

    def op_full_int_array_18(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, full_int_array_17, slice_15, gather_15):
    
        # EarlyReturn(0, 103)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [17]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, full_int_array_17, slice_15, gather_15, full_int_array_18]

    def op_slice_16(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, full_int_array_17, slice_15, gather_15, full_int_array_18):
    
        # EarlyReturn(0, 104)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(data_1, [0], full_int_array_17, full_int_array_18, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, full_int_array_18, slice_16]

    def op_gather_16(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, full_int_array_18, slice_16):
    
        # EarlyReturn(0, 105)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_16 = paddle._C_ops.gather(transpose_1, slice_16, assign_33)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, full_int_array_18, slice_16, gather_16]

    def op_full_int_array_19(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, full_int_array_18, slice_16, gather_16):
    
        # EarlyReturn(0, 106)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [18]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, full_int_array_18, slice_16, gather_16, full_int_array_19]

    def op_slice_17(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, full_int_array_18, slice_16, gather_16, full_int_array_19):
    
        # EarlyReturn(0, 107)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(data_1, [0], full_int_array_18, full_int_array_19, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, full_int_array_19, slice_17]

    def op_gather_17(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, full_int_array_19, slice_17):
    
        # EarlyReturn(0, 108)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_17 = paddle._C_ops.gather(transpose_1, slice_17, assign_32)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, full_int_array_19, slice_17, gather_17]

    def op_full_int_array_20(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, full_int_array_19, slice_17, gather_17):
    
        # EarlyReturn(0, 109)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [19]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, full_int_array_19, slice_17, gather_17, full_int_array_20]

    def op_slice_18(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, full_int_array_19, slice_17, gather_17, full_int_array_20):
    
        # EarlyReturn(0, 110)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(data_1, [0], full_int_array_19, full_int_array_20, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, full_int_array_20, slice_18]

    def op_gather_18(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, full_int_array_20, slice_18):
    
        # EarlyReturn(0, 111)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_18 = paddle._C_ops.gather(transpose_1, slice_18, assign_31)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, full_int_array_20, slice_18, gather_18]

    def op_full_int_array_21(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, full_int_array_20, slice_18, gather_18):
    
        # EarlyReturn(0, 112)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [20]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, full_int_array_20, slice_18, gather_18, full_int_array_21]

    def op_slice_19(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, full_int_array_20, slice_18, gather_18, full_int_array_21):
    
        # EarlyReturn(0, 113)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(data_1, [0], full_int_array_20, full_int_array_21, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, full_int_array_21, slice_19]

    def op_gather_19(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, full_int_array_21, slice_19):
    
        # EarlyReturn(0, 114)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_19 = paddle._C_ops.gather(transpose_1, slice_19, assign_30)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, full_int_array_21, slice_19, gather_19]

    def op_full_int_array_22(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, full_int_array_21, slice_19, gather_19):
    
        # EarlyReturn(0, 115)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [21]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, full_int_array_21, slice_19, gather_19, full_int_array_22]

    def op_slice_20(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, full_int_array_21, slice_19, gather_19, full_int_array_22):
    
        # EarlyReturn(0, 116)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(data_1, [0], full_int_array_21, full_int_array_22, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, full_int_array_22, slice_20]

    def op_gather_20(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, full_int_array_22, slice_20):
    
        # EarlyReturn(0, 117)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_20 = paddle._C_ops.gather(transpose_1, slice_20, assign_29)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, full_int_array_22, slice_20, gather_20]

    def op_full_int_array_23(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, full_int_array_22, slice_20, gather_20):
    
        # EarlyReturn(0, 118)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [22]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, full_int_array_22, slice_20, gather_20, full_int_array_23]

    def op_slice_21(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, full_int_array_22, slice_20, gather_20, full_int_array_23):
    
        # EarlyReturn(0, 119)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(data_1, [0], full_int_array_22, full_int_array_23, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, full_int_array_23, slice_21]

    def op_gather_21(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, full_int_array_23, slice_21):
    
        # EarlyReturn(0, 120)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_21 = paddle._C_ops.gather(transpose_1, slice_21, assign_28)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, full_int_array_23, slice_21, gather_21]

    def op_full_int_array_24(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, full_int_array_23, slice_21, gather_21):
    
        # EarlyReturn(0, 121)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [23]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, full_int_array_23, slice_21, gather_21, full_int_array_24]

    def op_slice_22(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, full_int_array_23, slice_21, gather_21, full_int_array_24):
    
        # EarlyReturn(0, 122)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(data_1, [0], full_int_array_23, full_int_array_24, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, full_int_array_24, slice_22]

    def op_gather_22(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, full_int_array_24, slice_22):
    
        # EarlyReturn(0, 123)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_22 = paddle._C_ops.gather(transpose_1, slice_22, assign_27)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, full_int_array_24, slice_22, gather_22]

    def op_full_int_array_25(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, full_int_array_24, slice_22, gather_22):
    
        # EarlyReturn(0, 124)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [24]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, full_int_array_24, slice_22, gather_22, full_int_array_25]

    def op_slice_23(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, full_int_array_24, slice_22, gather_22, full_int_array_25):
    
        # EarlyReturn(0, 125)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(data_1, [0], full_int_array_24, full_int_array_25, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, full_int_array_25, slice_23]

    def op_gather_23(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, full_int_array_25, slice_23):
    
        # EarlyReturn(0, 126)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_23 = paddle._C_ops.gather(transpose_1, slice_23, assign_26)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, full_int_array_25, slice_23, gather_23]

    def op_full_int_array_26(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, full_int_array_25, slice_23, gather_23):
    
        # EarlyReturn(0, 127)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [25]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, full_int_array_25, slice_23, gather_23, full_int_array_26]

    def op_slice_24(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, full_int_array_25, slice_23, gather_23, full_int_array_26):
    
        # EarlyReturn(0, 128)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(data_1, [0], full_int_array_25, full_int_array_26, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, full_int_array_26, slice_24]

    def op_gather_24(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, full_int_array_26, slice_24):
    
        # EarlyReturn(0, 129)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_24 = paddle._C_ops.gather(transpose_1, slice_24, assign_25)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, full_int_array_26, slice_24, gather_24]

    def op_full_int_array_27(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, full_int_array_26, slice_24, gather_24):
    
        # EarlyReturn(0, 130)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [26]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, full_int_array_26, slice_24, gather_24, full_int_array_27]

    def op_slice_25(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, full_int_array_26, slice_24, gather_24, full_int_array_27):
    
        # EarlyReturn(0, 131)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(data_1, [0], full_int_array_26, full_int_array_27, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, full_int_array_27, slice_25]

    def op_gather_25(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, full_int_array_27, slice_25):
    
        # EarlyReturn(0, 132)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_25 = paddle._C_ops.gather(transpose_1, slice_25, assign_24)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, full_int_array_27, slice_25, gather_25]

    def op_full_int_array_28(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, full_int_array_27, slice_25, gather_25):
    
        # EarlyReturn(0, 133)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [27]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, full_int_array_27, slice_25, gather_25, full_int_array_28]

    def op_slice_26(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, full_int_array_27, slice_25, gather_25, full_int_array_28):
    
        # EarlyReturn(0, 134)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(data_1, [0], full_int_array_27, full_int_array_28, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, full_int_array_28, slice_26]

    def op_gather_26(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, full_int_array_28, slice_26):
    
        # EarlyReturn(0, 135)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_26 = paddle._C_ops.gather(transpose_1, slice_26, assign_23)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, full_int_array_28, slice_26, gather_26]

    def op_full_int_array_29(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, full_int_array_28, slice_26, gather_26):
    
        # EarlyReturn(0, 136)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [28]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, full_int_array_28, slice_26, gather_26, full_int_array_29]

    def op_slice_27(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, full_int_array_28, slice_26, gather_26, full_int_array_29):
    
        # EarlyReturn(0, 137)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(data_1, [0], full_int_array_28, full_int_array_29, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, full_int_array_29, slice_27]

    def op_gather_27(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, full_int_array_29, slice_27):
    
        # EarlyReturn(0, 138)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_27 = paddle._C_ops.gather(transpose_1, slice_27, assign_22)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, full_int_array_29, slice_27, gather_27]

    def op_full_int_array_30(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, full_int_array_29, slice_27, gather_27):
    
        # EarlyReturn(0, 139)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [29]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, full_int_array_29, slice_27, gather_27, full_int_array_30]

    def op_slice_28(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, full_int_array_29, slice_27, gather_27, full_int_array_30):
    
        # EarlyReturn(0, 140)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(data_1, [0], full_int_array_29, full_int_array_30, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, full_int_array_30, slice_28]

    def op_gather_28(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, full_int_array_30, slice_28):
    
        # EarlyReturn(0, 141)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_28 = paddle._C_ops.gather(transpose_1, slice_28, assign_21)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, full_int_array_30, slice_28, gather_28]

    def op_full_int_array_31(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, full_int_array_30, slice_28, gather_28):
    
        # EarlyReturn(0, 142)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [30]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, full_int_array_30, slice_28, gather_28, full_int_array_31]

    def op_slice_29(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, full_int_array_30, slice_28, gather_28, full_int_array_31):
    
        # EarlyReturn(0, 143)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(data_1, [0], full_int_array_30, full_int_array_31, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, full_int_array_31, slice_29]

    def op_gather_29(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, full_int_array_31, slice_29):
    
        # EarlyReturn(0, 144)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_29 = paddle._C_ops.gather(transpose_1, slice_29, assign_20)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, full_int_array_31, slice_29, gather_29]

    def op_full_int_array_32(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, full_int_array_31, slice_29, gather_29):
    
        # EarlyReturn(0, 145)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [31]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, full_int_array_31, slice_29, gather_29, full_int_array_32]

    def op_slice_30(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, full_int_array_31, slice_29, gather_29, full_int_array_32):
    
        # EarlyReturn(0, 146)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(data_1, [0], full_int_array_31, full_int_array_32, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, full_int_array_32, slice_30]

    def op_gather_30(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, full_int_array_32, slice_30):
    
        # EarlyReturn(0, 147)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_30 = paddle._C_ops.gather(transpose_1, slice_30, assign_19)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, full_int_array_32, slice_30, gather_30]

    def op_full_int_array_33(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, full_int_array_32, slice_30, gather_30):
    
        # EarlyReturn(0, 148)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [32]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, full_int_array_32, slice_30, gather_30, full_int_array_33]

    def op_slice_31(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, full_int_array_32, slice_30, gather_30, full_int_array_33):
    
        # EarlyReturn(0, 149)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(data_1, [0], full_int_array_32, full_int_array_33, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, full_int_array_33, slice_31]

    def op_gather_31(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, full_int_array_33, slice_31):
    
        # EarlyReturn(0, 150)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_31 = paddle._C_ops.gather(transpose_1, slice_31, assign_18)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, full_int_array_33, slice_31, gather_31]

    def op_full_int_array_34(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, full_int_array_33, slice_31, gather_31):
    
        # EarlyReturn(0, 151)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [33]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, full_int_array_33, slice_31, gather_31, full_int_array_34]

    def op_slice_32(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, full_int_array_33, slice_31, gather_31, full_int_array_34):
    
        # EarlyReturn(0, 152)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(data_1, [0], full_int_array_33, full_int_array_34, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, full_int_array_34, slice_32]

    def op_gather_32(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, full_int_array_34, slice_32):
    
        # EarlyReturn(0, 153)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_32 = paddle._C_ops.gather(transpose_1, slice_32, assign_17)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, full_int_array_34, slice_32, gather_32]

    def op_full_int_array_35(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, full_int_array_34, slice_32, gather_32):
    
        # EarlyReturn(0, 154)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [34]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, full_int_array_34, slice_32, gather_32, full_int_array_35]

    def op_slice_33(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, full_int_array_34, slice_32, gather_32, full_int_array_35):
    
        # EarlyReturn(0, 155)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(data_1, [0], full_int_array_34, full_int_array_35, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, full_int_array_35, slice_33]

    def op_gather_33(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, full_int_array_35, slice_33):
    
        # EarlyReturn(0, 156)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_33 = paddle._C_ops.gather(transpose_1, slice_33, assign_16)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, full_int_array_35, slice_33, gather_33]

    def op_full_int_array_36(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, full_int_array_35, slice_33, gather_33):
    
        # EarlyReturn(0, 157)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [35]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, full_int_array_35, slice_33, gather_33, full_int_array_36]

    def op_slice_34(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, full_int_array_35, slice_33, gather_33, full_int_array_36):
    
        # EarlyReturn(0, 158)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(data_1, [0], full_int_array_35, full_int_array_36, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, full_int_array_36, slice_34]

    def op_gather_34(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, full_int_array_36, slice_34):
    
        # EarlyReturn(0, 159)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_34 = paddle._C_ops.gather(transpose_1, slice_34, assign_15)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, full_int_array_36, slice_34, gather_34]

    def op_full_int_array_37(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, full_int_array_36, slice_34, gather_34):
    
        # EarlyReturn(0, 160)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [36]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, full_int_array_36, slice_34, gather_34, full_int_array_37]

    def op_slice_35(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, full_int_array_36, slice_34, gather_34, full_int_array_37):
    
        # EarlyReturn(0, 161)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(data_1, [0], full_int_array_36, full_int_array_37, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, full_int_array_37, slice_35]

    def op_gather_35(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, full_int_array_37, slice_35):
    
        # EarlyReturn(0, 162)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_35 = paddle._C_ops.gather(transpose_1, slice_35, assign_14)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, full_int_array_37, slice_35, gather_35]

    def op_full_int_array_38(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, full_int_array_37, slice_35, gather_35):
    
        # EarlyReturn(0, 163)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [37]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, full_int_array_37, slice_35, gather_35, full_int_array_38]

    def op_slice_36(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, full_int_array_37, slice_35, gather_35, full_int_array_38):
    
        # EarlyReturn(0, 164)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(data_1, [0], full_int_array_37, full_int_array_38, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, full_int_array_38, slice_36]

    def op_gather_36(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, full_int_array_38, slice_36):
    
        # EarlyReturn(0, 165)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_36 = paddle._C_ops.gather(transpose_1, slice_36, assign_13)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, full_int_array_38, slice_36, gather_36]

    def op_full_int_array_39(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, full_int_array_38, slice_36, gather_36):
    
        # EarlyReturn(0, 166)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [38]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, full_int_array_38, slice_36, gather_36, full_int_array_39]

    def op_slice_37(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, full_int_array_38, slice_36, gather_36, full_int_array_39):
    
        # EarlyReturn(0, 167)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(data_1, [0], full_int_array_38, full_int_array_39, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, full_int_array_39, slice_37]

    def op_gather_37(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, full_int_array_39, slice_37):
    
        # EarlyReturn(0, 168)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_37 = paddle._C_ops.gather(transpose_1, slice_37, assign_12)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, full_int_array_39, slice_37, gather_37]

    def op_full_int_array_40(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, full_int_array_39, slice_37, gather_37):
    
        # EarlyReturn(0, 169)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [39]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, full_int_array_39, slice_37, gather_37, full_int_array_40]

    def op_slice_38(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, full_int_array_39, slice_37, gather_37, full_int_array_40):
    
        # EarlyReturn(0, 170)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(data_1, [0], full_int_array_39, full_int_array_40, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, full_int_array_40, slice_38]

    def op_gather_38(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, full_int_array_40, slice_38):
    
        # EarlyReturn(0, 171)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_38 = paddle._C_ops.gather(transpose_1, slice_38, assign_11)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, full_int_array_40, slice_38, gather_38]

    def op_full_int_array_41(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, full_int_array_40, slice_38, gather_38):
    
        # EarlyReturn(0, 172)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [40]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, full_int_array_40, slice_38, gather_38, full_int_array_41]

    def op_slice_39(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, full_int_array_40, slice_38, gather_38, full_int_array_41):
    
        # EarlyReturn(0, 173)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(data_1, [0], full_int_array_40, full_int_array_41, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, full_int_array_41, slice_39]

    def op_gather_39(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, full_int_array_41, slice_39):
    
        # EarlyReturn(0, 174)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_39 = paddle._C_ops.gather(transpose_1, slice_39, assign_10)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, full_int_array_41, slice_39, gather_39]

    def op_full_int_array_42(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, full_int_array_41, slice_39, gather_39):
    
        # EarlyReturn(0, 175)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [41]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, full_int_array_41, slice_39, gather_39, full_int_array_42]

    def op_slice_40(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, full_int_array_41, slice_39, gather_39, full_int_array_42):
    
        # EarlyReturn(0, 176)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(data_1, [0], full_int_array_41, full_int_array_42, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, full_int_array_42, slice_40]

    def op_gather_40(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, full_int_array_42, slice_40):
    
        # EarlyReturn(0, 177)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_40 = paddle._C_ops.gather(transpose_1, slice_40, assign_9)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, full_int_array_42, slice_40, gather_40]

    def op_full_int_array_43(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, full_int_array_42, slice_40, gather_40):
    
        # EarlyReturn(0, 178)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [42]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, full_int_array_42, slice_40, gather_40, full_int_array_43]

    def op_slice_41(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, full_int_array_42, slice_40, gather_40, full_int_array_43):
    
        # EarlyReturn(0, 179)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(data_1, [0], full_int_array_42, full_int_array_43, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, full_int_array_43, slice_41]

    def op_gather_41(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, full_int_array_43, slice_41):
    
        # EarlyReturn(0, 180)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_41 = paddle._C_ops.gather(transpose_1, slice_41, assign_8)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, full_int_array_43, slice_41, gather_41]

    def op_full_int_array_44(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, full_int_array_43, slice_41, gather_41):
    
        # EarlyReturn(0, 181)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [43]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, full_int_array_43, slice_41, gather_41, full_int_array_44]

    def op_slice_42(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, full_int_array_43, slice_41, gather_41, full_int_array_44):
    
        # EarlyReturn(0, 182)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(data_1, [0], full_int_array_43, full_int_array_44, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, full_int_array_44, slice_42]

    def op_gather_42(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, full_int_array_44, slice_42):
    
        # EarlyReturn(0, 183)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_42 = paddle._C_ops.gather(transpose_1, slice_42, assign_7)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, full_int_array_44, slice_42, gather_42]

    def op_full_int_array_45(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, full_int_array_44, slice_42, gather_42):
    
        # EarlyReturn(0, 184)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [44]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, full_int_array_44, slice_42, gather_42, full_int_array_45]

    def op_slice_43(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, full_int_array_44, slice_42, gather_42, full_int_array_45):
    
        # EarlyReturn(0, 185)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(data_1, [0], full_int_array_44, full_int_array_45, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, full_int_array_45, slice_43]

    def op_gather_43(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, full_int_array_45, slice_43):
    
        # EarlyReturn(0, 186)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_43 = paddle._C_ops.gather(transpose_1, slice_43, assign_6)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, full_int_array_45, slice_43, gather_43]

    def op_full_int_array_46(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, full_int_array_45, slice_43, gather_43):
    
        # EarlyReturn(0, 187)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [45]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, full_int_array_45, slice_43, gather_43, full_int_array_46]

    def op_slice_44(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, full_int_array_45, slice_43, gather_43, full_int_array_46):
    
        # EarlyReturn(0, 188)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(data_1, [0], full_int_array_45, full_int_array_46, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, full_int_array_46, slice_44]

    def op_gather_44(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, full_int_array_46, slice_44):
    
        # EarlyReturn(0, 189)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_44 = paddle._C_ops.gather(transpose_1, slice_44, assign_5)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, full_int_array_46, slice_44, gather_44]

    def op_full_int_array_47(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, full_int_array_46, slice_44, gather_44):
    
        # EarlyReturn(0, 190)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [46]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, full_int_array_46, slice_44, gather_44, full_int_array_47]

    def op_slice_45(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, full_int_array_46, slice_44, gather_44, full_int_array_47):
    
        # EarlyReturn(0, 191)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(data_1, [0], full_int_array_46, full_int_array_47, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, full_int_array_47, slice_45]

    def op_gather_45(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, full_int_array_47, slice_45):
    
        # EarlyReturn(0, 192)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_45 = paddle._C_ops.gather(transpose_1, slice_45, assign_4)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, full_int_array_47, slice_45, gather_45]

    def op_full_int_array_48(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, full_int_array_47, slice_45, gather_45):
    
        # EarlyReturn(0, 193)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [47]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, full_int_array_47, slice_45, gather_45, full_int_array_48]

    def op_slice_46(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, full_int_array_47, slice_45, gather_45, full_int_array_48):
    
        # EarlyReturn(0, 194)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(data_1, [0], full_int_array_47, full_int_array_48, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, full_int_array_48, slice_46]

    def op_gather_46(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, full_int_array_48, slice_46):
    
        # EarlyReturn(0, 195)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_46 = paddle._C_ops.gather(transpose_1, slice_46, assign_3)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, full_int_array_48, slice_46, gather_46]

    def op_full_int_array_49(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, full_int_array_48, slice_46, gather_46):
    
        # EarlyReturn(0, 196)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [48]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, full_int_array_48, slice_46, gather_46, full_int_array_49]

    def op_slice_47(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, full_int_array_48, slice_46, gather_46, full_int_array_49):
    
        # EarlyReturn(0, 197)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(data_1, [0], full_int_array_48, full_int_array_49, [1], [0])

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, full_int_array_49, slice_47]

    def op_gather_47(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, full_int_array_49, slice_47):
    
        # EarlyReturn(0, 198)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_47 = paddle._C_ops.gather(transpose_1, slice_47, assign_2)

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, full_int_array_49, slice_47, gather_47]

    def op_full_int_array_50(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, full_int_array_49, slice_47, gather_47):
    
        # EarlyReturn(0, 199)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [49]

        return [data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, full_int_array_49, slice_47, gather_47, full_int_array_50]

    def op_slice_48(self, data_2, data_1, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, full_int_array_49, slice_47, gather_47, full_int_array_50):
    
        # EarlyReturn(0, 200)

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(data_1, [0], full_int_array_49, full_int_array_50, [1], [0])

        return [data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48]

    def op_gather_48(self, data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48):
    
        # EarlyReturn(0, 201)

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_48 = paddle._C_ops.gather(transpose_1, slice_48, assign_1)

        return [data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48]

    def op_combine_0(self, data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48):
    
        # EarlyReturn(0, 202)

        # builtin.combine: ([196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32]) <- (196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32)
        combine_0 = [gather_0, gather_1, gather_2, gather_3, gather_4, gather_5, gather_6, gather_7, gather_8, gather_9, gather_10, gather_11, gather_12, gather_13, gather_14, gather_15, gather_16, gather_17, gather_18, gather_19, gather_20, gather_21, gather_22, gather_23, gather_24, gather_25, gather_26, gather_27, gather_28, gather_29, gather_30, gather_31, gather_32, gather_33, gather_34, gather_35, gather_36, gather_37, gather_38, gather_39, gather_40, gather_41, gather_42, gather_43, gather_44, gather_45, gather_46, gather_47, gather_48]

        return [data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, combine_0]

    def op_concat_0(self, data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, combine_0):
    
        # EarlyReturn(0, 203)

        # pd_op.concat: (9604x8xf32) <- ([196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, assign_0)

        return [data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, concat_0]

    def op_transpose_2(self, data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, concat_0):
    
        # EarlyReturn(0, 204)

        # pd_op.transpose: (8x9604xf32) <- (9604x8xf32)
        transpose_2 = paddle.transpose(concat_0, perm=[1, 0])

        return [data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, transpose_2]

    def op_full_int_array_51(self, data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, transpose_2):
    
        # EarlyReturn(0, 205)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_51 = [0, 49, 196]

        return [data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, transpose_2, full_int_array_51]

    def op_reshape_1(self, data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, transpose_2, full_int_array_51):
    
        # EarlyReturn(0, 206)

        # pd_op.reshape: (8x49x196xf32, 0x8x9604xi64) <- (8x9604xf32, 3xi64)
        reshape_2, reshape_3 = paddle.reshape(transpose_2, full_int_array_51), None

        return [data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, reshape_2, reshape_3]

    def op_transpose_3(self, data_2, reshape_1, transpose_0, transpose_1, slice_0, full_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, assign_7, assign_8, assign_9, assign_10, assign_11, assign_12, assign_13, assign_14, assign_15, assign_16, assign_17, assign_18, assign_19, assign_20, assign_21, assign_22, assign_23, assign_24, assign_25, assign_26, assign_27, assign_28, assign_29, assign_30, assign_31, assign_32, assign_33, assign_34, assign_35, assign_36, assign_37, assign_38, assign_39, assign_40, assign_41, assign_42, assign_43, assign_44, assign_45, assign_46, assign_47, assign_48, gather_0, slice_1, gather_1, slice_2, gather_2, slice_3, gather_3, slice_4, gather_4, slice_5, gather_5, slice_6, gather_6, slice_7, gather_7, slice_8, gather_8, slice_9, gather_9, slice_10, gather_10, slice_11, gather_11, slice_12, gather_12, slice_13, gather_13, slice_14, gather_14, slice_15, gather_15, slice_16, gather_16, slice_17, gather_17, slice_18, gather_18, slice_19, gather_19, slice_20, gather_20, slice_21, gather_21, slice_22, gather_22, slice_23, gather_23, slice_24, gather_24, slice_25, gather_25, slice_26, gather_26, slice_27, gather_27, slice_28, gather_28, slice_29, gather_29, slice_30, gather_30, slice_31, gather_31, slice_32, gather_32, slice_33, gather_33, slice_34, gather_34, slice_35, gather_35, slice_36, gather_36, slice_37, gather_37, slice_38, gather_38, slice_39, gather_39, slice_40, gather_40, slice_41, gather_41, slice_42, gather_42, slice_43, gather_43, slice_44, gather_44, slice_45, gather_45, slice_46, gather_46, slice_47, gather_47, slice_48, gather_48, reshape_2, reshape_3):
    
        # EarlyReturn(0, 207)

        # pd_op.transpose: (22x8x16x196xf32) <- (22x8x196x16xf32)
        transpose_3 = paddle.transpose(data_2, perm=[0, 1, 3, 2])

        return [reshape_1, transpose_1, slice_0, full_0, gather_0, slice_1, assign_48, gather_1, slice_2, assign_47, gather_2, slice_3, assign_46, gather_3, slice_4, assign_45, gather_4, slice_5, assign_44, gather_5, slice_6, assign_43, gather_6, slice_7, assign_42, gather_7, slice_8, assign_41, gather_8, slice_9, assign_40, gather_9, slice_10, assign_39, gather_10, slice_11, assign_38, gather_11, slice_12, assign_37, gather_12, slice_13, assign_36, gather_13, slice_14, assign_35, gather_14, slice_15, assign_34, gather_15, slice_16, assign_33, gather_16, slice_17, assign_32, gather_17, slice_18, assign_31, gather_18, slice_19, assign_30, gather_19, slice_20, assign_29, gather_20, slice_21, assign_28, gather_21, slice_22, assign_27, gather_22, slice_23, assign_26, gather_23, slice_24, assign_25, gather_24, slice_25, assign_24, gather_25, slice_26, assign_23, gather_26, slice_27, assign_22, gather_27, slice_28, assign_21, gather_28, slice_29, assign_20, gather_29, slice_30, assign_19, gather_30, slice_31, assign_18, gather_31, slice_32, assign_17, gather_32, slice_33, assign_16, gather_33, slice_34, assign_15, gather_34, slice_35, assign_14, gather_35, slice_36, assign_13, gather_36, slice_37, assign_12, gather_37, slice_38, assign_11, gather_38, slice_39, assign_10, gather_39, slice_40, assign_9, gather_40, slice_41, assign_8, gather_41, slice_42, assign_7, gather_42, slice_43, assign_6, gather_43, slice_44, assign_5, gather_44, slice_45, assign_4, gather_45, slice_46, assign_3, gather_46, slice_47, assign_2, gather_47, slice_48, assign_1, gather_48, assign_0, reshape_3, transpose_0, transpose_3, reshape_2]

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([8, 196], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([22, 49, 128], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([22, 8, 196, 16], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[8, 196], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[22, 49, 128], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[22, 8, 196, 16], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()