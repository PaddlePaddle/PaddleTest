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
    return [3, 0, 163, 0, 930][block_idx] - 1 # number-of-ops-in-block

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
    def pd_op_if_4962_0_0(self, reshape__0):

        # pd_op.flip: (1x1x2x180x320xf16) <- (1x1x2x180x320xf16)
        flip_0 = paddle._C_ops.flip(reshape__0, [1])

        # pd_op.cast: (1x1x2x180x320xf32) <- (1x1x2x180x320xf16)
        cast_0 = paddle._C_ops.cast(flip_0, paddle.float32)

        # pd_op.assign_: (1x1x2x180x320xf32) <- (1x1x2x180x320xf32)
        assign__0 = paddle._C_ops.assign_(cast_0)
        return assign__0
    def pd_op_if_4962_1_0(self, parameter_0):
        return parameter_0
    def pd_op_if_4969_0_0(self, pool2d_0, cast_1, pool2d_1, parameter_0, constant_0, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, constant_1, cast_3, constant_2, constant_3, constant_4, constant_5, constant_6, constant_7, pool2d_2, pool2d_3, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, cast_7, constant_8, constant_9, divide__0, divide__1, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, constant_10, constant_11):

        # pd_op.cast: (1x3x48x80xf32) <- (1x3x48x80xf16)
        cast_0 = paddle._C_ops.cast(pool2d_0, paddle.float32)

        # pd_op.grid_sample: (1x3x48x80xf32) <- (1x3x48x80xf32, 1x48x80x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(cast_0, cast_1, 'bilinear', 'border', True)

        # pd_op.cast: (1x3x48x80xf16) <- (1x3x48x80xf32)
        cast_2 = paddle._C_ops.cast(grid_sample_0, paddle.float16)

        # builtin.combine: ([1x3x48x80xf16, 1x3x48x80xf16, 1x2x48x80xf16]) <- (1x3x48x80xf16, 1x3x48x80xf16, 1x2x48x80xf16)
        combine_0 = [pool2d_1, cast_2, parameter_0]

        # pd_op.concat: (1x8x48x80xf16) <- ([1x3x48x80xf16, 1x3x48x80xf16, 1x2x48x80xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, constant_0)

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x8x48x80xf16, 16x8x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(concat_0, parameter_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, parameter_2)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__0 = paddle._C_ops.leaky_relu_(add__0, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x16x48x80xf16, 16x16x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(leaky_relu__0, parameter_3, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_1, parameter_4)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__1 = paddle._C_ops.leaky_relu_(add__1, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf16) <- (1x16x48x80xf16, 32x16x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(leaky_relu__1, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x48x80xf16) <- (1x32x48x80xf16, 1x32x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_2, parameter_6)

        # pd_op.leaky_relu_: (1x32x48x80xf16) <- (1x32x48x80xf16)
        leaky_relu__2 = paddle._C_ops.leaky_relu_(add__2, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf16) <- (1x32x48x80xf16, 32x32x3x3xf16)
        conv2d_3 = paddle._C_ops.conv2d(leaky_relu__2, parameter_7, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x48x80xf16) <- (1x32x48x80xf16, 1x32x1x1xf16)
        add__3 = paddle._C_ops.add_(conv2d_3, parameter_8)

        # pd_op.leaky_relu_: (1x32x48x80xf16) <- (1x32x48x80xf16)
        leaky_relu__3 = paddle._C_ops.leaky_relu_(add__3, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf16) <- (1x32x48x80xf16, 32x32x3x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(leaky_relu__3, parameter_9, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x48x80xf16) <- (1x32x48x80xf16, 1x32x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_4, parameter_10)

        # pd_op.leaky_relu_: (1x32x48x80xf16) <- (1x32x48x80xf16)
        leaky_relu__4 = paddle._C_ops.leaky_relu_(add__4, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf16) <- (1x32x48x80xf16, 32x32x3x3xf16)
        conv2d_5 = paddle._C_ops.conv2d(leaky_relu__4, parameter_11, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x48x80xf16) <- (1x32x48x80xf16, 1x32x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_5, parameter_12)

        # pd_op.leaky_relu_: (1x32x48x80xf16) <- (1x32x48x80xf16)
        leaky_relu__5 = paddle._C_ops.leaky_relu_(add__5, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x32x48x80xf16, 16x32x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(leaky_relu__5, parameter_13, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__6 = paddle._C_ops.add_(conv2d_6, parameter_14)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__6 = paddle._C_ops.leaky_relu_(add__6, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x16x48x80xf16, 16x16x3x3xf16)
        conv2d_7 = paddle._C_ops.conv2d(leaky_relu__6, parameter_15, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__7 = paddle._C_ops.add_(conv2d_7, parameter_16)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__7 = paddle._C_ops.leaky_relu_(add__7, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x16x48x80xf16, 16x16x3x3xf16)
        conv2d_8 = paddle._C_ops.conv2d(leaky_relu__7, parameter_17, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_8, parameter_18)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__8 = paddle._C_ops.leaky_relu_(add__8, float('0.1'))

        # pd_op.conv2d: (1x8x48x80xf16) <- (1x16x48x80xf16, 8x16x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(leaky_relu__8, parameter_19, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x48x80xf16) <- (1x8x48x80xf16, 1x8x1x1xf16)
        add__9 = paddle._C_ops.add_(conv2d_9, parameter_20)

        # pd_op.leaky_relu_: (1x8x48x80xf16) <- (1x8x48x80xf16)
        leaky_relu__9 = paddle._C_ops.leaky_relu_(add__9, float('0.1'))

        # pd_op.conv2d: (1x8x48x80xf16) <- (1x8x48x80xf16, 8x8x3x3xf16)
        conv2d_10 = paddle._C_ops.conv2d(leaky_relu__9, parameter_21, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x48x80xf16) <- (1x8x48x80xf16, 1x8x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_10, parameter_22)

        # pd_op.leaky_relu_: (1x8x48x80xf16) <- (1x8x48x80xf16)
        leaky_relu__10 = paddle._C_ops.leaky_relu_(add__10, float('0.1'))

        # pd_op.conv2d: (1x2x48x80xf16) <- (1x8x48x80xf16, 2x8x3x3xf16)
        conv2d_11 = paddle._C_ops.conv2d(leaky_relu__10, parameter_23, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x2x48x80xf16) <- (1x2x48x80xf16, 1x2x1x1xf16)
        add__11 = paddle._C_ops.add_(conv2d_11, parameter_24)

        # pd_op.add: (1x2x48x80xf16) <- (1x2x48x80xf16, 1x2x48x80xf16)
        add_0 = parameter_0 + add__11

        # pd_op.bilinear_interp: (1x2x96x160xf16) <- (1x2x48x80xf16, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(add_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

        # pd_op.scale_: (1x2x96x160xf16) <- (1x2x96x160xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(bilinear_interp_0, constant_1, float('0'), True)

        # pd_op.transpose: (1x96x160x2xf16) <- (1x2x96x160xf16)
        transpose_0 = paddle._C_ops.transpose(scale__0, [0, 2, 3, 1])

        # pd_op.add: (1x96x160x2xf16) <- (96x160x2xf16, 1x96x160x2xf16)
        add_1 = cast_3 + transpose_0

        # pd_op.slice: (1x96x160xf16) <- (1x96x160x2xf16, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(add_1, [3], constant_2, constant_3, [1], [3])

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(slice_0, constant_1, float('0'), True)

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(scale__1, constant_4, float('0'), True)

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(scale__2, constant_5, float('-1'), True)

        # pd_op.slice: (1x96x160xf16) <- (1x96x160x2xf16, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(add_1, [3], constant_3, constant_6, [1], [3])

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(slice_1, constant_1, float('0'), True)

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(scale__4, constant_7, float('0'), True)

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(scale__5, constant_5, float('-1'), True)

        # builtin.combine: ([1x96x160xf16, 1x96x160xf16]) <- (1x96x160xf16, 1x96x160xf16)
        combine_1 = [scale__3, scale__6]

        # pd_op.stack: (1x96x160x2xf16) <- ([1x96x160xf16, 1x96x160xf16])
        stack_0 = paddle._C_ops.stack(combine_1, 3)

        # pd_op.cast: (1x96x160x2xf32) <- (1x96x160x2xf16)
        cast_4 = paddle._C_ops.cast(stack_0, paddle.float32)

        # pd_op.cast: (1x3x96x160xf32) <- (1x3x96x160xf16)
        cast_5 = paddle._C_ops.cast(pool2d_2, paddle.float32)

        # pd_op.grid_sample: (1x3x96x160xf32) <- (1x3x96x160xf32, 1x96x160x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(cast_5, cast_4, 'bilinear', 'border', True)

        # pd_op.cast: (1x3x96x160xf16) <- (1x3x96x160xf32)
        cast_6 = paddle._C_ops.cast(grid_sample_1, paddle.float16)

        # builtin.combine: ([1x3x96x160xf16, 1x3x96x160xf16, 1x2x96x160xf16]) <- (1x3x96x160xf16, 1x3x96x160xf16, 1x2x96x160xf16)
        combine_2 = [pool2d_3, cast_6, scale__0]

        # pd_op.concat: (1x8x96x160xf16) <- ([1x3x96x160xf16, 1x3x96x160xf16, 1x2x96x160xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, constant_0)

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x8x96x160xf16, 16x8x3x3xf16)
        conv2d_12 = paddle._C_ops.conv2d(concat_1, parameter_25, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__12 = paddle._C_ops.add_(conv2d_12, parameter_26)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__11 = paddle._C_ops.leaky_relu_(add__12, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x16x96x160xf16, 16x16x3x3xf16)
        conv2d_13 = paddle._C_ops.conv2d(leaky_relu__11, parameter_27, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__13 = paddle._C_ops.add_(conv2d_13, parameter_28)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__12 = paddle._C_ops.leaky_relu_(add__13, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf16) <- (1x16x96x160xf16, 32x16x3x3xf16)
        conv2d_14 = paddle._C_ops.conv2d(leaky_relu__12, parameter_29, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x96x160xf16) <- (1x32x96x160xf16, 1x32x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_14, parameter_30)

        # pd_op.leaky_relu_: (1x32x96x160xf16) <- (1x32x96x160xf16)
        leaky_relu__13 = paddle._C_ops.leaky_relu_(add__14, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf16) <- (1x32x96x160xf16, 32x32x3x3xf16)
        conv2d_15 = paddle._C_ops.conv2d(leaky_relu__13, parameter_31, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x96x160xf16) <- (1x32x96x160xf16, 1x32x1x1xf16)
        add__15 = paddle._C_ops.add_(conv2d_15, parameter_32)

        # pd_op.leaky_relu_: (1x32x96x160xf16) <- (1x32x96x160xf16)
        leaky_relu__14 = paddle._C_ops.leaky_relu_(add__15, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf16) <- (1x32x96x160xf16, 32x32x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(leaky_relu__14, parameter_33, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x96x160xf16) <- (1x32x96x160xf16, 1x32x1x1xf16)
        add__16 = paddle._C_ops.add_(conv2d_16, parameter_34)

        # pd_op.leaky_relu_: (1x32x96x160xf16) <- (1x32x96x160xf16)
        leaky_relu__15 = paddle._C_ops.leaky_relu_(add__16, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf16) <- (1x32x96x160xf16, 32x32x3x3xf16)
        conv2d_17 = paddle._C_ops.conv2d(leaky_relu__15, parameter_35, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x96x160xf16) <- (1x32x96x160xf16, 1x32x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_17, parameter_36)

        # pd_op.leaky_relu_: (1x32x96x160xf16) <- (1x32x96x160xf16)
        leaky_relu__16 = paddle._C_ops.leaky_relu_(add__17, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x32x96x160xf16, 16x32x3x3xf16)
        conv2d_18 = paddle._C_ops.conv2d(leaky_relu__16, parameter_37, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__18 = paddle._C_ops.add_(conv2d_18, parameter_38)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__17 = paddle._C_ops.leaky_relu_(add__18, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x16x96x160xf16, 16x16x3x3xf16)
        conv2d_19 = paddle._C_ops.conv2d(leaky_relu__17, parameter_39, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__19 = paddle._C_ops.add_(conv2d_19, parameter_40)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__18 = paddle._C_ops.leaky_relu_(add__19, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x16x96x160xf16, 16x16x3x3xf16)
        conv2d_20 = paddle._C_ops.conv2d(leaky_relu__18, parameter_41, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__20 = paddle._C_ops.add_(conv2d_20, parameter_42)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__19 = paddle._C_ops.leaky_relu_(add__20, float('0.1'))

        # pd_op.conv2d: (1x8x96x160xf16) <- (1x16x96x160xf16, 8x16x3x3xf16)
        conv2d_21 = paddle._C_ops.conv2d(leaky_relu__19, parameter_43, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x96x160xf16) <- (1x8x96x160xf16, 1x8x1x1xf16)
        add__21 = paddle._C_ops.add_(conv2d_21, parameter_44)

        # pd_op.leaky_relu_: (1x8x96x160xf16) <- (1x8x96x160xf16)
        leaky_relu__20 = paddle._C_ops.leaky_relu_(add__21, float('0.1'))

        # pd_op.conv2d: (1x8x96x160xf16) <- (1x8x96x160xf16, 8x8x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(leaky_relu__20, parameter_45, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x96x160xf16) <- (1x8x96x160xf16, 1x8x1x1xf16)
        add__22 = paddle._C_ops.add_(conv2d_22, parameter_46)

        # pd_op.leaky_relu_: (1x8x96x160xf16) <- (1x8x96x160xf16)
        leaky_relu__21 = paddle._C_ops.leaky_relu_(add__22, float('0.1'))

        # pd_op.conv2d: (1x2x96x160xf16) <- (1x8x96x160xf16, 2x8x3x3xf16)
        conv2d_23 = paddle._C_ops.conv2d(leaky_relu__21, parameter_47, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x2x96x160xf16) <- (1x2x96x160xf16, 1x2x1x1xf16)
        add__23 = paddle._C_ops.add_(conv2d_23, parameter_48)

        # pd_op.add_: (1x2x96x160xf16) <- (1x2x96x160xf16, 1x2x96x160xf16)
        add__24 = paddle._C_ops.add_(scale__0, add__23)

        # pd_op.bilinear_interp: (1x2x192x320xf16) <- (1x2x96x160xf16, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(add__24, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

        # pd_op.scale_: (1x2x192x320xf16) <- (1x2x192x320xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(bilinear_interp_1, constant_1, float('0'), True)

        # pd_op.transpose: (1x192x320x2xf16) <- (1x2x192x320xf16)
        transpose_1 = paddle._C_ops.transpose(scale__7, [0, 2, 3, 1])

        # pd_op.add: (1x192x320x2xf16) <- (192x320x2xf16, 1x192x320x2xf16)
        add_2 = cast_7 + transpose_1

        # pd_op.slice: (1x192x320xf16) <- (1x192x320x2xf16, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(add_2, [3], constant_2, constant_3, [1], [3])

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__8 = paddle._C_ops.scale_(slice_2, constant_1, float('0'), True)

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__9 = paddle._C_ops.scale_(scale__8, constant_8, float('0'), True)

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__10 = paddle._C_ops.scale_(scale__9, constant_5, float('-1'), True)

        # pd_op.slice: (1x192x320xf16) <- (1x192x320x2xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(add_2, [3], constant_3, constant_6, [1], [3])

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__11 = paddle._C_ops.scale_(slice_3, constant_1, float('0'), True)

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__12 = paddle._C_ops.scale_(scale__11, constant_9, float('0'), True)

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__13 = paddle._C_ops.scale_(scale__12, constant_5, float('-1'), True)

        # builtin.combine: ([1x192x320xf16, 1x192x320xf16]) <- (1x192x320xf16, 1x192x320xf16)
        combine_3 = [scale__10, scale__13]

        # pd_op.stack: (1x192x320x2xf16) <- ([1x192x320xf16, 1x192x320xf16])
        stack_1 = paddle._C_ops.stack(combine_3, 3)

        # pd_op.cast: (1x192x320x2xf32) <- (1x192x320x2xf16)
        cast_8 = paddle._C_ops.cast(stack_1, paddle.float32)

        # pd_op.cast: (1x3x192x320xf32) <- (1x3x192x320xf16)
        cast_9 = paddle._C_ops.cast(divide__0, paddle.float32)

        # pd_op.grid_sample: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x192x320x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(cast_9, cast_8, 'bilinear', 'border', True)

        # pd_op.cast: (1x3x192x320xf16) <- (1x3x192x320xf32)
        cast_10 = paddle._C_ops.cast(grid_sample_2, paddle.float16)

        # builtin.combine: ([1x3x192x320xf16, 1x3x192x320xf16, 1x2x192x320xf16]) <- (1x3x192x320xf16, 1x3x192x320xf16, 1x2x192x320xf16)
        combine_4 = [divide__1, cast_10, scale__7]

        # pd_op.concat: (1x8x192x320xf16) <- ([1x3x192x320xf16, 1x3x192x320xf16, 1x2x192x320xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_4, constant_0)

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x8x192x320xf16, 16x8x3x3xf16)
        conv2d_24 = paddle._C_ops.conv2d(concat_2, parameter_49, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__25 = paddle._C_ops.add_(conv2d_24, parameter_50)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__22 = paddle._C_ops.leaky_relu_(add__25, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x16x192x320xf16, 16x16x3x3xf16)
        conv2d_25 = paddle._C_ops.conv2d(leaky_relu__22, parameter_51, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__26 = paddle._C_ops.add_(conv2d_25, parameter_52)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__23 = paddle._C_ops.leaky_relu_(add__26, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf16) <- (1x16x192x320xf16, 32x16x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(leaky_relu__23, parameter_53, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x192x320xf16) <- (1x32x192x320xf16, 1x32x1x1xf16)
        add__27 = paddle._C_ops.add_(conv2d_26, parameter_54)

        # pd_op.leaky_relu_: (1x32x192x320xf16) <- (1x32x192x320xf16)
        leaky_relu__24 = paddle._C_ops.leaky_relu_(add__27, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf16) <- (1x32x192x320xf16, 32x32x3x3xf16)
        conv2d_27 = paddle._C_ops.conv2d(leaky_relu__24, parameter_55, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x192x320xf16) <- (1x32x192x320xf16, 1x32x1x1xf16)
        add__28 = paddle._C_ops.add_(conv2d_27, parameter_56)

        # pd_op.leaky_relu_: (1x32x192x320xf16) <- (1x32x192x320xf16)
        leaky_relu__25 = paddle._C_ops.leaky_relu_(add__28, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf16) <- (1x32x192x320xf16, 32x32x3x3xf16)
        conv2d_28 = paddle._C_ops.conv2d(leaky_relu__25, parameter_57, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x192x320xf16) <- (1x32x192x320xf16, 1x32x1x1xf16)
        add__29 = paddle._C_ops.add_(conv2d_28, parameter_58)

        # pd_op.leaky_relu_: (1x32x192x320xf16) <- (1x32x192x320xf16)
        leaky_relu__26 = paddle._C_ops.leaky_relu_(add__29, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf16) <- (1x32x192x320xf16, 32x32x3x3xf16)
        conv2d_29 = paddle._C_ops.conv2d(leaky_relu__26, parameter_59, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x192x320xf16) <- (1x32x192x320xf16, 1x32x1x1xf16)
        add__30 = paddle._C_ops.add_(conv2d_29, parameter_60)

        # pd_op.leaky_relu_: (1x32x192x320xf16) <- (1x32x192x320xf16)
        leaky_relu__27 = paddle._C_ops.leaky_relu_(add__30, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x32x192x320xf16, 16x32x3x3xf16)
        conv2d_30 = paddle._C_ops.conv2d(leaky_relu__27, parameter_61, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__31 = paddle._C_ops.add_(conv2d_30, parameter_62)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__28 = paddle._C_ops.leaky_relu_(add__31, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x16x192x320xf16, 16x16x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(leaky_relu__28, parameter_63, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__32 = paddle._C_ops.add_(conv2d_31, parameter_64)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__29 = paddle._C_ops.leaky_relu_(add__32, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x16x192x320xf16, 16x16x3x3xf16)
        conv2d_32 = paddle._C_ops.conv2d(leaky_relu__29, parameter_65, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__33 = paddle._C_ops.add_(conv2d_32, parameter_66)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__30 = paddle._C_ops.leaky_relu_(add__33, float('0.1'))

        # pd_op.conv2d: (1x8x192x320xf16) <- (1x16x192x320xf16, 8x16x3x3xf16)
        conv2d_33 = paddle._C_ops.conv2d(leaky_relu__30, parameter_67, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x192x320xf16) <- (1x8x192x320xf16, 1x8x1x1xf16)
        add__34 = paddle._C_ops.add_(conv2d_33, parameter_68)

        # pd_op.leaky_relu_: (1x8x192x320xf16) <- (1x8x192x320xf16)
        leaky_relu__31 = paddle._C_ops.leaky_relu_(add__34, float('0.1'))

        # pd_op.conv2d: (1x8x192x320xf16) <- (1x8x192x320xf16, 8x8x3x3xf16)
        conv2d_34 = paddle._C_ops.conv2d(leaky_relu__31, parameter_69, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x192x320xf16) <- (1x8x192x320xf16, 1x8x1x1xf16)
        add__35 = paddle._C_ops.add_(conv2d_34, parameter_70)

        # pd_op.leaky_relu_: (1x8x192x320xf16) <- (1x8x192x320xf16)
        leaky_relu__32 = paddle._C_ops.leaky_relu_(add__35, float('0.1'))

        # pd_op.conv2d: (1x2x192x320xf16) <- (1x8x192x320xf16, 2x8x3x3xf16)
        conv2d_35 = paddle._C_ops.conv2d(leaky_relu__32, parameter_71, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x2x192x320xf16) <- (1x2x192x320xf16, 1x2x1x1xf16)
        add__36 = paddle._C_ops.add_(conv2d_35, parameter_72)

        # pd_op.add_: (1x2x192x320xf16) <- (1x2x192x320xf16, 1x2x192x320xf16)
        add__37 = paddle._C_ops.add_(scale__7, add__36)

        # pd_op.bilinear_interp: (1x2x180x320xf16) <- (1x2x192x320xf16, None, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(add__37, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

        # pd_op.slice: (1x180x320xf16) <- (1x2x180x320xf16, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(bilinear_interp_2, [1], constant_2, constant_3, [1], [1])

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__14 = paddle._C_ops.scale_(slice_4, constant_5, float('0'), True)

        # pd_op.set_value_with_tensor_: (1x2x180x320xf16) <- (1x2x180x320xf16, 1x180x320xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(bilinear_interp_2, scale__14, constant_2, constant_3, constant_3, [1], [1], [])

        # pd_op.slice: (1x180x320xf16) <- (1x2x180x320xf16, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(set_value_with_tensor__0, [1], constant_3, constant_6, [1], [1])

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__15 = paddle._C_ops.scale_(slice_5, constant_10, float('0'), True)

        # pd_op.set_value_with_tensor_: (1x2x180x320xf16) <- (1x2x180x320xf16, 1x180x320xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__0, scale__15, constant_3, constant_6, constant_3, [1], [1], [])

        # pd_op.reshape_: (1x1x2x180x320xf16, 0x1x2x180x320xf16) <- (1x2x180x320xf16, 5xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(set_value_with_tensor__1, constant_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (1x1x2x180x320xf32) <- (1x1x2x180x320xf16)
        cast_11 = paddle._C_ops.cast(reshape__0, paddle.float32)

        # pd_op.assign_: (1x1x2x180x320xf32) <- (1x1x2x180x320xf32)
        assign__0 = paddle._C_ops.assign_(cast_11)
        return assign__0
    def pd_op_if_4969_1_0(self, parameter_0):
        return parameter_0
    def builtin_module_4432_0_0(self, parameter_96, parameter_95, parameter_262, parameter_260, parameter_258, parameter_256, parameter_254, parameter_252, parameter_250, parameter_248, parameter_246, parameter_244, parameter_242, parameter_240, parameter_238, parameter_236, parameter_234, parameter_232, parameter_230, parameter_228, parameter_226, parameter_224, parameter_222, parameter_220, parameter_218, constant_22, parameter_216, parameter_214, parameter_212, parameter_210, parameter_208, parameter_206, parameter_204, parameter_202, parameter_200, parameter_198, parameter_196, parameter_194, parameter_192, parameter_190, parameter_188, parameter_186, parameter_184, parameter_182, parameter_180, parameter_178, parameter_176, parameter_174, parameter_172, parameter_170, parameter_168, parameter_166, parameter_164, parameter_162, parameter_160, parameter_158, parameter_156, parameter_154, parameter_152, parameter_150, parameter_148, parameter_146, constant_21, parameter_144, parameter_142, parameter_140, parameter_138, parameter_136, parameter_134, parameter_132, parameter_130, parameter_128, parameter_126, parameter_124, parameter_122, parameter_120, parameter_118, parameter_116, parameter_114, parameter_112, parameter_110, parameter_107, constant_20, constant_19, parameter_105, parameter_103, parameter_101, parameter_99, constant_18, parameter_97, parameter_108, constant_17, constant_16, parameter_94, parameter_92, parameter_90, parameter_88, parameter_86, parameter_84, parameter_82, parameter_80, parameter_78, parameter_76, parameter_74, parameter_72, constant_15, constant_14, parameter_70, parameter_69, parameter_68, parameter_66, parameter_64, parameter_62, parameter_60, parameter_58, parameter_56, parameter_54, parameter_52, parameter_50, parameter_48, parameter_46, constant_13, constant_12, parameter_44, parameter_43, parameter_42, parameter_40, parameter_38, parameter_36, parameter_34, parameter_32, parameter_30, parameter_28, parameter_26, parameter_24, parameter_22, parameter_20, constant_11, constant_10, constant_9, parameter_16, parameter_15, parameter_17, parameter_18, constant_8, constant_7, constant_6, constant_5, constant_4, parameter_12, parameter_10, constant_3, parameter_8, parameter_6, parameter_4, constant_2, parameter_2, parameter_1, parameter_0, constant_1, constant_0, parameter_3, parameter_5, parameter_7, parameter_9, parameter_11, parameter_13, parameter_14, parameter_19, parameter_21, parameter_23, parameter_25, parameter_27, parameter_29, parameter_31, parameter_33, parameter_35, parameter_37, parameter_39, parameter_41, parameter_45, parameter_47, parameter_49, parameter_51, parameter_53, parameter_55, parameter_57, parameter_59, parameter_61, parameter_63, parameter_65, parameter_67, parameter_71, parameter_73, parameter_75, parameter_77, parameter_79, parameter_81, parameter_83, parameter_85, parameter_87, parameter_89, parameter_91, parameter_93, parameter_98, parameter_100, parameter_102, parameter_104, parameter_106, parameter_109, parameter_111, parameter_113, parameter_115, parameter_117, parameter_119, parameter_121, parameter_123, parameter_125, parameter_127, parameter_129, parameter_131, parameter_133, parameter_135, parameter_137, parameter_139, parameter_141, parameter_143, parameter_145, parameter_147, parameter_149, parameter_151, parameter_153, parameter_155, parameter_157, parameter_159, parameter_161, parameter_163, parameter_165, parameter_167, parameter_169, parameter_171, parameter_173, parameter_175, parameter_177, parameter_179, parameter_181, parameter_183, parameter_185, parameter_187, parameter_189, parameter_191, parameter_193, parameter_195, parameter_197, parameter_199, parameter_201, parameter_203, parameter_205, parameter_207, parameter_209, parameter_211, parameter_213, parameter_215, parameter_217, parameter_219, parameter_221, parameter_223, parameter_225, parameter_227, parameter_229, parameter_231, parameter_233, parameter_235, parameter_237, parameter_239, parameter_241, parameter_243, parameter_245, parameter_247, parameter_249, parameter_251, parameter_253, parameter_255, parameter_257, parameter_259, parameter_261, feed_0):

        # pd_op.cast: (1x2x3x180x320xf16) <- (1x2x3x180x320xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.split_with_num: ([1x1x3x180x320xf16, 1x1x3x180x320xf16]) <- (1x2x3x180x320xf16, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(cast_0, 2, constant_0)

        # builtin.slice: (1x1x3x180x320xf16) <- ([1x1x3x180x320xf16, 1x1x3x180x320xf16])
        slice_0 = split_with_num_0[1]

        # pd_op.flip: (1x1x3x180x320xf16) <- (1x1x3x180x320xf16)
        flip_0 = paddle._C_ops.flip(slice_0, [1])

        # builtin.slice: (1x1x3x180x320xf16) <- ([1x1x3x180x320xf16, 1x1x3x180x320xf16])
        slice_1 = split_with_num_0[0]

        # pd_op.subtract_: (1x1x3x180x320xf16) <- (1x1x3x180x320xf16, 1x1x3x180x320xf16)
        subtract__0 = paddle._C_ops.subtract_(slice_1, flip_0)

        # pd_op.cast: (1x1x3x180x320xf32) <- (1x1x3x180x320xf16)
        cast_1 = paddle._C_ops.cast(subtract__0, paddle.float32)

        # pd_op.frobenius_norm: (xf32) <- (1x1x3x180x320xf32, 1xi64)
        frobenius_norm_0 = paddle._C_ops.frobenius_norm(cast_1, constant_1, False, True)

        # pd_op.cast: (xf16) <- (xf32)
        cast_2 = paddle._C_ops.cast(frobenius_norm_0, paddle.float16)

        # pd_op.equal: (xb) <- (xf16, xf16)
        equal_0 = paddle._C_ops.equal(cast_2, parameter_0)

        # pd_op.cast: (xi32) <- (xb)
        cast_3 = paddle._C_ops.cast(equal_0, paddle.int32)

        # pd_op.select_input: (xb) <- (xi32, xb, xb)
        select_input_0 = [parameter_1, parameter_2][int(cast_3)]

        # pd_op.reshape: (2x3x180x320xf16, 0x1x2x3x180x320xf16) <- (1x2x3x180x320xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_0, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (2x32x180x320xf16) <- (2x3x180x320xf16, 32x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(reshape_0, parameter_3, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (2x32x180x320xf16) <- (2x32x180x320xf16, 1x32x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, parameter_4)

        # pd_op.leaky_relu_: (2x32x180x320xf16) <- (2x32x180x320xf16)
        leaky_relu__0 = paddle._C_ops.leaky_relu_(add__0, float('0.1'))

        # pd_op.conv2d: (2x32x180x320xf16) <- (2x32x180x320xf16, 32x32x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(leaky_relu__0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (2x32x180x320xf16) <- (2x32x180x320xf16, 1x32x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_1, parameter_6)

        # pd_op.relu_: (2x32x180x320xf16) <- (2x32x180x320xf16)
        relu__0 = paddle._C_ops.relu_(add__1)

        # pd_op.conv2d: (2x32x180x320xf16) <- (2x32x180x320xf16, 32x32x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__0, parameter_7, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (2x32x180x320xf16) <- (2x32x180x320xf16, 1x32x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_2, parameter_8)

        # pd_op.scale_: (2x32x180x320xf16) <- (2x32x180x320xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(add__2, constant_3, float('0'), True)

        # pd_op.add_: (2x32x180x320xf16) <- (2x32x180x320xf16, 2x32x180x320xf16)
        add__3 = paddle._C_ops.add_(leaky_relu__0, scale__0)

        # pd_op.conv2d: (2x32x180x320xf16) <- (2x32x180x320xf16, 32x32x3x3xf16)
        conv2d_3 = paddle._C_ops.conv2d(add__3, parameter_9, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (2x32x180x320xf16) <- (2x32x180x320xf16, 1x32x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_3, parameter_10)

        # pd_op.relu_: (2x32x180x320xf16) <- (2x32x180x320xf16)
        relu__1 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (2x32x180x320xf16) <- (2x32x180x320xf16, 32x32x3x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__1, parameter_11, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (2x32x180x320xf16) <- (2x32x180x320xf16, 1x32x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_4, parameter_12)

        # pd_op.scale_: (2x32x180x320xf16) <- (2x32x180x320xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(add__5, constant_3, float('0'), True)

        # pd_op.add_: (2x32x180x320xf16) <- (2x32x180x320xf16, 2x32x180x320xf16)
        add__6 = paddle._C_ops.add_(add__3, scale__1)

        # pd_op.reshape_: (1x2x32x180x320xf16, 0x2x32x180x320xf16) <- (2x32x180x320xf16, 5xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__6, constant_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.slice: (1x32x180x320xf16) <- (1x2x32x180x320xf16, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(reshape__0, [1], constant_1, constant_5, [1], [1])

        # pd_op.slice: (1x32x180x320xf16) <- (1x2x32x180x320xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(reshape__0, [1], constant_5, constant_6, [1], [1])

        # pd_op.slice: (1x1x3x180x320xf16) <- (1x2x3x180x320xf16, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(cast_0, [1], constant_1, constant_7, [1], [])

        # pd_op.reshape_: (1x3x180x320xf16, 0x1x1x3x180x320xf16) <- (1x1x3x180x320xf16, 4xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_4, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.slice: (1x1x3x180x320xf16) <- (1x2x3x180x320xf16, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(cast_0, [1], constant_5, constant_6, [1], [])

        # pd_op.reshape_: (1x3x180x320xf16, 0x1x1x3x180x320xf16) <- (1x1x3x180x320xf16, 4xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_5, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.bilinear_interp: (1x3x192x320xf16) <- (1x3x180x320xf16, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(reshape__2, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

        # pd_op.bilinear_interp: (1x3x192x320xf16) <- (1x3x180x320xf16, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(reshape__4, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

        # pd_op.subtract_: (1x3x192x320xf16) <- (1x3x192x320xf16, 1x3x1x1xf16)
        subtract__1 = paddle._C_ops.subtract_(bilinear_interp_0, parameter_13)

        # pd_op.divide_: (1x3x192x320xf16) <- (1x3x192x320xf16, 1x3x1x1xf16)
        divide__0 = paddle._C_ops.divide_(subtract__1, parameter_14)

        # pd_op.subtract_: (1x3x192x320xf16) <- (1x3x192x320xf16, 1x3x1x1xf16)
        subtract__2 = paddle._C_ops.subtract_(bilinear_interp_1, parameter_13)

        # pd_op.divide_: (1x3x192x320xf16) <- (1x3x192x320xf16, 1x3x1x1xf16)
        divide__1 = paddle._C_ops.divide_(subtract__2, parameter_14)

        # pd_op.pool2d: (1x3x96x160xf16) <- (1x3x192x320xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(divide__0, constant_8, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.pool2d: (1x3x96x160xf16) <- (1x3x192x320xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(divide__1, constant_8, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.pool2d: (1x3x48x80xf16) <- (1x3x96x160xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(pool2d_0, constant_8, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.pool2d: (1x3x48x80xf16) <- (1x3x96x160xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(pool2d_1, constant_8, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # builtin.combine: ([48xi64, 80xi64]) <- (48xi64, 80xi64)
        combine_0 = [parameter_15, parameter_16]

        # pd_op.meshgrid: ([48x80xi64, 48x80xi64]) <- ([48xi64, 80xi64])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.slice: (48x80xi64) <- ([48x80xi64, 48x80xi64])
        slice_6 = meshgrid_0[1]

        # builtin.slice: (48x80xi64) <- ([48x80xi64, 48x80xi64])
        slice_7 = meshgrid_0[0]

        # builtin.combine: ([48x80xi64, 48x80xi64]) <- (48x80xi64, 48x80xi64)
        combine_1 = [slice_6, slice_7]

        # pd_op.stack: (48x80x2xi64) <- ([48x80xi64, 48x80xi64])
        stack_0 = paddle._C_ops.stack(combine_1, 2)

        # pd_op.cast: (48x80x2xf16) <- (48x80x2xi64)
        cast_4 = paddle._C_ops.cast(stack_0, paddle.float16)

        # pd_op.add_: (1x48x80x2xf16) <- (48x80x2xf16, 1x48x80x2xf16)
        add__7 = paddle._C_ops.add_(cast_4, parameter_17)

        # pd_op.slice: (1x48x80xf16) <- (1x48x80x2xf16, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(add__7, [3], constant_1, constant_5, [1], [3])

        # pd_op.scale_: (1x48x80xf16) <- (1x48x80xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(slice_8, constant_9, float('0'), True)

        # pd_op.scale_: (1x48x80xf16) <- (1x48x80xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(scale__2, constant_10, float('0'), True)

        # pd_op.scale_: (1x48x80xf16) <- (1x48x80xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(scale__3, constant_3, float('-1'), True)

        # pd_op.slice: (1x48x80xf16) <- (1x48x80x2xf16, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(add__7, [3], constant_5, constant_6, [1], [3])

        # pd_op.scale_: (1x48x80xf16) <- (1x48x80xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(slice_9, constant_9, float('0'), True)

        # pd_op.scale_: (1x48x80xf16) <- (1x48x80xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(scale__5, constant_11, float('0'), True)

        # pd_op.scale_: (1x48x80xf16) <- (1x48x80xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(scale__6, constant_3, float('-1'), True)

        # builtin.combine: ([1x48x80xf16, 1x48x80xf16]) <- (1x48x80xf16, 1x48x80xf16)
        combine_2 = [scale__4, scale__7]

        # pd_op.stack: (1x48x80x2xf16) <- ([1x48x80xf16, 1x48x80xf16])
        stack_1 = paddle._C_ops.stack(combine_2, 3)

        # pd_op.cast: (1x48x80x2xf32) <- (1x48x80x2xf16)
        cast_5 = paddle._C_ops.cast(stack_1, paddle.float32)

        # pd_op.cast: (1x3x48x80xf32) <- (1x3x48x80xf16)
        cast_6 = paddle._C_ops.cast(pool2d_3, paddle.float32)

        # pd_op.grid_sample: (1x3x48x80xf32) <- (1x3x48x80xf32, 1x48x80x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(cast_6, cast_5, 'bilinear', 'border', True)

        # pd_op.cast: (1x3x48x80xf16) <- (1x3x48x80xf32)
        cast_7 = paddle._C_ops.cast(grid_sample_0, paddle.float16)

        # builtin.combine: ([1x3x48x80xf16, 1x3x48x80xf16, 1x2x48x80xf16]) <- (1x3x48x80xf16, 1x3x48x80xf16, 1x2x48x80xf16)
        combine_3 = [pool2d_2, cast_7, parameter_18]

        # pd_op.concat: (1x8x48x80xf16) <- ([1x3x48x80xf16, 1x3x48x80xf16, 1x2x48x80xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_3, constant_0)

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x8x48x80xf16, 16x8x3x3xf16)
        conv2d_5 = paddle._C_ops.conv2d(concat_0, parameter_19, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_5, parameter_20)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__1 = paddle._C_ops.leaky_relu_(add__8, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x16x48x80xf16, 16x16x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(leaky_relu__1, parameter_21, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__9 = paddle._C_ops.add_(conv2d_6, parameter_22)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__2 = paddle._C_ops.leaky_relu_(add__9, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf16) <- (1x16x48x80xf16, 32x16x3x3xf16)
        conv2d_7 = paddle._C_ops.conv2d(leaky_relu__2, parameter_23, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x48x80xf16) <- (1x32x48x80xf16, 1x32x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_7, parameter_24)

        # pd_op.leaky_relu_: (1x32x48x80xf16) <- (1x32x48x80xf16)
        leaky_relu__3 = paddle._C_ops.leaky_relu_(add__10, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf16) <- (1x32x48x80xf16, 32x32x3x3xf16)
        conv2d_8 = paddle._C_ops.conv2d(leaky_relu__3, parameter_25, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x48x80xf16) <- (1x32x48x80xf16, 1x32x1x1xf16)
        add__11 = paddle._C_ops.add_(conv2d_8, parameter_26)

        # pd_op.leaky_relu_: (1x32x48x80xf16) <- (1x32x48x80xf16)
        leaky_relu__4 = paddle._C_ops.leaky_relu_(add__11, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf16) <- (1x32x48x80xf16, 32x32x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(leaky_relu__4, parameter_27, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x48x80xf16) <- (1x32x48x80xf16, 1x32x1x1xf16)
        add__12 = paddle._C_ops.add_(conv2d_9, parameter_28)

        # pd_op.leaky_relu_: (1x32x48x80xf16) <- (1x32x48x80xf16)
        leaky_relu__5 = paddle._C_ops.leaky_relu_(add__12, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf16) <- (1x32x48x80xf16, 32x32x3x3xf16)
        conv2d_10 = paddle._C_ops.conv2d(leaky_relu__5, parameter_29, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x48x80xf16) <- (1x32x48x80xf16, 1x32x1x1xf16)
        add__13 = paddle._C_ops.add_(conv2d_10, parameter_30)

        # pd_op.leaky_relu_: (1x32x48x80xf16) <- (1x32x48x80xf16)
        leaky_relu__6 = paddle._C_ops.leaky_relu_(add__13, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x32x48x80xf16, 16x32x3x3xf16)
        conv2d_11 = paddle._C_ops.conv2d(leaky_relu__6, parameter_31, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_11, parameter_32)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__7 = paddle._C_ops.leaky_relu_(add__14, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x16x48x80xf16, 16x16x3x3xf16)
        conv2d_12 = paddle._C_ops.conv2d(leaky_relu__7, parameter_33, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__15 = paddle._C_ops.add_(conv2d_12, parameter_34)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__8 = paddle._C_ops.leaky_relu_(add__15, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf16) <- (1x16x48x80xf16, 16x16x3x3xf16)
        conv2d_13 = paddle._C_ops.conv2d(leaky_relu__8, parameter_35, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x48x80xf16) <- (1x16x48x80xf16, 1x16x1x1xf16)
        add__16 = paddle._C_ops.add_(conv2d_13, parameter_36)

        # pd_op.leaky_relu_: (1x16x48x80xf16) <- (1x16x48x80xf16)
        leaky_relu__9 = paddle._C_ops.leaky_relu_(add__16, float('0.1'))

        # pd_op.conv2d: (1x8x48x80xf16) <- (1x16x48x80xf16, 8x16x3x3xf16)
        conv2d_14 = paddle._C_ops.conv2d(leaky_relu__9, parameter_37, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x48x80xf16) <- (1x8x48x80xf16, 1x8x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_14, parameter_38)

        # pd_op.leaky_relu_: (1x8x48x80xf16) <- (1x8x48x80xf16)
        leaky_relu__10 = paddle._C_ops.leaky_relu_(add__17, float('0.1'))

        # pd_op.conv2d: (1x8x48x80xf16) <- (1x8x48x80xf16, 8x8x3x3xf16)
        conv2d_15 = paddle._C_ops.conv2d(leaky_relu__10, parameter_39, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x48x80xf16) <- (1x8x48x80xf16, 1x8x1x1xf16)
        add__18 = paddle._C_ops.add_(conv2d_15, parameter_40)

        # pd_op.leaky_relu_: (1x8x48x80xf16) <- (1x8x48x80xf16)
        leaky_relu__11 = paddle._C_ops.leaky_relu_(add__18, float('0.1'))

        # pd_op.conv2d: (1x2x48x80xf16) <- (1x8x48x80xf16, 2x8x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(leaky_relu__11, parameter_41, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x2x48x80xf16) <- (1x2x48x80xf16, 1x2x1x1xf16)
        add__19 = paddle._C_ops.add_(conv2d_16, parameter_42)

        # pd_op.add: (1x2x48x80xf16) <- (1x2x48x80xf16, 1x2x48x80xf16)
        add_0 = parameter_18 + add__19

        # pd_op.bilinear_interp: (1x2x96x160xf16) <- (1x2x48x80xf16, None, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(add_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

        # pd_op.scale_: (1x2x96x160xf16) <- (1x2x96x160xf16, 1xf32)
        scale__8 = paddle._C_ops.scale_(bilinear_interp_2, constant_9, float('0'), True)

        # pd_op.transpose: (1x96x160x2xf16) <- (1x2x96x160xf16)
        transpose_0 = paddle._C_ops.transpose(scale__8, [0, 2, 3, 1])

        # builtin.combine: ([96xi64, 160xi64]) <- (96xi64, 160xi64)
        combine_4 = [parameter_43, parameter_44]

        # pd_op.meshgrid: ([96x160xi64, 96x160xi64]) <- ([96xi64, 160xi64])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_4)

        # builtin.slice: (96x160xi64) <- ([96x160xi64, 96x160xi64])
        slice_10 = meshgrid_1[1]

        # builtin.slice: (96x160xi64) <- ([96x160xi64, 96x160xi64])
        slice_11 = meshgrid_1[0]

        # builtin.combine: ([96x160xi64, 96x160xi64]) <- (96x160xi64, 96x160xi64)
        combine_5 = [slice_10, slice_11]

        # pd_op.stack: (96x160x2xi64) <- ([96x160xi64, 96x160xi64])
        stack_2 = paddle._C_ops.stack(combine_5, 2)

        # pd_op.cast: (96x160x2xf16) <- (96x160x2xi64)
        cast_8 = paddle._C_ops.cast(stack_2, paddle.float16)

        # pd_op.add: (1x96x160x2xf16) <- (96x160x2xf16, 1x96x160x2xf16)
        add_1 = cast_8 + transpose_0

        # pd_op.slice: (1x96x160xf16) <- (1x96x160x2xf16, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(add_1, [3], constant_1, constant_5, [1], [3])

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__9 = paddle._C_ops.scale_(slice_12, constant_9, float('0'), True)

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__10 = paddle._C_ops.scale_(scale__9, constant_12, float('0'), True)

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__11 = paddle._C_ops.scale_(scale__10, constant_3, float('-1'), True)

        # pd_op.slice: (1x96x160xf16) <- (1x96x160x2xf16, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(add_1, [3], constant_5, constant_6, [1], [3])

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__12 = paddle._C_ops.scale_(slice_13, constant_9, float('0'), True)

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__13 = paddle._C_ops.scale_(scale__12, constant_13, float('0'), True)

        # pd_op.scale_: (1x96x160xf16) <- (1x96x160xf16, 1xf32)
        scale__14 = paddle._C_ops.scale_(scale__13, constant_3, float('-1'), True)

        # builtin.combine: ([1x96x160xf16, 1x96x160xf16]) <- (1x96x160xf16, 1x96x160xf16)
        combine_6 = [scale__11, scale__14]

        # pd_op.stack: (1x96x160x2xf16) <- ([1x96x160xf16, 1x96x160xf16])
        stack_3 = paddle._C_ops.stack(combine_6, 3)

        # pd_op.cast: (1x96x160x2xf32) <- (1x96x160x2xf16)
        cast_9 = paddle._C_ops.cast(stack_3, paddle.float32)

        # pd_op.cast: (1x3x96x160xf32) <- (1x3x96x160xf16)
        cast_10 = paddle._C_ops.cast(pool2d_1, paddle.float32)

        # pd_op.grid_sample: (1x3x96x160xf32) <- (1x3x96x160xf32, 1x96x160x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(cast_10, cast_9, 'bilinear', 'border', True)

        # pd_op.cast: (1x3x96x160xf16) <- (1x3x96x160xf32)
        cast_11 = paddle._C_ops.cast(grid_sample_1, paddle.float16)

        # builtin.combine: ([1x3x96x160xf16, 1x3x96x160xf16, 1x2x96x160xf16]) <- (1x3x96x160xf16, 1x3x96x160xf16, 1x2x96x160xf16)
        combine_7 = [pool2d_0, cast_11, scale__8]

        # pd_op.concat: (1x8x96x160xf16) <- ([1x3x96x160xf16, 1x3x96x160xf16, 1x2x96x160xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_7, constant_0)

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x8x96x160xf16, 16x8x3x3xf16)
        conv2d_17 = paddle._C_ops.conv2d(concat_1, parameter_45, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__20 = paddle._C_ops.add_(conv2d_17, parameter_46)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__12 = paddle._C_ops.leaky_relu_(add__20, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x16x96x160xf16, 16x16x3x3xf16)
        conv2d_18 = paddle._C_ops.conv2d(leaky_relu__12, parameter_47, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__21 = paddle._C_ops.add_(conv2d_18, parameter_48)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__13 = paddle._C_ops.leaky_relu_(add__21, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf16) <- (1x16x96x160xf16, 32x16x3x3xf16)
        conv2d_19 = paddle._C_ops.conv2d(leaky_relu__13, parameter_49, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x96x160xf16) <- (1x32x96x160xf16, 1x32x1x1xf16)
        add__22 = paddle._C_ops.add_(conv2d_19, parameter_50)

        # pd_op.leaky_relu_: (1x32x96x160xf16) <- (1x32x96x160xf16)
        leaky_relu__14 = paddle._C_ops.leaky_relu_(add__22, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf16) <- (1x32x96x160xf16, 32x32x3x3xf16)
        conv2d_20 = paddle._C_ops.conv2d(leaky_relu__14, parameter_51, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x96x160xf16) <- (1x32x96x160xf16, 1x32x1x1xf16)
        add__23 = paddle._C_ops.add_(conv2d_20, parameter_52)

        # pd_op.leaky_relu_: (1x32x96x160xf16) <- (1x32x96x160xf16)
        leaky_relu__15 = paddle._C_ops.leaky_relu_(add__23, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf16) <- (1x32x96x160xf16, 32x32x3x3xf16)
        conv2d_21 = paddle._C_ops.conv2d(leaky_relu__15, parameter_53, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x96x160xf16) <- (1x32x96x160xf16, 1x32x1x1xf16)
        add__24 = paddle._C_ops.add_(conv2d_21, parameter_54)

        # pd_op.leaky_relu_: (1x32x96x160xf16) <- (1x32x96x160xf16)
        leaky_relu__16 = paddle._C_ops.leaky_relu_(add__24, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf16) <- (1x32x96x160xf16, 32x32x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(leaky_relu__16, parameter_55, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x96x160xf16) <- (1x32x96x160xf16, 1x32x1x1xf16)
        add__25 = paddle._C_ops.add_(conv2d_22, parameter_56)

        # pd_op.leaky_relu_: (1x32x96x160xf16) <- (1x32x96x160xf16)
        leaky_relu__17 = paddle._C_ops.leaky_relu_(add__25, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x32x96x160xf16, 16x32x3x3xf16)
        conv2d_23 = paddle._C_ops.conv2d(leaky_relu__17, parameter_57, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__26 = paddle._C_ops.add_(conv2d_23, parameter_58)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__18 = paddle._C_ops.leaky_relu_(add__26, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x16x96x160xf16, 16x16x3x3xf16)
        conv2d_24 = paddle._C_ops.conv2d(leaky_relu__18, parameter_59, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__27 = paddle._C_ops.add_(conv2d_24, parameter_60)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__19 = paddle._C_ops.leaky_relu_(add__27, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf16) <- (1x16x96x160xf16, 16x16x3x3xf16)
        conv2d_25 = paddle._C_ops.conv2d(leaky_relu__19, parameter_61, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x96x160xf16) <- (1x16x96x160xf16, 1x16x1x1xf16)
        add__28 = paddle._C_ops.add_(conv2d_25, parameter_62)

        # pd_op.leaky_relu_: (1x16x96x160xf16) <- (1x16x96x160xf16)
        leaky_relu__20 = paddle._C_ops.leaky_relu_(add__28, float('0.1'))

        # pd_op.conv2d: (1x8x96x160xf16) <- (1x16x96x160xf16, 8x16x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(leaky_relu__20, parameter_63, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x96x160xf16) <- (1x8x96x160xf16, 1x8x1x1xf16)
        add__29 = paddle._C_ops.add_(conv2d_26, parameter_64)

        # pd_op.leaky_relu_: (1x8x96x160xf16) <- (1x8x96x160xf16)
        leaky_relu__21 = paddle._C_ops.leaky_relu_(add__29, float('0.1'))

        # pd_op.conv2d: (1x8x96x160xf16) <- (1x8x96x160xf16, 8x8x3x3xf16)
        conv2d_27 = paddle._C_ops.conv2d(leaky_relu__21, parameter_65, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x96x160xf16) <- (1x8x96x160xf16, 1x8x1x1xf16)
        add__30 = paddle._C_ops.add_(conv2d_27, parameter_66)

        # pd_op.leaky_relu_: (1x8x96x160xf16) <- (1x8x96x160xf16)
        leaky_relu__22 = paddle._C_ops.leaky_relu_(add__30, float('0.1'))

        # pd_op.conv2d: (1x2x96x160xf16) <- (1x8x96x160xf16, 2x8x3x3xf16)
        conv2d_28 = paddle._C_ops.conv2d(leaky_relu__22, parameter_67, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x2x96x160xf16) <- (1x2x96x160xf16, 1x2x1x1xf16)
        add__31 = paddle._C_ops.add_(conv2d_28, parameter_68)

        # pd_op.add_: (1x2x96x160xf16) <- (1x2x96x160xf16, 1x2x96x160xf16)
        add__32 = paddle._C_ops.add_(scale__8, add__31)

        # pd_op.bilinear_interp: (1x2x192x320xf16) <- (1x2x96x160xf16, None, None, None)
        bilinear_interp_3 = paddle._C_ops.bilinear_interp(add__32, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

        # pd_op.scale_: (1x2x192x320xf16) <- (1x2x192x320xf16, 1xf32)
        scale__15 = paddle._C_ops.scale_(bilinear_interp_3, constant_9, float('0'), True)

        # pd_op.transpose: (1x192x320x2xf16) <- (1x2x192x320xf16)
        transpose_1 = paddle._C_ops.transpose(scale__15, [0, 2, 3, 1])

        # builtin.combine: ([192xi64, 320xi64]) <- (192xi64, 320xi64)
        combine_8 = [parameter_69, parameter_70]

        # pd_op.meshgrid: ([192x320xi64, 192x320xi64]) <- ([192xi64, 320xi64])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_8)

        # builtin.slice: (192x320xi64) <- ([192x320xi64, 192x320xi64])
        slice_14 = meshgrid_2[1]

        # builtin.slice: (192x320xi64) <- ([192x320xi64, 192x320xi64])
        slice_15 = meshgrid_2[0]

        # builtin.combine: ([192x320xi64, 192x320xi64]) <- (192x320xi64, 192x320xi64)
        combine_9 = [slice_14, slice_15]

        # pd_op.stack: (192x320x2xi64) <- ([192x320xi64, 192x320xi64])
        stack_4 = paddle._C_ops.stack(combine_9, 2)

        # pd_op.cast: (192x320x2xf16) <- (192x320x2xi64)
        cast_12 = paddle._C_ops.cast(stack_4, paddle.float16)

        # pd_op.add: (1x192x320x2xf16) <- (192x320x2xf16, 1x192x320x2xf16)
        add_2 = cast_12 + transpose_1

        # pd_op.slice: (1x192x320xf16) <- (1x192x320x2xf16, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(add_2, [3], constant_1, constant_5, [1], [3])

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__16 = paddle._C_ops.scale_(slice_16, constant_9, float('0'), True)

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__17 = paddle._C_ops.scale_(scale__16, constant_14, float('0'), True)

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__18 = paddle._C_ops.scale_(scale__17, constant_3, float('-1'), True)

        # pd_op.slice: (1x192x320xf16) <- (1x192x320x2xf16, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(add_2, [3], constant_5, constant_6, [1], [3])

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__19 = paddle._C_ops.scale_(slice_17, constant_9, float('0'), True)

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__20 = paddle._C_ops.scale_(scale__19, constant_15, float('0'), True)

        # pd_op.scale_: (1x192x320xf16) <- (1x192x320xf16, 1xf32)
        scale__21 = paddle._C_ops.scale_(scale__20, constant_3, float('-1'), True)

        # builtin.combine: ([1x192x320xf16, 1x192x320xf16]) <- (1x192x320xf16, 1x192x320xf16)
        combine_10 = [scale__18, scale__21]

        # pd_op.stack: (1x192x320x2xf16) <- ([1x192x320xf16, 1x192x320xf16])
        stack_5 = paddle._C_ops.stack(combine_10, 3)

        # pd_op.cast: (1x192x320x2xf32) <- (1x192x320x2xf16)
        cast_13 = paddle._C_ops.cast(stack_5, paddle.float32)

        # pd_op.cast: (1x3x192x320xf32) <- (1x3x192x320xf16)
        cast_14 = paddle._C_ops.cast(divide__1, paddle.float32)

        # pd_op.grid_sample: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x192x320x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(cast_14, cast_13, 'bilinear', 'border', True)

        # pd_op.cast: (1x3x192x320xf16) <- (1x3x192x320xf32)
        cast_15 = paddle._C_ops.cast(grid_sample_2, paddle.float16)

        # builtin.combine: ([1x3x192x320xf16, 1x3x192x320xf16, 1x2x192x320xf16]) <- (1x3x192x320xf16, 1x3x192x320xf16, 1x2x192x320xf16)
        combine_11 = [divide__0, cast_15, scale__15]

        # pd_op.concat: (1x8x192x320xf16) <- ([1x3x192x320xf16, 1x3x192x320xf16, 1x2x192x320xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_11, constant_0)

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x8x192x320xf16, 16x8x3x3xf16)
        conv2d_29 = paddle._C_ops.conv2d(concat_2, parameter_71, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__33 = paddle._C_ops.add_(conv2d_29, parameter_72)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__23 = paddle._C_ops.leaky_relu_(add__33, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x16x192x320xf16, 16x16x3x3xf16)
        conv2d_30 = paddle._C_ops.conv2d(leaky_relu__23, parameter_73, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__34 = paddle._C_ops.add_(conv2d_30, parameter_74)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__24 = paddle._C_ops.leaky_relu_(add__34, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf16) <- (1x16x192x320xf16, 32x16x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(leaky_relu__24, parameter_75, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x192x320xf16) <- (1x32x192x320xf16, 1x32x1x1xf16)
        add__35 = paddle._C_ops.add_(conv2d_31, parameter_76)

        # pd_op.leaky_relu_: (1x32x192x320xf16) <- (1x32x192x320xf16)
        leaky_relu__25 = paddle._C_ops.leaky_relu_(add__35, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf16) <- (1x32x192x320xf16, 32x32x3x3xf16)
        conv2d_32 = paddle._C_ops.conv2d(leaky_relu__25, parameter_77, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x192x320xf16) <- (1x32x192x320xf16, 1x32x1x1xf16)
        add__36 = paddle._C_ops.add_(conv2d_32, parameter_78)

        # pd_op.leaky_relu_: (1x32x192x320xf16) <- (1x32x192x320xf16)
        leaky_relu__26 = paddle._C_ops.leaky_relu_(add__36, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf16) <- (1x32x192x320xf16, 32x32x3x3xf16)
        conv2d_33 = paddle._C_ops.conv2d(leaky_relu__26, parameter_79, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x192x320xf16) <- (1x32x192x320xf16, 1x32x1x1xf16)
        add__37 = paddle._C_ops.add_(conv2d_33, parameter_80)

        # pd_op.leaky_relu_: (1x32x192x320xf16) <- (1x32x192x320xf16)
        leaky_relu__27 = paddle._C_ops.leaky_relu_(add__37, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf16) <- (1x32x192x320xf16, 32x32x3x3xf16)
        conv2d_34 = paddle._C_ops.conv2d(leaky_relu__27, parameter_81, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x192x320xf16) <- (1x32x192x320xf16, 1x32x1x1xf16)
        add__38 = paddle._C_ops.add_(conv2d_34, parameter_82)

        # pd_op.leaky_relu_: (1x32x192x320xf16) <- (1x32x192x320xf16)
        leaky_relu__28 = paddle._C_ops.leaky_relu_(add__38, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x32x192x320xf16, 16x32x3x3xf16)
        conv2d_35 = paddle._C_ops.conv2d(leaky_relu__28, parameter_83, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__39 = paddle._C_ops.add_(conv2d_35, parameter_84)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__29 = paddle._C_ops.leaky_relu_(add__39, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x16x192x320xf16, 16x16x3x3xf16)
        conv2d_36 = paddle._C_ops.conv2d(leaky_relu__29, parameter_85, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__40 = paddle._C_ops.add_(conv2d_36, parameter_86)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__30 = paddle._C_ops.leaky_relu_(add__40, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf16) <- (1x16x192x320xf16, 16x16x3x3xf16)
        conv2d_37 = paddle._C_ops.conv2d(leaky_relu__30, parameter_87, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x16x192x320xf16) <- (1x16x192x320xf16, 1x16x1x1xf16)
        add__41 = paddle._C_ops.add_(conv2d_37, parameter_88)

        # pd_op.leaky_relu_: (1x16x192x320xf16) <- (1x16x192x320xf16)
        leaky_relu__31 = paddle._C_ops.leaky_relu_(add__41, float('0.1'))

        # pd_op.conv2d: (1x8x192x320xf16) <- (1x16x192x320xf16, 8x16x3x3xf16)
        conv2d_38 = paddle._C_ops.conv2d(leaky_relu__31, parameter_89, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x192x320xf16) <- (1x8x192x320xf16, 1x8x1x1xf16)
        add__42 = paddle._C_ops.add_(conv2d_38, parameter_90)

        # pd_op.leaky_relu_: (1x8x192x320xf16) <- (1x8x192x320xf16)
        leaky_relu__32 = paddle._C_ops.leaky_relu_(add__42, float('0.1'))

        # pd_op.conv2d: (1x8x192x320xf16) <- (1x8x192x320xf16, 8x8x3x3xf16)
        conv2d_39 = paddle._C_ops.conv2d(leaky_relu__32, parameter_91, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x8x192x320xf16) <- (1x8x192x320xf16, 1x8x1x1xf16)
        add__43 = paddle._C_ops.add_(conv2d_39, parameter_92)

        # pd_op.leaky_relu_: (1x8x192x320xf16) <- (1x8x192x320xf16)
        leaky_relu__33 = paddle._C_ops.leaky_relu_(add__43, float('0.1'))

        # pd_op.conv2d: (1x2x192x320xf16) <- (1x8x192x320xf16, 2x8x3x3xf16)
        conv2d_40 = paddle._C_ops.conv2d(leaky_relu__33, parameter_93, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x2x192x320xf16) <- (1x2x192x320xf16, 1x2x1x1xf16)
        add__44 = paddle._C_ops.add_(conv2d_40, parameter_94)

        # pd_op.add_: (1x2x192x320xf16) <- (1x2x192x320xf16, 1x2x192x320xf16)
        add__45 = paddle._C_ops.add_(scale__15, add__44)

        # pd_op.bilinear_interp: (1x2x180x320xf16) <- (1x2x192x320xf16, None, None, None)
        bilinear_interp_4 = paddle._C_ops.bilinear_interp(add__45, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

        # pd_op.slice: (1x180x320xf16) <- (1x2x180x320xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(bilinear_interp_4, [1], constant_1, constant_5, [1], [1])

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__22 = paddle._C_ops.scale_(slice_18, constant_3, float('0'), True)

        # pd_op.set_value_with_tensor_: (1x2x180x320xf16) <- (1x2x180x320xf16, 1x180x320xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(bilinear_interp_4, scale__22, constant_1, constant_5, constant_5, [1], [1], [])

        # pd_op.slice: (1x180x320xf16) <- (1x2x180x320xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(set_value_with_tensor__0, [1], constant_5, constant_6, [1], [1])

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__23 = paddle._C_ops.scale_(slice_19, constant_16, float('0'), True)

        # pd_op.set_value_with_tensor_: (1x2x180x320xf16) <- (1x2x180x320xf16, 1x180x320xf16, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__0, scale__23, constant_5, constant_6, constant_5, [1], [1], [])

        # pd_op.reshape_: (1x1x2x180x320xf16, 0x1x2x180x320xf16) <- (1x2x180x320xf16, 5xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(set_value_with_tensor__1, constant_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.if: (1x1x2x180x320xf32) <- (xb)
        if select_input_0:
            if_0, = self.pd_op_if_4962_0_0(reshape__6)
        else:
            if_0, = self.pd_op_if_4962_1_0(parameter_95)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(select_input_0)

        # pd_op.if: (1x1x2x180x320xf32) <- (xb)
        if logical_not_0:
            if_1, = self.pd_op_if_4969_0_0(pool2d_2, cast_5, pool2d_3, parameter_18, constant_0, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, constant_9, cast_8, constant_1, constant_5, constant_12, constant_3, constant_6, constant_13, pool2d_0, pool2d_1, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, cast_12, constant_14, constant_15, divide__0, divide__1, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_94, constant_16, constant_17)
        else:
            if_1, = self.pd_op_if_4969_1_0(parameter_96)

        # pd_op.cast: (xi32) <- (xb)
        cast_16 = paddle._C_ops.cast(select_input_0, paddle.int32)

        # pd_op.select_input: (1x1x2x180x320xf32) <- (xi32, 1x1x2x180x320xf32, 1x1x2x180x320xf32)
        select_input_1 = [if_1, if_0][int(cast_16)]

        # pd_op.slice: (1x2x180x320xf32) <- (1x1x2x180x320xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(select_input_1, [1], constant_1, constant_5, [1], [1])

        # pd_op.cast: (1x2x180x320xf16) <- (1x2x180x320xf32)
        cast_17 = paddle._C_ops.cast(slice_20, paddle.float16)

        # pd_op.transpose: (1x180x320x2xf16) <- (1x2x180x320xf16)
        transpose_2 = paddle._C_ops.transpose(cast_17, [0, 2, 3, 1])

        # builtin.combine: ([180xi64, 320xi64]) <- (180xi64, 320xi64)
        combine_12 = [parameter_97, parameter_70]

        # pd_op.meshgrid: ([180x320xi64, 180x320xi64]) <- ([180xi64, 320xi64])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_12)

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_21 = meshgrid_3[1]

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_22 = meshgrid_3[0]

        # builtin.combine: ([180x320xi64, 180x320xi64]) <- (180x320xi64, 180x320xi64)
        combine_13 = [slice_21, slice_22]

        # pd_op.stack: (180x320x2xi64) <- ([180x320xi64, 180x320xi64])
        stack_6 = paddle._C_ops.stack(combine_13, 2)

        # pd_op.cast: (180x320x2xf16) <- (180x320x2xi64)
        cast_18 = paddle._C_ops.cast(stack_6, paddle.float16)

        # pd_op.add: (1x180x320x2xf16) <- (180x320x2xf16, 1x180x320x2xf16)
        add_3 = cast_18 + transpose_2

        # pd_op.slice: (1x180x320xf16) <- (1x180x320x2xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(add_3, [3], constant_1, constant_5, [1], [3])

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__24 = paddle._C_ops.scale_(slice_23, constant_9, float('0'), True)

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__25 = paddle._C_ops.scale_(scale__24, constant_14, float('0'), True)

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__26 = paddle._C_ops.scale_(scale__25, constant_3, float('-1'), True)

        # pd_op.slice: (1x180x320xf16) <- (1x180x320x2xf16, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(add_3, [3], constant_5, constant_6, [1], [3])

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__27 = paddle._C_ops.scale_(slice_24, constant_9, float('0'), True)

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__28 = paddle._C_ops.scale_(scale__27, constant_18, float('0'), True)

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__29 = paddle._C_ops.scale_(scale__28, constant_3, float('-1'), True)

        # builtin.combine: ([1x180x320xf16, 1x180x320xf16]) <- (1x180x320xf16, 1x180x320xf16)
        combine_14 = [scale__26, scale__29]

        # pd_op.stack: (1x180x320x2xf16) <- ([1x180x320xf16, 1x180x320xf16])
        stack_7 = paddle._C_ops.stack(combine_14, 3)

        # pd_op.cast: (1x180x320x2xf32) <- (1x180x320x2xf16)
        cast_19 = paddle._C_ops.cast(stack_7, paddle.float32)

        # pd_op.cast: (1x32x180x320xf32) <- (1x32x180x320xf16)
        cast_20 = paddle._C_ops.cast(slice_2, paddle.float32)

        # pd_op.grid_sample: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x180x320x2xf32)
        grid_sample_3 = paddle._C_ops.grid_sample(cast_20, cast_19, 'bilinear', 'zeros', True)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_21 = paddle._C_ops.cast(grid_sample_3, paddle.float16)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16)
        combine_15 = [cast_21, slice_3]

        # pd_op.concat: (1x64x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_15, constant_0)

        # builtin.combine: ([1x64x180x320xf16, 1x2x180x320xf16]) <- (1x64x180x320xf16, 1x2x180x320xf16)
        combine_16 = [concat_3, cast_17]

        # pd_op.concat: (1x66x180x320xf16) <- ([1x64x180x320xf16, 1x2x180x320xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_16, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x66x180x320xf16, 32x66x3x3xf16)
        conv2d_41 = paddle._C_ops.conv2d(concat_4, parameter_98, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__46 = paddle._C_ops.add_(conv2d_41, parameter_99)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__34 = paddle._C_ops.leaky_relu_(add__46, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_42 = paddle._C_ops.conv2d(leaky_relu__34, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__47 = paddle._C_ops.add_(conv2d_42, parameter_101)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__35 = paddle._C_ops.leaky_relu_(add__47, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_43 = paddle._C_ops.conv2d(leaky_relu__35, parameter_102, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__48 = paddle._C_ops.add_(conv2d_43, parameter_103)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__36 = paddle._C_ops.leaky_relu_(add__48, float('0.1'))

        # pd_op.conv2d: (1x216x180x320xf16) <- (1x32x180x320xf16, 216x32x3x3xf16)
        conv2d_44 = paddle._C_ops.conv2d(leaky_relu__36, parameter_104, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x216x180x320xf16) <- (1x216x180x320xf16, 1x216x1x1xf16)
        add__49 = paddle._C_ops.add_(conv2d_44, parameter_105)

        # pd_op.split_with_num: ([1x72x180x320xf16, 1x72x180x320xf16, 1x72x180x320xf16]) <- (1x216x180x320xf16, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(add__49, 3, constant_0)

        # builtin.slice: (1x72x180x320xf16) <- ([1x72x180x320xf16, 1x72x180x320xf16, 1x72x180x320xf16])
        slice_25 = split_with_num_1[0]

        # builtin.slice: (1x72x180x320xf16) <- ([1x72x180x320xf16, 1x72x180x320xf16, 1x72x180x320xf16])
        slice_26 = split_with_num_1[1]

        # builtin.combine: ([1x72x180x320xf16, 1x72x180x320xf16]) <- (1x72x180x320xf16, 1x72x180x320xf16)
        combine_17 = [slice_25, slice_26]

        # pd_op.concat: (1x144x180x320xf16) <- ([1x72x180x320xf16, 1x72x180x320xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_17, constant_0)

        # pd_op.tanh_: (1x144x180x320xf16) <- (1x144x180x320xf16)
        tanh__0 = paddle._C_ops.tanh_(concat_5)

        # pd_op.scale_: (1x144x180x320xf16) <- (1x144x180x320xf16, 1xf32)
        scale__30 = paddle._C_ops.scale_(tanh__0, constant_19, float('0'), True)

        # pd_op.flip: (1x2x180x320xf16) <- (1x2x180x320xf16)
        flip_1 = paddle._C_ops.flip(cast_17, [1])

        # pd_op.tile: (1x144x180x320xf16) <- (1x2x180x320xf16, 4xi64)
        tile_0 = paddle._C_ops.tile(flip_1, constant_20)

        # pd_op.add_: (1x144x180x320xf16) <- (1x144x180x320xf16, 1x144x180x320xf16)
        add__50 = paddle._C_ops.add_(scale__30, tile_0)

        # builtin.slice: (1x72x180x320xf16) <- ([1x72x180x320xf16, 1x72x180x320xf16, 1x72x180x320xf16])
        slice_27 = split_with_num_1[2]

        # pd_op.sigmoid_: (1x72x180x320xf16) <- (1x72x180x320xf16)
        sigmoid__0 = paddle._C_ops.sigmoid_(slice_27)

        # pd_op.cast: (1x72x180x320xf32) <- (1x72x180x320xf16)
        cast_22 = paddle._C_ops.cast(sigmoid__0, paddle.float32)

        # pd_op.cast: (1x144x180x320xf32) <- (1x144x180x320xf16)
        cast_23 = paddle._C_ops.cast(add__50, paddle.float32)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x144x180x320xf32, 32x32x3x3xf32, 1x72x180x320xf32)
        deformable_conv_0 = paddle._C_ops.deformable_conv(cast_20, cast_23, parameter_106, cast_22, [1, 1], [1, 1], [1, 1], 8, 1, 1)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_24 = paddle._C_ops.cast(deformable_conv_0, paddle.float16)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__51 = paddle._C_ops.add_(cast_24, parameter_107)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_18 = [slice_3, parameter_108, add__51]

        # pd_op.concat: (1x96x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_18, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x96x180x320xf16, 32x96x3x3xf16)
        conv2d_45 = paddle._C_ops.conv2d(concat_6, parameter_109, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__52 = paddle._C_ops.add_(conv2d_45, parameter_110)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__37 = paddle._C_ops.leaky_relu_(add__52, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_46 = paddle._C_ops.conv2d(leaky_relu__37, parameter_111, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__53 = paddle._C_ops.add_(conv2d_46, parameter_112)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__2 = paddle._C_ops.relu_(add__53)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu__2, parameter_113, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__54 = paddle._C_ops.add_(conv2d_47, parameter_114)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__31 = paddle._C_ops.scale_(add__54, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__55 = paddle._C_ops.add_(leaky_relu__37, scale__31)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_48 = paddle._C_ops.conv2d(add__55, parameter_115, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__56 = paddle._C_ops.add_(conv2d_48, parameter_116)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__3 = paddle._C_ops.relu_(add__56)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu__3, parameter_117, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__57 = paddle._C_ops.add_(conv2d_49, parameter_118)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__32 = paddle._C_ops.scale_(add__57, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__58 = paddle._C_ops.add_(add__55, scale__32)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_50 = paddle._C_ops.conv2d(add__58, parameter_119, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__59 = paddle._C_ops.add_(conv2d_50, parameter_120)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__4 = paddle._C_ops.relu_(add__59)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu__4, parameter_121, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__60 = paddle._C_ops.add_(conv2d_51, parameter_122)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__33 = paddle._C_ops.scale_(add__60, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__61 = paddle._C_ops.add_(add__58, scale__33)

        # pd_op.slice: (1x2x180x320xf16) <- (1x1x2x180x320xf16, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(reshape__6, [1], constant_1, constant_5, [1], [1])

        # pd_op.transpose: (1x180x320x2xf16) <- (1x2x180x320xf16)
        transpose_3 = paddle._C_ops.transpose(slice_28, [0, 2, 3, 1])

        # pd_op.add_: (1x180x320x2xf16) <- (180x320x2xf16, 1x180x320x2xf16)
        add__62 = paddle._C_ops.add_(cast_18, transpose_3)

        # pd_op.slice: (1x180x320xf16) <- (1x180x320x2xf16, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(add__62, [3], constant_1, constant_5, [1], [3])

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__34 = paddle._C_ops.scale_(slice_29, constant_9, float('0'), True)

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__35 = paddle._C_ops.scale_(scale__34, constant_14, float('0'), True)

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__36 = paddle._C_ops.scale_(scale__35, constant_3, float('-1'), True)

        # pd_op.slice: (1x180x320xf16) <- (1x180x320x2xf16, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(add__62, [3], constant_5, constant_6, [1], [3])

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__37 = paddle._C_ops.scale_(slice_30, constant_9, float('0'), True)

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__38 = paddle._C_ops.scale_(scale__37, constant_18, float('0'), True)

        # pd_op.scale_: (1x180x320xf16) <- (1x180x320xf16, 1xf32)
        scale__39 = paddle._C_ops.scale_(scale__38, constant_3, float('-1'), True)

        # builtin.combine: ([1x180x320xf16, 1x180x320xf16]) <- (1x180x320xf16, 1x180x320xf16)
        combine_19 = [scale__36, scale__39]

        # pd_op.stack: (1x180x320x2xf16) <- ([1x180x320xf16, 1x180x320xf16])
        stack_8 = paddle._C_ops.stack(combine_19, 3)

        # pd_op.cast: (1x180x320x2xf32) <- (1x180x320x2xf16)
        cast_25 = paddle._C_ops.cast(stack_8, paddle.float32)

        # pd_op.cast: (1x32x180x320xf32) <- (1x32x180x320xf16)
        cast_26 = paddle._C_ops.cast(slice_3, paddle.float32)

        # pd_op.grid_sample: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x180x320x2xf32)
        grid_sample_4 = paddle._C_ops.grid_sample(cast_26, cast_25, 'bilinear', 'zeros', True)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_27 = paddle._C_ops.cast(grid_sample_4, paddle.float16)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16)
        combine_20 = [cast_27, slice_2]

        # pd_op.concat: (1x64x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_20, constant_0)

        # builtin.combine: ([1x64x180x320xf16, 1x2x180x320xf16]) <- (1x64x180x320xf16, 1x2x180x320xf16)
        combine_21 = [concat_7, slice_28]

        # pd_op.concat: (1x66x180x320xf16) <- ([1x64x180x320xf16, 1x2x180x320xf16], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_21, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x66x180x320xf16, 32x66x3x3xf16)
        conv2d_52 = paddle._C_ops.conv2d(concat_8, parameter_98, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__63 = paddle._C_ops.add_(conv2d_52, parameter_99)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__38 = paddle._C_ops.leaky_relu_(add__63, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_53 = paddle._C_ops.conv2d(leaky_relu__38, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__64 = paddle._C_ops.add_(conv2d_53, parameter_101)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__39 = paddle._C_ops.leaky_relu_(add__64, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_54 = paddle._C_ops.conv2d(leaky_relu__39, parameter_102, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__65 = paddle._C_ops.add_(conv2d_54, parameter_103)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__40 = paddle._C_ops.leaky_relu_(add__65, float('0.1'))

        # pd_op.conv2d: (1x216x180x320xf16) <- (1x32x180x320xf16, 216x32x3x3xf16)
        conv2d_55 = paddle._C_ops.conv2d(leaky_relu__40, parameter_104, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x216x180x320xf16) <- (1x216x180x320xf16, 1x216x1x1xf16)
        add__66 = paddle._C_ops.add_(conv2d_55, parameter_105)

        # pd_op.split_with_num: ([1x72x180x320xf16, 1x72x180x320xf16, 1x72x180x320xf16]) <- (1x216x180x320xf16, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(add__66, 3, constant_0)

        # builtin.slice: (1x72x180x320xf16) <- ([1x72x180x320xf16, 1x72x180x320xf16, 1x72x180x320xf16])
        slice_31 = split_with_num_2[0]

        # builtin.slice: (1x72x180x320xf16) <- ([1x72x180x320xf16, 1x72x180x320xf16, 1x72x180x320xf16])
        slice_32 = split_with_num_2[1]

        # builtin.combine: ([1x72x180x320xf16, 1x72x180x320xf16]) <- (1x72x180x320xf16, 1x72x180x320xf16)
        combine_22 = [slice_31, slice_32]

        # pd_op.concat: (1x144x180x320xf16) <- ([1x72x180x320xf16, 1x72x180x320xf16], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_22, constant_0)

        # pd_op.tanh_: (1x144x180x320xf16) <- (1x144x180x320xf16)
        tanh__1 = paddle._C_ops.tanh_(concat_9)

        # pd_op.scale_: (1x144x180x320xf16) <- (1x144x180x320xf16, 1xf32)
        scale__40 = paddle._C_ops.scale_(tanh__1, constant_19, float('0'), True)

        # pd_op.flip: (1x2x180x320xf16) <- (1x2x180x320xf16)
        flip_2 = paddle._C_ops.flip(slice_28, [1])

        # pd_op.tile: (1x144x180x320xf16) <- (1x2x180x320xf16, 4xi64)
        tile_1 = paddle._C_ops.tile(flip_2, constant_20)

        # pd_op.add_: (1x144x180x320xf16) <- (1x144x180x320xf16, 1x144x180x320xf16)
        add__67 = paddle._C_ops.add_(scale__40, tile_1)

        # builtin.slice: (1x72x180x320xf16) <- ([1x72x180x320xf16, 1x72x180x320xf16, 1x72x180x320xf16])
        slice_33 = split_with_num_2[2]

        # pd_op.sigmoid_: (1x72x180x320xf16) <- (1x72x180x320xf16)
        sigmoid__1 = paddle._C_ops.sigmoid_(slice_33)

        # pd_op.cast: (1x72x180x320xf32) <- (1x72x180x320xf16)
        cast_28 = paddle._C_ops.cast(sigmoid__1, paddle.float32)

        # pd_op.cast: (1x144x180x320xf32) <- (1x144x180x320xf16)
        cast_29 = paddle._C_ops.cast(add__67, paddle.float32)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x144x180x320xf32, 32x32x3x3xf32, 1x72x180x320xf32)
        deformable_conv_1 = paddle._C_ops.deformable_conv(cast_26, cast_29, parameter_106, cast_28, [1, 1], [1, 1], [1, 1], 8, 1, 1)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_30 = paddle._C_ops.cast(deformable_conv_1, paddle.float16)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__68 = paddle._C_ops.add_(cast_30, parameter_107)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_23 = [slice_2, add__68, parameter_108]

        # pd_op.concat: (1x96x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_23, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x96x180x320xf16, 32x96x3x3xf16)
        conv2d_56 = paddle._C_ops.conv2d(concat_10, parameter_109, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__69 = paddle._C_ops.add_(conv2d_56, parameter_110)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__41 = paddle._C_ops.leaky_relu_(add__69, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_57 = paddle._C_ops.conv2d(leaky_relu__41, parameter_111, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__70 = paddle._C_ops.add_(conv2d_57, parameter_112)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__5 = paddle._C_ops.relu_(add__70)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_58 = paddle._C_ops.conv2d(relu__5, parameter_113, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__71 = paddle._C_ops.add_(conv2d_58, parameter_114)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__41 = paddle._C_ops.scale_(add__71, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__72 = paddle._C_ops.add_(leaky_relu__41, scale__41)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_59 = paddle._C_ops.conv2d(add__72, parameter_115, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__73 = paddle._C_ops.add_(conv2d_59, parameter_116)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__6 = paddle._C_ops.relu_(add__73)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_60 = paddle._C_ops.conv2d(relu__6, parameter_117, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__74 = paddle._C_ops.add_(conv2d_60, parameter_118)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__42 = paddle._C_ops.scale_(add__74, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__75 = paddle._C_ops.add_(add__72, scale__42)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_61 = paddle._C_ops.conv2d(add__75, parameter_119, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__76 = paddle._C_ops.add_(conv2d_61, parameter_120)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__7 = paddle._C_ops.relu_(add__76)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_62 = paddle._C_ops.conv2d(relu__7, parameter_121, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__77 = paddle._C_ops.add_(conv2d_62, parameter_122)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__43 = paddle._C_ops.scale_(add__77, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__78 = paddle._C_ops.add_(add__75, scale__43)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_24 = [slice_3, add__61, parameter_108]

        # pd_op.concat: (1x96x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_24, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x96x180x320xf16, 32x96x3x3xf16)
        conv2d_63 = paddle._C_ops.conv2d(concat_11, parameter_123, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__79 = paddle._C_ops.add_(conv2d_63, parameter_124)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__42 = paddle._C_ops.leaky_relu_(add__79, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_64 = paddle._C_ops.conv2d(leaky_relu__42, parameter_125, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__80 = paddle._C_ops.add_(conv2d_64, parameter_126)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__8 = paddle._C_ops.relu_(add__80)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_65 = paddle._C_ops.conv2d(relu__8, parameter_127, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__81 = paddle._C_ops.add_(conv2d_65, parameter_128)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__44 = paddle._C_ops.scale_(add__81, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__82 = paddle._C_ops.add_(leaky_relu__42, scale__44)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_66 = paddle._C_ops.conv2d(add__82, parameter_129, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__83 = paddle._C_ops.add_(conv2d_66, parameter_130)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__9 = paddle._C_ops.relu_(add__83)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_67 = paddle._C_ops.conv2d(relu__9, parameter_131, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__84 = paddle._C_ops.add_(conv2d_67, parameter_132)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__45 = paddle._C_ops.scale_(add__84, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__85 = paddle._C_ops.add_(add__82, scale__45)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_68 = paddle._C_ops.conv2d(add__85, parameter_133, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__86 = paddle._C_ops.add_(conv2d_68, parameter_134)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__10 = paddle._C_ops.relu_(add__86)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_69 = paddle._C_ops.conv2d(relu__10, parameter_135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__87 = paddle._C_ops.add_(conv2d_69, parameter_136)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__46 = paddle._C_ops.scale_(add__87, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__88 = paddle._C_ops.add_(add__85, scale__46)

        # pd_op.add: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add_4 = parameter_108 + add__88

        # pd_op.cast: (1x32x180x320xf32) <- (1x32x180x320xf16)
        cast_31 = paddle._C_ops.cast(add_4, paddle.float32)

        # pd_op.grid_sample: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x180x320x2xf32)
        grid_sample_5 = paddle._C_ops.grid_sample(cast_31, cast_25, 'bilinear', 'zeros', True)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_32 = paddle._C_ops.cast(grid_sample_5, paddle.float16)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16)
        combine_25 = [cast_32, slice_2]

        # pd_op.concat: (1x64x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_25, constant_0)

        # builtin.combine: ([1x64x180x320xf16, 1x2x180x320xf16]) <- (1x64x180x320xf16, 1x2x180x320xf16)
        combine_26 = [concat_12, slice_28]

        # pd_op.concat: (1x66x180x320xf16) <- ([1x64x180x320xf16, 1x2x180x320xf16], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_26, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x66x180x320xf16, 32x66x3x3xf16)
        conv2d_70 = paddle._C_ops.conv2d(concat_13, parameter_137, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__89 = paddle._C_ops.add_(conv2d_70, parameter_138)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__43 = paddle._C_ops.leaky_relu_(add__89, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_71 = paddle._C_ops.conv2d(leaky_relu__43, parameter_139, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__90 = paddle._C_ops.add_(conv2d_71, parameter_140)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__44 = paddle._C_ops.leaky_relu_(add__90, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_72 = paddle._C_ops.conv2d(leaky_relu__44, parameter_141, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__91 = paddle._C_ops.add_(conv2d_72, parameter_142)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__45 = paddle._C_ops.leaky_relu_(add__91, float('0.1'))

        # pd_op.conv2d: (1x108x180x320xf16) <- (1x32x180x320xf16, 108x32x3x3xf16)
        conv2d_73 = paddle._C_ops.conv2d(leaky_relu__45, parameter_143, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x108x180x320xf16) <- (1x108x180x320xf16, 1x108x1x1xf16)
        add__92 = paddle._C_ops.add_(conv2d_73, parameter_144)

        # pd_op.split_with_num: ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16]) <- (1x108x180x320xf16, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(add__92, 3, constant_0)

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_34 = split_with_num_3[0]

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_35 = split_with_num_3[1]

        # builtin.combine: ([1x36x180x320xf16, 1x36x180x320xf16]) <- (1x36x180x320xf16, 1x36x180x320xf16)
        combine_27 = [slice_34, slice_35]

        # pd_op.concat: (1x72x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_27, constant_0)

        # pd_op.tanh_: (1x72x180x320xf16) <- (1x72x180x320xf16)
        tanh__2 = paddle._C_ops.tanh_(concat_14)

        # pd_op.scale_: (1x72x180x320xf16) <- (1x72x180x320xf16, 1xf32)
        scale__47 = paddle._C_ops.scale_(tanh__2, constant_19, float('0'), True)

        # pd_op.tile: (1x72x180x320xf16) <- (1x2x180x320xf16, 4xi64)
        tile_2 = paddle._C_ops.tile(flip_2, constant_21)

        # pd_op.add_: (1x72x180x320xf16) <- (1x72x180x320xf16, 1x72x180x320xf16)
        add__93 = paddle._C_ops.add_(scale__47, tile_2)

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_36 = split_with_num_3[2]

        # pd_op.sigmoid_: (1x36x180x320xf16) <- (1x36x180x320xf16)
        sigmoid__2 = paddle._C_ops.sigmoid_(slice_36)

        # pd_op.cast: (1x36x180x320xf32) <- (1x36x180x320xf16)
        cast_33 = paddle._C_ops.cast(sigmoid__2, paddle.float32)

        # pd_op.cast: (1x72x180x320xf32) <- (1x72x180x320xf16)
        cast_34 = paddle._C_ops.cast(add__93, paddle.float32)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_2 = paddle._C_ops.deformable_conv(cast_31, cast_34, parameter_145, cast_33, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_35 = paddle._C_ops.cast(deformable_conv_2, paddle.float16)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__94 = paddle._C_ops.add_(cast_35, parameter_146)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_28 = [slice_2, add__78, add__94]

        # pd_op.concat: (1x96x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_28, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x96x180x320xf16, 32x96x3x3xf16)
        conv2d_74 = paddle._C_ops.conv2d(concat_15, parameter_123, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__95 = paddle._C_ops.add_(conv2d_74, parameter_124)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__46 = paddle._C_ops.leaky_relu_(add__95, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_75 = paddle._C_ops.conv2d(leaky_relu__46, parameter_125, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__96 = paddle._C_ops.add_(conv2d_75, parameter_126)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__11 = paddle._C_ops.relu_(add__96)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_76 = paddle._C_ops.conv2d(relu__11, parameter_127, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__97 = paddle._C_ops.add_(conv2d_76, parameter_128)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__48 = paddle._C_ops.scale_(add__97, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__98 = paddle._C_ops.add_(leaky_relu__46, scale__48)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_77 = paddle._C_ops.conv2d(add__98, parameter_129, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__99 = paddle._C_ops.add_(conv2d_77, parameter_130)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__12 = paddle._C_ops.relu_(add__99)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_78 = paddle._C_ops.conv2d(relu__12, parameter_131, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__100 = paddle._C_ops.add_(conv2d_78, parameter_132)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__49 = paddle._C_ops.scale_(add__100, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__101 = paddle._C_ops.add_(add__98, scale__49)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_79 = paddle._C_ops.conv2d(add__101, parameter_133, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__102 = paddle._C_ops.add_(conv2d_79, parameter_134)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__13 = paddle._C_ops.relu_(add__102)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_80 = paddle._C_ops.conv2d(relu__13, parameter_135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__103 = paddle._C_ops.add_(conv2d_80, parameter_136)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__50 = paddle._C_ops.scale_(add__103, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__104 = paddle._C_ops.add_(add__101, scale__50)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__105 = paddle._C_ops.add_(add__94, add__104)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_29 = [slice_2, add__78, add__105, parameter_108]

        # pd_op.concat: (1x128x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_29, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x128x180x320xf16, 32x128x3x3xf16)
        conv2d_81 = paddle._C_ops.conv2d(concat_16, parameter_147, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__106 = paddle._C_ops.add_(conv2d_81, parameter_148)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__47 = paddle._C_ops.leaky_relu_(add__106, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_82 = paddle._C_ops.conv2d(leaky_relu__47, parameter_149, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__107 = paddle._C_ops.add_(conv2d_82, parameter_150)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__14 = paddle._C_ops.relu_(add__107)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_83 = paddle._C_ops.conv2d(relu__14, parameter_151, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__108 = paddle._C_ops.add_(conv2d_83, parameter_152)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__51 = paddle._C_ops.scale_(add__108, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__109 = paddle._C_ops.add_(leaky_relu__47, scale__51)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_84 = paddle._C_ops.conv2d(add__109, parameter_153, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__110 = paddle._C_ops.add_(conv2d_84, parameter_154)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__15 = paddle._C_ops.relu_(add__110)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_85 = paddle._C_ops.conv2d(relu__15, parameter_155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__111 = paddle._C_ops.add_(conv2d_85, parameter_156)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__52 = paddle._C_ops.scale_(add__111, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__112 = paddle._C_ops.add_(add__109, scale__52)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_86 = paddle._C_ops.conv2d(add__112, parameter_157, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__113 = paddle._C_ops.add_(conv2d_86, parameter_158)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__16 = paddle._C_ops.relu_(add__113)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_87 = paddle._C_ops.conv2d(relu__16, parameter_159, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__114 = paddle._C_ops.add_(conv2d_87, parameter_160)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__53 = paddle._C_ops.scale_(add__114, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__115 = paddle._C_ops.add_(add__112, scale__53)

        # pd_op.add: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add_5 = parameter_108 + add__115

        # pd_op.cast: (1x32x180x320xf32) <- (1x32x180x320xf16)
        cast_36 = paddle._C_ops.cast(add_5, paddle.float32)

        # pd_op.grid_sample: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x180x320x2xf32)
        grid_sample_6 = paddle._C_ops.grid_sample(cast_36, cast_19, 'bilinear', 'zeros', True)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_37 = paddle._C_ops.cast(grid_sample_6, paddle.float16)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16)
        combine_30 = [cast_37, slice_3]

        # pd_op.concat: (1x64x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_30, constant_0)

        # builtin.combine: ([1x64x180x320xf16, 1x2x180x320xf16]) <- (1x64x180x320xf16, 1x2x180x320xf16)
        combine_31 = [concat_17, cast_17]

        # pd_op.concat: (1x66x180x320xf16) <- ([1x64x180x320xf16, 1x2x180x320xf16], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_31, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x66x180x320xf16, 32x66x3x3xf16)
        conv2d_88 = paddle._C_ops.conv2d(concat_18, parameter_161, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__116 = paddle._C_ops.add_(conv2d_88, parameter_162)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__48 = paddle._C_ops.leaky_relu_(add__116, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_89 = paddle._C_ops.conv2d(leaky_relu__48, parameter_163, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__117 = paddle._C_ops.add_(conv2d_89, parameter_164)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__49 = paddle._C_ops.leaky_relu_(add__117, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_90 = paddle._C_ops.conv2d(leaky_relu__49, parameter_165, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__118 = paddle._C_ops.add_(conv2d_90, parameter_166)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__50 = paddle._C_ops.leaky_relu_(add__118, float('0.1'))

        # pd_op.conv2d: (1x108x180x320xf16) <- (1x32x180x320xf16, 108x32x3x3xf16)
        conv2d_91 = paddle._C_ops.conv2d(leaky_relu__50, parameter_167, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x108x180x320xf16) <- (1x108x180x320xf16, 1x108x1x1xf16)
        add__119 = paddle._C_ops.add_(conv2d_91, parameter_168)

        # pd_op.split_with_num: ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16]) <- (1x108x180x320xf16, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(add__119, 3, constant_0)

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_37 = split_with_num_4[0]

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_38 = split_with_num_4[1]

        # builtin.combine: ([1x36x180x320xf16, 1x36x180x320xf16]) <- (1x36x180x320xf16, 1x36x180x320xf16)
        combine_32 = [slice_37, slice_38]

        # pd_op.concat: (1x72x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_32, constant_0)

        # pd_op.tanh_: (1x72x180x320xf16) <- (1x72x180x320xf16)
        tanh__3 = paddle._C_ops.tanh_(concat_19)

        # pd_op.scale_: (1x72x180x320xf16) <- (1x72x180x320xf16, 1xf32)
        scale__54 = paddle._C_ops.scale_(tanh__3, constant_19, float('0'), True)

        # pd_op.tile: (1x72x180x320xf16) <- (1x2x180x320xf16, 4xi64)
        tile_3 = paddle._C_ops.tile(flip_1, constant_21)

        # pd_op.add_: (1x72x180x320xf16) <- (1x72x180x320xf16, 1x72x180x320xf16)
        add__120 = paddle._C_ops.add_(scale__54, tile_3)

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_39 = split_with_num_4[2]

        # pd_op.sigmoid_: (1x36x180x320xf16) <- (1x36x180x320xf16)
        sigmoid__3 = paddle._C_ops.sigmoid_(slice_39)

        # pd_op.cast: (1x36x180x320xf32) <- (1x36x180x320xf16)
        cast_38 = paddle._C_ops.cast(sigmoid__3, paddle.float32)

        # pd_op.cast: (1x72x180x320xf32) <- (1x72x180x320xf16)
        cast_39 = paddle._C_ops.cast(add__120, paddle.float32)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_3 = paddle._C_ops.deformable_conv(cast_36, cast_39, parameter_169, cast_38, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_40 = paddle._C_ops.cast(deformable_conv_3, paddle.float16)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__121 = paddle._C_ops.add_(cast_40, parameter_170)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_33 = [slice_3, add__61, add_4, add__121]

        # pd_op.concat: (1x128x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_33, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x128x180x320xf16, 32x128x3x3xf16)
        conv2d_92 = paddle._C_ops.conv2d(concat_20, parameter_147, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__122 = paddle._C_ops.add_(conv2d_92, parameter_148)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__51 = paddle._C_ops.leaky_relu_(add__122, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_93 = paddle._C_ops.conv2d(leaky_relu__51, parameter_149, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__123 = paddle._C_ops.add_(conv2d_93, parameter_150)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__17 = paddle._C_ops.relu_(add__123)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_94 = paddle._C_ops.conv2d(relu__17, parameter_151, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__124 = paddle._C_ops.add_(conv2d_94, parameter_152)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__55 = paddle._C_ops.scale_(add__124, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__125 = paddle._C_ops.add_(leaky_relu__51, scale__55)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_95 = paddle._C_ops.conv2d(add__125, parameter_153, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__126 = paddle._C_ops.add_(conv2d_95, parameter_154)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__18 = paddle._C_ops.relu_(add__126)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_96 = paddle._C_ops.conv2d(relu__18, parameter_155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__127 = paddle._C_ops.add_(conv2d_96, parameter_156)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__56 = paddle._C_ops.scale_(add__127, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__128 = paddle._C_ops.add_(add__125, scale__56)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_97 = paddle._C_ops.conv2d(add__128, parameter_157, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__129 = paddle._C_ops.add_(conv2d_97, parameter_158)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__19 = paddle._C_ops.relu_(add__129)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_98 = paddle._C_ops.conv2d(relu__19, parameter_159, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__130 = paddle._C_ops.add_(conv2d_98, parameter_160)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__57 = paddle._C_ops.scale_(add__130, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__131 = paddle._C_ops.add_(add__128, scale__57)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__132 = paddle._C_ops.add_(add__121, add__131)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_34 = [slice_2, add__78, add__105, add_5]

        # pd_op.concat: (1x128x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_34, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x128x180x320xf16, 32x128x3x3xf16)
        conv2d_99 = paddle._C_ops.conv2d(concat_21, parameter_171, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__133 = paddle._C_ops.add_(conv2d_99, parameter_172)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__52 = paddle._C_ops.leaky_relu_(add__133, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_100 = paddle._C_ops.conv2d(leaky_relu__52, parameter_173, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__134 = paddle._C_ops.add_(conv2d_100, parameter_174)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__20 = paddle._C_ops.relu_(add__134)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_101 = paddle._C_ops.conv2d(relu__20, parameter_175, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__135 = paddle._C_ops.add_(conv2d_101, parameter_176)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__58 = paddle._C_ops.scale_(add__135, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__136 = paddle._C_ops.add_(leaky_relu__52, scale__58)

        # pd_op.conv2d: (1x128x180x320xf16) <- (1x32x180x320xf16, 128x32x3x3xf16)
        conv2d_102 = paddle._C_ops.conv2d(add__136, parameter_177, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x180x320xf16) <- (1x128x180x320xf16, 1x128x1x1xf16)
        add__137 = paddle._C_ops.add_(conv2d_102, parameter_178)

        # pd_op.pixel_shuffle: (1x32x360x640xf16) <- (1x128x180x320xf16)
        pixel_shuffle_0 = paddle._C_ops.pixel_shuffle(add__137, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf16) <- (1x32x360x640xf16)
        leaky_relu__53 = paddle._C_ops.leaky_relu_(pixel_shuffle_0, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf16) <- (1x32x360x640xf16, 128x32x3x3xf16)
        conv2d_103 = paddle._C_ops.conv2d(leaky_relu__53, parameter_179, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x360x640xf16) <- (1x128x360x640xf16, 1x128x1x1xf16)
        add__138 = paddle._C_ops.add_(conv2d_103, parameter_180)

        # pd_op.pixel_shuffle: (1x32x720x1280xf16) <- (1x128x360x640xf16)
        pixel_shuffle_1 = paddle._C_ops.pixel_shuffle(add__138, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf16) <- (1x32x720x1280xf16)
        leaky_relu__54 = paddle._C_ops.leaky_relu_(pixel_shuffle_1, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf16) <- (1x32x720x1280xf16, 3x32x3x3xf16)
        conv2d_104 = paddle._C_ops.conv2d(leaky_relu__54, parameter_181, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x1x1xf16)
        add__139 = paddle._C_ops.add_(conv2d_104, parameter_182)

        # pd_op.slice: (1x3x180x320xf16) <- (1x2x3x180x320xf16, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(cast_0, [1], constant_1, constant_5, [1], [1])

        # pd_op.bilinear_interp: (1x3x720x1280xf16) <- (1x3x180x320xf16, None, None, None)
        bilinear_interp_5 = paddle._C_ops.bilinear_interp(slice_40, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x720x1280xf16)
        add__140 = paddle._C_ops.add_(add__139, bilinear_interp_5)

        # builtin.combine: ([1x3x720x1280xf16, 1x32x720x1280xf16]) <- (1x3x720x1280xf16, 1x32x720x1280xf16)
        combine_35 = [add__140, leaky_relu__54]

        # pd_op.concat: (1x35x720x1280xf16) <- ([1x3x720x1280xf16, 1x32x720x1280xf16], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_35, constant_0)

        # pd_op.conv2d: (1x32x360x640xf16) <- (1x35x720x1280xf16, 32x35x3x3xf16)
        conv2d_105 = paddle._C_ops.conv2d(concat_22, parameter_183, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x360x640xf16) <- (1x32x360x640xf16, 1x32x1x1xf16)
        add__141 = paddle._C_ops.add_(conv2d_105, parameter_184)

        # pd_op.leaky_relu_: (1x32x360x640xf16) <- (1x32x360x640xf16)
        leaky_relu__55 = paddle._C_ops.leaky_relu_(add__141, float('0.1'))

        # pd_op.conv2d: (1x32x360x640xf16) <- (1x32x360x640xf16, 32x32x3x3xf16)
        conv2d_106 = paddle._C_ops.conv2d(leaky_relu__55, parameter_185, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x360x640xf16) <- (1x32x360x640xf16, 1x32x1x1xf16)
        add__142 = paddle._C_ops.add_(conv2d_106, parameter_186)

        # builtin.combine: ([1x32x360x640xf16, 1x32x360x640xf16]) <- (1x32x360x640xf16, 1x32x360x640xf16)
        combine_36 = [add__142, leaky_relu__53]

        # pd_op.concat: (1x64x360x640xf16) <- ([1x32x360x640xf16, 1x32x360x640xf16], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_36, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x64x360x640xf16, 32x64x3x3xf16)
        conv2d_107 = paddle._C_ops.conv2d(concat_23, parameter_187, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__143 = paddle._C_ops.add_(conv2d_107, parameter_188)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__56 = paddle._C_ops.leaky_relu_(add__143, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_108 = paddle._C_ops.conv2d(leaky_relu__56, parameter_189, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__144 = paddle._C_ops.add_(conv2d_108, parameter_190)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16)
        combine_37 = [add__144, add__136]

        # pd_op.concat: (1x64x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_37, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x64x180x320xf16, 32x64x3x3xf16)
        conv2d_109 = paddle._C_ops.conv2d(concat_24, parameter_191, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__145 = paddle._C_ops.add_(conv2d_109, parameter_192)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_38 = [slice_3, add__61, add_4, add__132]

        # pd_op.concat: (1x128x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_38, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x128x180x320xf16, 32x128x3x3xf16)
        conv2d_110 = paddle._C_ops.conv2d(concat_25, parameter_171, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__146 = paddle._C_ops.add_(conv2d_110, parameter_172)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__57 = paddle._C_ops.leaky_relu_(add__146, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_111 = paddle._C_ops.conv2d(leaky_relu__57, parameter_173, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__147 = paddle._C_ops.add_(conv2d_111, parameter_174)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__21 = paddle._C_ops.relu_(add__147)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_112 = paddle._C_ops.conv2d(relu__21, parameter_175, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__148 = paddle._C_ops.add_(conv2d_112, parameter_176)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__59 = paddle._C_ops.scale_(add__148, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__149 = paddle._C_ops.add_(leaky_relu__57, scale__59)

        # pd_op.conv2d: (1x128x180x320xf16) <- (1x32x180x320xf16, 128x32x3x3xf16)
        conv2d_113 = paddle._C_ops.conv2d(add__149, parameter_177, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x180x320xf16) <- (1x128x180x320xf16, 1x128x1x1xf16)
        add__150 = paddle._C_ops.add_(conv2d_113, parameter_178)

        # pd_op.pixel_shuffle: (1x32x360x640xf16) <- (1x128x180x320xf16)
        pixel_shuffle_2 = paddle._C_ops.pixel_shuffle(add__150, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf16) <- (1x32x360x640xf16)
        leaky_relu__58 = paddle._C_ops.leaky_relu_(pixel_shuffle_2, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf16) <- (1x32x360x640xf16, 128x32x3x3xf16)
        conv2d_114 = paddle._C_ops.conv2d(leaky_relu__58, parameter_179, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x360x640xf16) <- (1x128x360x640xf16, 1x128x1x1xf16)
        add__151 = paddle._C_ops.add_(conv2d_114, parameter_180)

        # pd_op.pixel_shuffle: (1x32x720x1280xf16) <- (1x128x360x640xf16)
        pixel_shuffle_3 = paddle._C_ops.pixel_shuffle(add__151, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf16) <- (1x32x720x1280xf16)
        leaky_relu__59 = paddle._C_ops.leaky_relu_(pixel_shuffle_3, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf16) <- (1x32x720x1280xf16, 3x32x3x3xf16)
        conv2d_115 = paddle._C_ops.conv2d(leaky_relu__59, parameter_181, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x1x1xf16)
        add__152 = paddle._C_ops.add_(conv2d_115, parameter_182)

        # pd_op.slice: (1x3x180x320xf16) <- (1x2x3x180x320xf16, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(cast_0, [1], constant_5, constant_6, [1], [1])

        # pd_op.bilinear_interp: (1x3x720x1280xf16) <- (1x3x180x320xf16, None, None, None)
        bilinear_interp_6 = paddle._C_ops.bilinear_interp(slice_41, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x720x1280xf16)
        add__153 = paddle._C_ops.add_(add__152, bilinear_interp_6)

        # builtin.combine: ([1x3x720x1280xf16, 1x32x720x1280xf16]) <- (1x3x720x1280xf16, 1x32x720x1280xf16)
        combine_39 = [add__153, leaky_relu__59]

        # pd_op.concat: (1x35x720x1280xf16) <- ([1x3x720x1280xf16, 1x32x720x1280xf16], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_39, constant_0)

        # pd_op.conv2d: (1x32x360x640xf16) <- (1x35x720x1280xf16, 32x35x3x3xf16)
        conv2d_116 = paddle._C_ops.conv2d(concat_26, parameter_183, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x360x640xf16) <- (1x32x360x640xf16, 1x32x1x1xf16)
        add__154 = paddle._C_ops.add_(conv2d_116, parameter_184)

        # pd_op.leaky_relu_: (1x32x360x640xf16) <- (1x32x360x640xf16)
        leaky_relu__60 = paddle._C_ops.leaky_relu_(add__154, float('0.1'))

        # pd_op.conv2d: (1x32x360x640xf16) <- (1x32x360x640xf16, 32x32x3x3xf16)
        conv2d_117 = paddle._C_ops.conv2d(leaky_relu__60, parameter_185, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x360x640xf16) <- (1x32x360x640xf16, 1x32x1x1xf16)
        add__155 = paddle._C_ops.add_(conv2d_117, parameter_186)

        # builtin.combine: ([1x32x360x640xf16, 1x32x360x640xf16]) <- (1x32x360x640xf16, 1x32x360x640xf16)
        combine_40 = [add__155, leaky_relu__58]

        # pd_op.concat: (1x64x360x640xf16) <- ([1x32x360x640xf16, 1x32x360x640xf16], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_40, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x64x360x640xf16, 32x64x3x3xf16)
        conv2d_118 = paddle._C_ops.conv2d(concat_27, parameter_187, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__156 = paddle._C_ops.add_(conv2d_118, parameter_188)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__61 = paddle._C_ops.leaky_relu_(add__156, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_119 = paddle._C_ops.conv2d(leaky_relu__61, parameter_189, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__157 = paddle._C_ops.add_(conv2d_119, parameter_190)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16)
        combine_41 = [add__157, add__149]

        # pd_op.concat: (1x64x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_41, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x64x180x320xf16, 32x64x3x3xf16)
        conv2d_120 = paddle._C_ops.conv2d(concat_28, parameter_191, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__158 = paddle._C_ops.add_(conv2d_120, parameter_192)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_42 = [add__158, add__61, add_4, add__132, parameter_108]

        # pd_op.concat: (1x160x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_42, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x160x180x320xf16, 32x160x3x3xf16)
        conv2d_121 = paddle._C_ops.conv2d(concat_29, parameter_193, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__159 = paddle._C_ops.add_(conv2d_121, parameter_194)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__62 = paddle._C_ops.leaky_relu_(add__159, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_122 = paddle._C_ops.conv2d(leaky_relu__62, parameter_195, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__160 = paddle._C_ops.add_(conv2d_122, parameter_196)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__22 = paddle._C_ops.relu_(add__160)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_123 = paddle._C_ops.conv2d(relu__22, parameter_197, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__161 = paddle._C_ops.add_(conv2d_123, parameter_198)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__60 = paddle._C_ops.scale_(add__161, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__162 = paddle._C_ops.add_(leaky_relu__62, scale__60)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_124 = paddle._C_ops.conv2d(add__162, parameter_199, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__163 = paddle._C_ops.add_(conv2d_124, parameter_200)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__23 = paddle._C_ops.relu_(add__163)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_125 = paddle._C_ops.conv2d(relu__23, parameter_201, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__164 = paddle._C_ops.add_(conv2d_125, parameter_202)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__61 = paddle._C_ops.scale_(add__164, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__165 = paddle._C_ops.add_(add__162, scale__61)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_126 = paddle._C_ops.conv2d(add__165, parameter_203, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__166 = paddle._C_ops.add_(conv2d_126, parameter_204)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__24 = paddle._C_ops.relu_(add__166)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_127 = paddle._C_ops.conv2d(relu__24, parameter_205, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__167 = paddle._C_ops.add_(conv2d_127, parameter_206)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__62 = paddle._C_ops.scale_(add__167, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__168 = paddle._C_ops.add_(add__165, scale__62)

        # pd_op.add: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add_6 = parameter_108 + add__168

        # pd_op.cast: (1x32x180x320xf32) <- (1x32x180x320xf16)
        cast_41 = paddle._C_ops.cast(add_6, paddle.float32)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_4 = paddle._C_ops.deformable_conv(cast_41, cast_34, parameter_207, cast_33, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_42 = paddle._C_ops.cast(deformable_conv_4, paddle.float16)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__169 = paddle._C_ops.add_(cast_42, parameter_208)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x2x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x2x180x320xf16)
        combine_43 = [add__169, add__145, slice_28]

        # pd_op.concat: (1x66x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x2x180x320xf16], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_43, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x66x180x320xf16, 32x66x3x3xf16)
        conv2d_128 = paddle._C_ops.conv2d(concat_30, parameter_209, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__170 = paddle._C_ops.add_(conv2d_128, parameter_210)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__63 = paddle._C_ops.leaky_relu_(add__170, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_129 = paddle._C_ops.conv2d(leaky_relu__63, parameter_211, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__171 = paddle._C_ops.add_(conv2d_129, parameter_212)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__64 = paddle._C_ops.leaky_relu_(add__171, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_130 = paddle._C_ops.conv2d(leaky_relu__64, parameter_213, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__172 = paddle._C_ops.add_(conv2d_130, parameter_214)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__65 = paddle._C_ops.leaky_relu_(add__172, float('0.1'))

        # pd_op.conv2d: (1x108x180x320xf16) <- (1x32x180x320xf16, 108x32x3x3xf16)
        conv2d_131 = paddle._C_ops.conv2d(leaky_relu__65, parameter_215, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x108x180x320xf16) <- (1x108x180x320xf16, 1x108x1x1xf16)
        add__173 = paddle._C_ops.add_(conv2d_131, parameter_216)

        # pd_op.split_with_num: ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16]) <- (1x108x180x320xf16, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(add__173, 3, constant_0)

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_42 = split_with_num_5[0]

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_43 = split_with_num_5[1]

        # builtin.combine: ([1x36x180x320xf16, 1x36x180x320xf16]) <- (1x36x180x320xf16, 1x36x180x320xf16)
        combine_44 = [slice_42, slice_43]

        # pd_op.concat: (1x72x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_44, constant_0)

        # pd_op.tanh_: (1x72x180x320xf16) <- (1x72x180x320xf16)
        tanh__4 = paddle._C_ops.tanh_(concat_31)

        # pd_op.scale_: (1x72x180x320xf16) <- (1x72x180x320xf16, 1xf32)
        scale__63 = paddle._C_ops.scale_(tanh__4, constant_19, float('0'), True)

        # pd_op.add_: (1x72x180x320xf16) <- (1x72x180x320xf16, 1x72x180x320xf16)
        add__174 = paddle._C_ops.add_(scale__63, add__93)

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_44 = split_with_num_5[2]

        # pd_op.sigmoid_: (1x36x180x320xf16) <- (1x36x180x320xf16)
        sigmoid__4 = paddle._C_ops.sigmoid_(slice_44)

        # pd_op.add_: (1x36x180x320xf16) <- (1x36x180x320xf16, 1x36x180x320xf16)
        add__175 = paddle._C_ops.add_(sigmoid__4, sigmoid__2)

        # pd_op.scale_: (1x36x180x320xf16) <- (1x36x180x320xf16, 1xf32)
        scale__64 = paddle._C_ops.scale_(add__175, constant_22, float('0'), True)

        # pd_op.cast: (1x36x180x320xf32) <- (1x36x180x320xf16)
        cast_43 = paddle._C_ops.cast(scale__64, paddle.float32)

        # pd_op.cast: (1x72x180x320xf32) <- (1x72x180x320xf16)
        cast_44 = paddle._C_ops.cast(add__174, paddle.float32)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_5 = paddle._C_ops.deformable_conv(cast_41, cast_44, parameter_217, cast_43, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_45 = paddle._C_ops.cast(deformable_conv_5, paddle.float16)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__176 = paddle._C_ops.add_(cast_45, parameter_218)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_45 = [add__145, add__78, add__105, add_5, add__176]

        # pd_op.concat: (1x160x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_45, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x160x180x320xf16, 32x160x3x3xf16)
        conv2d_132 = paddle._C_ops.conv2d(concat_32, parameter_193, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__177 = paddle._C_ops.add_(conv2d_132, parameter_194)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__66 = paddle._C_ops.leaky_relu_(add__177, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_133 = paddle._C_ops.conv2d(leaky_relu__66, parameter_195, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__178 = paddle._C_ops.add_(conv2d_133, parameter_196)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__25 = paddle._C_ops.relu_(add__178)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_134 = paddle._C_ops.conv2d(relu__25, parameter_197, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__179 = paddle._C_ops.add_(conv2d_134, parameter_198)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__65 = paddle._C_ops.scale_(add__179, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__180 = paddle._C_ops.add_(leaky_relu__66, scale__65)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_135 = paddle._C_ops.conv2d(add__180, parameter_199, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__181 = paddle._C_ops.add_(conv2d_135, parameter_200)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__26 = paddle._C_ops.relu_(add__181)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_136 = paddle._C_ops.conv2d(relu__26, parameter_201, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__182 = paddle._C_ops.add_(conv2d_136, parameter_202)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__66 = paddle._C_ops.scale_(add__182, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__183 = paddle._C_ops.add_(add__180, scale__66)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_137 = paddle._C_ops.conv2d(add__183, parameter_203, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__184 = paddle._C_ops.add_(conv2d_137, parameter_204)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__27 = paddle._C_ops.relu_(add__184)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_138 = paddle._C_ops.conv2d(relu__27, parameter_205, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__185 = paddle._C_ops.add_(conv2d_138, parameter_206)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__67 = paddle._C_ops.scale_(add__185, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__186 = paddle._C_ops.add_(add__183, scale__67)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__187 = paddle._C_ops.add_(add__176, add__186)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_46 = [add__145, add__78, add__105, add_5, add__187, parameter_108]

        # pd_op.concat: (1x192x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_46, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x192x180x320xf16, 32x192x3x3xf16)
        conv2d_139 = paddle._C_ops.conv2d(concat_33, parameter_219, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__188 = paddle._C_ops.add_(conv2d_139, parameter_220)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__67 = paddle._C_ops.leaky_relu_(add__188, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_140 = paddle._C_ops.conv2d(leaky_relu__67, parameter_221, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__189 = paddle._C_ops.add_(conv2d_140, parameter_222)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__28 = paddle._C_ops.relu_(add__189)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_141 = paddle._C_ops.conv2d(relu__28, parameter_223, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__190 = paddle._C_ops.add_(conv2d_141, parameter_224)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__68 = paddle._C_ops.scale_(add__190, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__191 = paddle._C_ops.add_(leaky_relu__67, scale__68)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_142 = paddle._C_ops.conv2d(add__191, parameter_225, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__192 = paddle._C_ops.add_(conv2d_142, parameter_226)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__29 = paddle._C_ops.relu_(add__192)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_143 = paddle._C_ops.conv2d(relu__29, parameter_227, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__193 = paddle._C_ops.add_(conv2d_143, parameter_228)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__69 = paddle._C_ops.scale_(add__193, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__194 = paddle._C_ops.add_(add__191, scale__69)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_144 = paddle._C_ops.conv2d(add__194, parameter_229, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__195 = paddle._C_ops.add_(conv2d_144, parameter_230)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__30 = paddle._C_ops.relu_(add__195)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_145 = paddle._C_ops.conv2d(relu__30, parameter_231, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__196 = paddle._C_ops.add_(conv2d_145, parameter_232)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__70 = paddle._C_ops.scale_(add__196, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__197 = paddle._C_ops.add_(add__194, scale__70)

        # pd_op.add: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add_7 = parameter_108 + add__197

        # pd_op.cast: (1x32x180x320xf32) <- (1x32x180x320xf16)
        cast_46 = paddle._C_ops.cast(add_7, paddle.float32)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_6 = paddle._C_ops.deformable_conv(cast_46, cast_39, parameter_233, cast_38, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_47 = paddle._C_ops.cast(deformable_conv_6, paddle.float16)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__198 = paddle._C_ops.add_(cast_47, parameter_234)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x2x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x2x180x320xf16)
        combine_47 = [add__198, add__158, cast_17]

        # pd_op.concat: (1x66x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x2x180x320xf16], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_47, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x66x180x320xf16, 32x66x3x3xf16)
        conv2d_146 = paddle._C_ops.conv2d(concat_34, parameter_235, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__199 = paddle._C_ops.add_(conv2d_146, parameter_236)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__68 = paddle._C_ops.leaky_relu_(add__199, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_147 = paddle._C_ops.conv2d(leaky_relu__68, parameter_237, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__200 = paddle._C_ops.add_(conv2d_147, parameter_238)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__69 = paddle._C_ops.leaky_relu_(add__200, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_148 = paddle._C_ops.conv2d(leaky_relu__69, parameter_239, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__201 = paddle._C_ops.add_(conv2d_148, parameter_240)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__70 = paddle._C_ops.leaky_relu_(add__201, float('0.1'))

        # pd_op.conv2d: (1x108x180x320xf16) <- (1x32x180x320xf16, 108x32x3x3xf16)
        conv2d_149 = paddle._C_ops.conv2d(leaky_relu__70, parameter_241, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x108x180x320xf16) <- (1x108x180x320xf16, 1x108x1x1xf16)
        add__202 = paddle._C_ops.add_(conv2d_149, parameter_242)

        # pd_op.split_with_num: ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16]) <- (1x108x180x320xf16, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(add__202, 3, constant_0)

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_45 = split_with_num_6[0]

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_46 = split_with_num_6[1]

        # builtin.combine: ([1x36x180x320xf16, 1x36x180x320xf16]) <- (1x36x180x320xf16, 1x36x180x320xf16)
        combine_48 = [slice_45, slice_46]

        # pd_op.concat: (1x72x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_48, constant_0)

        # pd_op.tanh_: (1x72x180x320xf16) <- (1x72x180x320xf16)
        tanh__5 = paddle._C_ops.tanh_(concat_35)

        # pd_op.scale_: (1x72x180x320xf16) <- (1x72x180x320xf16, 1xf32)
        scale__71 = paddle._C_ops.scale_(tanh__5, constant_19, float('0'), True)

        # pd_op.add_: (1x72x180x320xf16) <- (1x72x180x320xf16, 1x72x180x320xf16)
        add__203 = paddle._C_ops.add_(scale__71, add__120)

        # builtin.slice: (1x36x180x320xf16) <- ([1x36x180x320xf16, 1x36x180x320xf16, 1x36x180x320xf16])
        slice_47 = split_with_num_6[2]

        # pd_op.sigmoid_: (1x36x180x320xf16) <- (1x36x180x320xf16)
        sigmoid__5 = paddle._C_ops.sigmoid_(slice_47)

        # pd_op.add_: (1x36x180x320xf16) <- (1x36x180x320xf16, 1x36x180x320xf16)
        add__204 = paddle._C_ops.add_(sigmoid__5, sigmoid__3)

        # pd_op.scale_: (1x36x180x320xf16) <- (1x36x180x320xf16, 1xf32)
        scale__72 = paddle._C_ops.scale_(add__204, constant_22, float('0'), True)

        # pd_op.cast: (1x36x180x320xf32) <- (1x36x180x320xf16)
        cast_48 = paddle._C_ops.cast(scale__72, paddle.float32)

        # pd_op.cast: (1x72x180x320xf32) <- (1x72x180x320xf16)
        cast_49 = paddle._C_ops.cast(add__203, paddle.float32)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_7 = paddle._C_ops.deformable_conv(cast_46, cast_49, parameter_243, cast_48, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.cast: (1x32x180x320xf16) <- (1x32x180x320xf32)
        cast_50 = paddle._C_ops.cast(deformable_conv_7, paddle.float16)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__205 = paddle._C_ops.add_(cast_50, parameter_244)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_49 = [add__158, add__61, add_4, add__132, add_6, add__205]

        # pd_op.concat: (1x192x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_49, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x192x180x320xf16, 32x192x3x3xf16)
        conv2d_150 = paddle._C_ops.conv2d(concat_36, parameter_219, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__206 = paddle._C_ops.add_(conv2d_150, parameter_220)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__71 = paddle._C_ops.leaky_relu_(add__206, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_151 = paddle._C_ops.conv2d(leaky_relu__71, parameter_221, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__207 = paddle._C_ops.add_(conv2d_151, parameter_222)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__31 = paddle._C_ops.relu_(add__207)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_152 = paddle._C_ops.conv2d(relu__31, parameter_223, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__208 = paddle._C_ops.add_(conv2d_152, parameter_224)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__73 = paddle._C_ops.scale_(add__208, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__209 = paddle._C_ops.add_(leaky_relu__71, scale__73)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_153 = paddle._C_ops.conv2d(add__209, parameter_225, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__210 = paddle._C_ops.add_(conv2d_153, parameter_226)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__32 = paddle._C_ops.relu_(add__210)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_154 = paddle._C_ops.conv2d(relu__32, parameter_227, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__211 = paddle._C_ops.add_(conv2d_154, parameter_228)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__74 = paddle._C_ops.scale_(add__211, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__212 = paddle._C_ops.add_(add__209, scale__74)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_155 = paddle._C_ops.conv2d(add__212, parameter_229, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__213 = paddle._C_ops.add_(conv2d_155, parameter_230)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__33 = paddle._C_ops.relu_(add__213)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_156 = paddle._C_ops.conv2d(relu__33, parameter_231, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__214 = paddle._C_ops.add_(conv2d_156, parameter_232)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__75 = paddle._C_ops.scale_(add__214, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__215 = paddle._C_ops.add_(add__212, scale__75)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__216 = paddle._C_ops.add_(add__205, add__215)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_50 = [slice_2, add__78, add__105, add_5, add__187, add_7]

        # pd_op.concat: (1x192x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_50, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x192x180x320xf16, 32x192x3x3xf16)
        conv2d_157 = paddle._C_ops.conv2d(concat_37, parameter_245, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__217 = paddle._C_ops.add_(conv2d_157, parameter_246)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__72 = paddle._C_ops.leaky_relu_(add__217, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_158 = paddle._C_ops.conv2d(leaky_relu__72, parameter_247, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__218 = paddle._C_ops.add_(conv2d_158, parameter_248)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__34 = paddle._C_ops.relu_(add__218)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_159 = paddle._C_ops.conv2d(relu__34, parameter_249, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__219 = paddle._C_ops.add_(conv2d_159, parameter_250)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__76 = paddle._C_ops.scale_(add__219, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__220 = paddle._C_ops.add_(leaky_relu__72, scale__76)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_160 = paddle._C_ops.conv2d(add__220, parameter_251, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__221 = paddle._C_ops.add_(conv2d_160, parameter_252)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__35 = paddle._C_ops.relu_(add__221)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_161 = paddle._C_ops.conv2d(relu__35, parameter_253, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__222 = paddle._C_ops.add_(conv2d_161, parameter_254)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__77 = paddle._C_ops.scale_(add__222, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__223 = paddle._C_ops.add_(add__220, scale__77)

        # pd_op.conv2d: (1x128x180x320xf16) <- (1x32x180x320xf16, 128x32x3x3xf16)
        conv2d_162 = paddle._C_ops.conv2d(add__223, parameter_255, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x180x320xf16) <- (1x128x180x320xf16, 1x128x1x1xf16)
        add__224 = paddle._C_ops.add_(conv2d_162, parameter_256)

        # pd_op.pixel_shuffle: (1x32x360x640xf16) <- (1x128x180x320xf16)
        pixel_shuffle_4 = paddle._C_ops.pixel_shuffle(add__224, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf16) <- (1x32x360x640xf16)
        leaky_relu__73 = paddle._C_ops.leaky_relu_(pixel_shuffle_4, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf16) <- (1x32x360x640xf16, 128x32x3x3xf16)
        conv2d_163 = paddle._C_ops.conv2d(leaky_relu__73, parameter_257, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x360x640xf16) <- (1x128x360x640xf16, 1x128x1x1xf16)
        add__225 = paddle._C_ops.add_(conv2d_163, parameter_258)

        # pd_op.pixel_shuffle: (1x32x720x1280xf16) <- (1x128x360x640xf16)
        pixel_shuffle_5 = paddle._C_ops.pixel_shuffle(add__225, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf16) <- (1x32x720x1280xf16)
        leaky_relu__74 = paddle._C_ops.leaky_relu_(pixel_shuffle_5, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf16) <- (1x32x720x1280xf16, 3x32x3x3xf16)
        conv2d_164 = paddle._C_ops.conv2d(leaky_relu__74, parameter_259, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x1x1xf16)
        add__226 = paddle._C_ops.add_(conv2d_164, parameter_260)

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x720x1280xf16)
        add__227 = paddle._C_ops.add_(add__226, bilinear_interp_5)

        # pd_op.conv2d: (1x128x180x320xf16) <- (1x32x180x320xf16, 128x32x3x3xf16)
        conv2d_165 = paddle._C_ops.conv2d(add__78, parameter_177, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x180x320xf16) <- (1x128x180x320xf16, 1x128x1x1xf16)
        add__228 = paddle._C_ops.add_(conv2d_165, parameter_178)

        # pd_op.pixel_shuffle: (1x32x360x640xf16) <- (1x128x180x320xf16)
        pixel_shuffle_6 = paddle._C_ops.pixel_shuffle(add__228, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf16) <- (1x32x360x640xf16)
        leaky_relu__75 = paddle._C_ops.leaky_relu_(pixel_shuffle_6, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf16) <- (1x32x360x640xf16, 128x32x3x3xf16)
        conv2d_166 = paddle._C_ops.conv2d(leaky_relu__75, parameter_179, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x360x640xf16) <- (1x128x360x640xf16, 1x128x1x1xf16)
        add__229 = paddle._C_ops.add_(conv2d_166, parameter_180)

        # pd_op.pixel_shuffle: (1x32x720x1280xf16) <- (1x128x360x640xf16)
        pixel_shuffle_7 = paddle._C_ops.pixel_shuffle(add__229, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf16) <- (1x32x720x1280xf16)
        leaky_relu__76 = paddle._C_ops.leaky_relu_(pixel_shuffle_7, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf16) <- (1x32x720x1280xf16, 3x32x3x3xf16)
        conv2d_167 = paddle._C_ops.conv2d(leaky_relu__76, parameter_261, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x1x1xf16)
        add__230 = paddle._C_ops.add_(conv2d_167, parameter_262)

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x720x1280xf16)
        add__231 = paddle._C_ops.add_(add__230, add__227)

        # builtin.combine: ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16]) <- (1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16)
        combine_51 = [slice_3, add__61, add_4, add__132, add_6, add__216]

        # pd_op.concat: (1x192x180x320xf16) <- ([1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16, 1x32x180x320xf16], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_51, constant_0)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x192x180x320xf16, 32x192x3x3xf16)
        conv2d_168 = paddle._C_ops.conv2d(concat_38, parameter_245, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__232 = paddle._C_ops.add_(conv2d_168, parameter_246)

        # pd_op.leaky_relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        leaky_relu__77 = paddle._C_ops.leaky_relu_(add__232, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_169 = paddle._C_ops.conv2d(leaky_relu__77, parameter_247, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__233 = paddle._C_ops.add_(conv2d_169, parameter_248)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__36 = paddle._C_ops.relu_(add__233)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_170 = paddle._C_ops.conv2d(relu__36, parameter_249, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__234 = paddle._C_ops.add_(conv2d_170, parameter_250)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__78 = paddle._C_ops.scale_(add__234, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__235 = paddle._C_ops.add_(leaky_relu__77, scale__78)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_171 = paddle._C_ops.conv2d(add__235, parameter_251, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__236 = paddle._C_ops.add_(conv2d_171, parameter_252)

        # pd_op.relu_: (1x32x180x320xf16) <- (1x32x180x320xf16)
        relu__37 = paddle._C_ops.relu_(add__236)

        # pd_op.conv2d: (1x32x180x320xf16) <- (1x32x180x320xf16, 32x32x3x3xf16)
        conv2d_172 = paddle._C_ops.conv2d(relu__37, parameter_253, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x1x1xf16)
        add__237 = paddle._C_ops.add_(conv2d_172, parameter_254)

        # pd_op.scale_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1xf32)
        scale__79 = paddle._C_ops.scale_(add__237, constant_3, float('0'), True)

        # pd_op.add_: (1x32x180x320xf16) <- (1x32x180x320xf16, 1x32x180x320xf16)
        add__238 = paddle._C_ops.add_(add__235, scale__79)

        # pd_op.conv2d: (1x128x180x320xf16) <- (1x32x180x320xf16, 128x32x3x3xf16)
        conv2d_173 = paddle._C_ops.conv2d(add__238, parameter_255, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x180x320xf16) <- (1x128x180x320xf16, 1x128x1x1xf16)
        add__239 = paddle._C_ops.add_(conv2d_173, parameter_256)

        # pd_op.pixel_shuffle: (1x32x360x640xf16) <- (1x128x180x320xf16)
        pixel_shuffle_8 = paddle._C_ops.pixel_shuffle(add__239, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf16) <- (1x32x360x640xf16)
        leaky_relu__78 = paddle._C_ops.leaky_relu_(pixel_shuffle_8, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf16) <- (1x32x360x640xf16, 128x32x3x3xf16)
        conv2d_174 = paddle._C_ops.conv2d(leaky_relu__78, parameter_257, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x360x640xf16) <- (1x128x360x640xf16, 1x128x1x1xf16)
        add__240 = paddle._C_ops.add_(conv2d_174, parameter_258)

        # pd_op.pixel_shuffle: (1x32x720x1280xf16) <- (1x128x360x640xf16)
        pixel_shuffle_9 = paddle._C_ops.pixel_shuffle(add__240, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf16) <- (1x32x720x1280xf16)
        leaky_relu__79 = paddle._C_ops.leaky_relu_(pixel_shuffle_9, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf16) <- (1x32x720x1280xf16, 3x32x3x3xf16)
        conv2d_175 = paddle._C_ops.conv2d(leaky_relu__79, parameter_259, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x1x1xf16)
        add__241 = paddle._C_ops.add_(conv2d_175, parameter_260)

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x720x1280xf16)
        add__242 = paddle._C_ops.add_(add__241, bilinear_interp_6)

        # pd_op.conv2d: (1x128x180x320xf16) <- (1x32x180x320xf16, 128x32x3x3xf16)
        conv2d_176 = paddle._C_ops.conv2d(add__61, parameter_177, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x180x320xf16) <- (1x128x180x320xf16, 1x128x1x1xf16)
        add__243 = paddle._C_ops.add_(conv2d_176, parameter_178)

        # pd_op.pixel_shuffle: (1x32x360x640xf16) <- (1x128x180x320xf16)
        pixel_shuffle_10 = paddle._C_ops.pixel_shuffle(add__243, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf16) <- (1x32x360x640xf16)
        leaky_relu__80 = paddle._C_ops.leaky_relu_(pixel_shuffle_10, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf16) <- (1x32x360x640xf16, 128x32x3x3xf16)
        conv2d_177 = paddle._C_ops.conv2d(leaky_relu__80, parameter_179, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x360x640xf16) <- (1x128x360x640xf16, 1x128x1x1xf16)
        add__244 = paddle._C_ops.add_(conv2d_177, parameter_180)

        # pd_op.pixel_shuffle: (1x32x720x1280xf16) <- (1x128x360x640xf16)
        pixel_shuffle_11 = paddle._C_ops.pixel_shuffle(add__244, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf16) <- (1x32x720x1280xf16)
        leaky_relu__81 = paddle._C_ops.leaky_relu_(pixel_shuffle_11, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf16) <- (1x32x720x1280xf16, 3x32x3x3xf16)
        conv2d_178 = paddle._C_ops.conv2d(leaky_relu__81, parameter_261, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x1x1xf16)
        add__245 = paddle._C_ops.add_(conv2d_178, parameter_262)

        # pd_op.add_: (1x3x720x1280xf16) <- (1x3x720x1280xf16, 1x3x720x1280xf16)
        add__246 = paddle._C_ops.add_(add__245, add__242)

        # builtin.combine: ([1x3x720x1280xf16, 1x3x720x1280xf16]) <- (1x3x720x1280xf16, 1x3x720x1280xf16)
        combine_52 = [add__140, add__153]

        # pd_op.stack: (1x2x3x720x1280xf16) <- ([1x3x720x1280xf16, 1x3x720x1280xf16])
        stack_9 = paddle._C_ops.stack(combine_52, 1)

        # builtin.combine: ([1x3x720x1280xf16, 1x3x720x1280xf16]) <- (1x3x720x1280xf16, 1x3x720x1280xf16)
        combine_53 = [add__231, add__246]

        # pd_op.stack: (1x2x3x720x1280xf16) <- ([1x3x720x1280xf16, 1x3x720x1280xf16])
        stack_10 = paddle._C_ops.stack(combine_53, 1)

        # pd_op.cast: (1x2x3x720x1280xf32) <- (1x2x3x720x1280xf16)
        cast_51 = paddle._C_ops.cast(stack_9, paddle.float32)

        # pd_op.cast: (1x2x3x720x1280xf32) <- (1x2x3x720x1280xf16)
        cast_52 = paddle._C_ops.cast(stack_10, paddle.float32)
        return cast_51, cast_52



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

    def forward(self, parameter_96, parameter_95, parameter_262, parameter_260, parameter_258, parameter_256, parameter_254, parameter_252, parameter_250, parameter_248, parameter_246, parameter_244, parameter_242, parameter_240, parameter_238, parameter_236, parameter_234, parameter_232, parameter_230, parameter_228, parameter_226, parameter_224, parameter_222, parameter_220, parameter_218, constant_22, parameter_216, parameter_214, parameter_212, parameter_210, parameter_208, parameter_206, parameter_204, parameter_202, parameter_200, parameter_198, parameter_196, parameter_194, parameter_192, parameter_190, parameter_188, parameter_186, parameter_184, parameter_182, parameter_180, parameter_178, parameter_176, parameter_174, parameter_172, parameter_170, parameter_168, parameter_166, parameter_164, parameter_162, parameter_160, parameter_158, parameter_156, parameter_154, parameter_152, parameter_150, parameter_148, parameter_146, constant_21, parameter_144, parameter_142, parameter_140, parameter_138, parameter_136, parameter_134, parameter_132, parameter_130, parameter_128, parameter_126, parameter_124, parameter_122, parameter_120, parameter_118, parameter_116, parameter_114, parameter_112, parameter_110, parameter_107, constant_20, constant_19, parameter_105, parameter_103, parameter_101, parameter_99, constant_18, parameter_97, parameter_108, constant_17, constant_16, parameter_94, parameter_92, parameter_90, parameter_88, parameter_86, parameter_84, parameter_82, parameter_80, parameter_78, parameter_76, parameter_74, parameter_72, constant_15, constant_14, parameter_70, parameter_69, parameter_68, parameter_66, parameter_64, parameter_62, parameter_60, parameter_58, parameter_56, parameter_54, parameter_52, parameter_50, parameter_48, parameter_46, constant_13, constant_12, parameter_44, parameter_43, parameter_42, parameter_40, parameter_38, parameter_36, parameter_34, parameter_32, parameter_30, parameter_28, parameter_26, parameter_24, parameter_22, parameter_20, constant_11, constant_10, constant_9, parameter_16, parameter_15, parameter_17, parameter_18, constant_8, constant_7, constant_6, constant_5, constant_4, parameter_12, parameter_10, constant_3, parameter_8, parameter_6, parameter_4, constant_2, parameter_2, parameter_1, parameter_0, constant_1, constant_0, parameter_3, parameter_5, parameter_7, parameter_9, parameter_11, parameter_13, parameter_14, parameter_19, parameter_21, parameter_23, parameter_25, parameter_27, parameter_29, parameter_31, parameter_33, parameter_35, parameter_37, parameter_39, parameter_41, parameter_45, parameter_47, parameter_49, parameter_51, parameter_53, parameter_55, parameter_57, parameter_59, parameter_61, parameter_63, parameter_65, parameter_67, parameter_71, parameter_73, parameter_75, parameter_77, parameter_79, parameter_81, parameter_83, parameter_85, parameter_87, parameter_89, parameter_91, parameter_93, parameter_98, parameter_100, parameter_102, parameter_104, parameter_106, parameter_109, parameter_111, parameter_113, parameter_115, parameter_117, parameter_119, parameter_121, parameter_123, parameter_125, parameter_127, parameter_129, parameter_131, parameter_133, parameter_135, parameter_137, parameter_139, parameter_141, parameter_143, parameter_145, parameter_147, parameter_149, parameter_151, parameter_153, parameter_155, parameter_157, parameter_159, parameter_161, parameter_163, parameter_165, parameter_167, parameter_169, parameter_171, parameter_173, parameter_175, parameter_177, parameter_179, parameter_181, parameter_183, parameter_185, parameter_187, parameter_189, parameter_191, parameter_193, parameter_195, parameter_197, parameter_199, parameter_201, parameter_203, parameter_205, parameter_207, parameter_209, parameter_211, parameter_213, parameter_215, parameter_217, parameter_219, parameter_221, parameter_223, parameter_225, parameter_227, parameter_229, parameter_231, parameter_233, parameter_235, parameter_237, parameter_239, parameter_241, parameter_243, parameter_245, parameter_247, parameter_249, parameter_251, parameter_253, parameter_255, parameter_257, parameter_259, parameter_261, feed_0):
        return self.builtin_module_4432_0_0(parameter_96, parameter_95, parameter_262, parameter_260, parameter_258, parameter_256, parameter_254, parameter_252, parameter_250, parameter_248, parameter_246, parameter_244, parameter_242, parameter_240, parameter_238, parameter_236, parameter_234, parameter_232, parameter_230, parameter_228, parameter_226, parameter_224, parameter_222, parameter_220, parameter_218, constant_22, parameter_216, parameter_214, parameter_212, parameter_210, parameter_208, parameter_206, parameter_204, parameter_202, parameter_200, parameter_198, parameter_196, parameter_194, parameter_192, parameter_190, parameter_188, parameter_186, parameter_184, parameter_182, parameter_180, parameter_178, parameter_176, parameter_174, parameter_172, parameter_170, parameter_168, parameter_166, parameter_164, parameter_162, parameter_160, parameter_158, parameter_156, parameter_154, parameter_152, parameter_150, parameter_148, parameter_146, constant_21, parameter_144, parameter_142, parameter_140, parameter_138, parameter_136, parameter_134, parameter_132, parameter_130, parameter_128, parameter_126, parameter_124, parameter_122, parameter_120, parameter_118, parameter_116, parameter_114, parameter_112, parameter_110, parameter_107, constant_20, constant_19, parameter_105, parameter_103, parameter_101, parameter_99, constant_18, parameter_97, parameter_108, constant_17, constant_16, parameter_94, parameter_92, parameter_90, parameter_88, parameter_86, parameter_84, parameter_82, parameter_80, parameter_78, parameter_76, parameter_74, parameter_72, constant_15, constant_14, parameter_70, parameter_69, parameter_68, parameter_66, parameter_64, parameter_62, parameter_60, parameter_58, parameter_56, parameter_54, parameter_52, parameter_50, parameter_48, parameter_46, constant_13, constant_12, parameter_44, parameter_43, parameter_42, parameter_40, parameter_38, parameter_36, parameter_34, parameter_32, parameter_30, parameter_28, parameter_26, parameter_24, parameter_22, parameter_20, constant_11, constant_10, constant_9, parameter_16, parameter_15, parameter_17, parameter_18, constant_8, constant_7, constant_6, constant_5, constant_4, parameter_12, parameter_10, constant_3, parameter_8, parameter_6, parameter_4, constant_2, parameter_2, parameter_1, parameter_0, constant_1, constant_0, parameter_3, parameter_5, parameter_7, parameter_9, parameter_11, parameter_13, parameter_14, parameter_19, parameter_21, parameter_23, parameter_25, parameter_27, parameter_29, parameter_31, parameter_33, parameter_35, parameter_37, parameter_39, parameter_41, parameter_45, parameter_47, parameter_49, parameter_51, parameter_53, parameter_55, parameter_57, parameter_59, parameter_61, parameter_63, parameter_65, parameter_67, parameter_71, parameter_73, parameter_75, parameter_77, parameter_79, parameter_81, parameter_83, parameter_85, parameter_87, parameter_89, parameter_91, parameter_93, parameter_98, parameter_100, parameter_102, parameter_104, parameter_106, parameter_109, parameter_111, parameter_113, parameter_115, parameter_117, parameter_119, parameter_121, parameter_123, parameter_125, parameter_127, parameter_129, parameter_131, parameter_133, parameter_135, parameter_137, parameter_139, parameter_141, parameter_143, parameter_145, parameter_147, parameter_149, parameter_151, parameter_153, parameter_155, parameter_157, parameter_159, parameter_161, parameter_163, parameter_165, parameter_167, parameter_169, parameter_171, parameter_173, parameter_175, parameter_177, parameter_179, parameter_181, parameter_183, parameter_185, parameter_187, parameter_189, parameter_191, parameter_193, parameter_195, parameter_197, parameter_199, parameter_201, parameter_203, parameter_205, parameter_207, parameter_209, parameter_211, parameter_213, parameter_215, parameter_217, parameter_219, parameter_221, parameter_223, parameter_225, parameter_227, parameter_229, parameter_231, parameter_233, parameter_235, parameter_237, parameter_239, parameter_241, parameter_243, parameter_245, parameter_247, parameter_249, parameter_251, parameter_253, parameter_255, parameter_257, parameter_259, parameter_261, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_4432_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_96
            paddle.uniform([1, 1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([1, 1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([1, 3, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_260
            paddle.uniform([1, 3, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_258
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_252
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_250
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_248
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_246
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_242
            paddle.uniform([1, 108, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_240
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_236
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_232
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_230
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_228
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_226
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_222
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_220
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_22
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
            # parameter_216
            paddle.uniform([1, 108, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_212
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_210
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_206
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_200
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_198
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_192
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_182
            paddle.uniform([1, 3, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_180
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_176
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_170
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_168
            paddle.uniform([1, 108, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_162
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_160
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_150
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_21
            paddle.to_tensor([1, 36, 1, 1], dtype='int64').reshape([4]),
            # parameter_144
            paddle.uniform([1, 108, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_142
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_140
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_132
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_122
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_120
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_112
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_110
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_20
            paddle.to_tensor([1, 72, 1, 1], dtype='int64').reshape([4]),
            # constant_19
            paddle.to_tensor([10.0], dtype='float32').reshape([1]),
            # parameter_105
            paddle.uniform([1, 216, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_18
            paddle.to_tensor([0.00558659], dtype='float32').reshape([1]),
            # parameter_97
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179], dtype='int64').reshape([180]),
            # parameter_108
            paddle.uniform([1, 32, 180, 320], dtype='float16', min=0, max=0.5),
            # constant_17
            paddle.to_tensor([1, 1, 2, 180, 320], dtype='int64').reshape([5]),
            # constant_16
            paddle.to_tensor([0.9375], dtype='float32').reshape([1]),
            # parameter_94
            paddle.uniform([1, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_92
            paddle.uniform([1, 8, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([1, 8, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_82
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_15
            paddle.to_tensor([0.0052356], dtype='float32').reshape([1]),
            # constant_14
            paddle.to_tensor([0.0031348], dtype='float32').reshape([1]),
            # parameter_70
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319], dtype='int64').reshape([320]),
            # parameter_69
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191], dtype='int64').reshape([192]),
            # parameter_68
            paddle.uniform([1, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_66
            paddle.uniform([1, 8, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([1, 8, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_62
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_60
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_48
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_46
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_13
            paddle.to_tensor([0.0105263], dtype='float32').reshape([1]),
            # constant_12
            paddle.to_tensor([0.00628931], dtype='float32').reshape([1]),
            # parameter_44
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159], dtype='int64').reshape([160]),
            # parameter_43
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95], dtype='int64').reshape([96]),
            # parameter_42
            paddle.uniform([1, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_40
            paddle.uniform([1, 8, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([1, 8, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_36
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_32
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_30
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_26
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_11
            paddle.to_tensor([0.0212766], dtype='float32').reshape([1]),
            # constant_10
            paddle.to_tensor([0.0126582], dtype='float32').reshape([1]),
            # constant_9
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
            # parameter_16
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79], dtype='int64').reshape([80]),
            # parameter_15
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], dtype='int64').reshape([48]),
            # parameter_17
            paddle.uniform([1, 48, 80, 2], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([1, 2, 48, 80], dtype='float16', min=0, max=0.5),
            # constant_8
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            # constant_7
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            # constant_6
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_5
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_4
            paddle.to_tensor([1, 2, -1, 180, 320], dtype='int64').reshape([5]),
            # parameter_12
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_10
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_3
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            # parameter_8
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_6
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_2
            paddle.to_tensor([-1, 3, 180, 320], dtype='int64').reshape([4]),
            # parameter_2
            paddle.cast(paddle.randint(low=0, high=2, shape=[], dtype='int32'), 'bool'),
            # parameter_1
            paddle.cast(paddle.randint(low=0, high=2, shape=[], dtype='int32'), 'bool'),
            # parameter_0
            paddle.uniform([], dtype='float16', min=0, max=0.5),
            # constant_1
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # constant_0
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # parameter_3
            paddle.uniform([32, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([1, 3, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([1, 3, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([16, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([16, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([32, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_25
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([16, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([16, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([16, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([8, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([8, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([2, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([16, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_47
            paddle.uniform([16, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([32, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([16, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([16, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([16, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([8, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([8, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([2, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([16, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([16, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([32, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_77
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([16, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([16, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([16, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([8, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([8, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([2, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_98
            paddle.uniform([32, 66, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_100
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([216, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([32, 96, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([32, 96, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([32, 66, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([108, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([32, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([32, 66, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([108, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([32, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_175
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_181
            paddle.uniform([3, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([32, 35, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_185
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_187
            paddle.uniform([32, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([32, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([32, 160, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_195
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_197
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([32, 66, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([108, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([32, 192, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_231
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_233
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([32, 66, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_237
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([108, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([32, 192, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_247
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_255
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_257
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([3, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_261
            paddle.uniform([3, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 2, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_96
            paddle.static.InputSpec(shape=[1, 1, 2, 180, 320], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[1, 1, 2, 180, 320], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float16'),
            # parameter_260
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float16'),
            # parameter_258
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_252
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_250
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_248
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_246
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_242
            paddle.static.InputSpec(shape=[1, 108, 1, 1], dtype='float16'),
            # parameter_240
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_236
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_232
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_230
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_228
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_226
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_222
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_220
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # constant_22
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[1, 108, 1, 1], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_212
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_210
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_206
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_202
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_200
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_198
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_192
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_182
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float16'),
            # parameter_180
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_176
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_170
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_168
            paddle.static.InputSpec(shape=[1, 108, 1, 1], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_162
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_160
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_150
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # constant_21
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # parameter_144
            paddle.static.InputSpec(shape=[1, 108, 1, 1], dtype='float16'),
            # parameter_142
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_140
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_132
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_122
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_120
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_112
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_110
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # constant_20
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_19
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[1, 216, 1, 1], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # constant_18
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[180], dtype='int64'),
            # parameter_108
            paddle.static.InputSpec(shape=[1, 32, 180, 320], dtype='float16'),
            # constant_17
            paddle.static.InputSpec(shape=[5], dtype='int64'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[1, 2, 1, 1], dtype='float16'),
            # parameter_92
            paddle.static.InputSpec(shape=[1, 8, 1, 1], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[1, 8, 1, 1], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_82
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # constant_15
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_14
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[320], dtype='int64'),
            # parameter_69
            paddle.static.InputSpec(shape=[192], dtype='int64'),
            # parameter_68
            paddle.static.InputSpec(shape=[1, 2, 1, 1], dtype='float16'),
            # parameter_66
            paddle.static.InputSpec(shape=[1, 8, 1, 1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[1, 8, 1, 1], dtype='float16'),
            # parameter_62
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_60
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_48
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_46
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[160], dtype='int64'),
            # parameter_43
            paddle.static.InputSpec(shape=[96], dtype='int64'),
            # parameter_42
            paddle.static.InputSpec(shape=[1, 2, 1, 1], dtype='float16'),
            # parameter_40
            paddle.static.InputSpec(shape=[1, 8, 1, 1], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[1, 8, 1, 1], dtype='float16'),
            # parameter_36
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_32
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_30
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_26
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float16'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_10
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[80], dtype='int64'),
            # parameter_15
            paddle.static.InputSpec(shape=[48], dtype='int64'),
            # parameter_17
            paddle.static.InputSpec(shape=[1, 48, 80, 2], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[1, 2, 48, 80], dtype='float16'),
            # constant_8
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_7
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_6
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_5
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_4
            paddle.static.InputSpec(shape=[5], dtype='int64'),
            # parameter_12
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_10
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_6
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float16'),
            # constant_2
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # parameter_2
            paddle.static.InputSpec(shape=[], dtype='bool'),
            # parameter_1
            paddle.static.InputSpec(shape=[], dtype='bool'),
            # parameter_0
            paddle.static.InputSpec(shape=[], dtype='float16'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float16'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_7
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[16, 8, 3, 3], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[32, 16, 3, 3], dtype='float16'),
            # parameter_25
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[16, 32, 3, 3], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[8, 16, 3, 3], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[8, 8, 3, 3], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[2, 8, 3, 3], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[16, 8, 3, 3], dtype='float16'),
            # parameter_47
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[32, 16, 3, 3], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[16, 32, 3, 3], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[8, 16, 3, 3], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[8, 8, 3, 3], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[2, 8, 3, 3], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[16, 8, 3, 3], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[32, 16, 3, 3], dtype='float16'),
            # parameter_77
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[16, 32, 3, 3], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[8, 16, 3, 3], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[8, 8, 3, 3], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[2, 8, 3, 3], dtype='float16'),
            # parameter_98
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float16'),
            # parameter_100
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[216, 32, 3, 3], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[32, 96, 3, 3], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[32, 96, 3, 3], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[108, 32, 3, 3], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[32, 128, 3, 3], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[108, 32, 3, 3], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[32, 128, 3, 3], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_175
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_181
            paddle.static.InputSpec(shape=[3, 32, 3, 3], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[32, 35, 3, 3], dtype='float16'),
            # parameter_185
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_187
            paddle.static.InputSpec(shape=[32, 64, 3, 3], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[32, 64, 3, 3], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[32, 160, 3, 3], dtype='float16'),
            # parameter_195
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_197
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[108, 32, 3, 3], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[32, 192, 3, 3], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_231
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_233
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float16'),
            # parameter_237
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[108, 32, 3, 3], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[32, 192, 3, 3], dtype='float16'),
            # parameter_247
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_255
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_257
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[3, 32, 3, 3], dtype='float16'),
            # parameter_261
            paddle.static.InputSpec(shape=[3, 32, 3, 3], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[1, 2, 3, 180, 320], dtype='float32'),
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