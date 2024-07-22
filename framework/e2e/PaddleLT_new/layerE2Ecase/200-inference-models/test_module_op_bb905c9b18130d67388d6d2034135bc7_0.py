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
    return [91][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_647_0_0(self, constant_0, parameter_55, parameter_53, parameter_51, parameter_49, parameter_47, parameter_45, parameter_43, parameter_41, parameter_39, parameter_37, parameter_35, parameter_33, parameter_31, parameter_29, parameter_27, parameter_25, parameter_23, parameter_21, parameter_19, parameter_17, parameter_15, parameter_13, parameter_11, parameter_9, parameter_7, parameter_5, parameter_3, parameter_1, parameter_0, parameter_2, parameter_4, parameter_6, parameter_8, parameter_10, parameter_12, parameter_14, parameter_16, parameter_18, parameter_20, parameter_22, parameter_24, parameter_26, parameter_28, parameter_30, parameter_32, parameter_34, parameter_36, parameter_38, parameter_40, parameter_42, parameter_44, parameter_46, parameter_48, parameter_50, parameter_52, parameter_54, parameter_56, parameter_57, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x64x112x112xf16) <- (-1x3x224x224xf16, 64x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x112x112xf16) <- (-1x64x112x112xf16, 1x64x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, parameter_1)

        # pd_op.relu_: (-1x64x112x112xf16) <- (-1x64x112x112xf16)
        relu__0 = paddle._C_ops.relu_(add__0)

        # pd_op.conv2d: (-1x192x56x56xf16) <- (-1x64x112x112xf16, 192x64x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_2, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x56x56xf16) <- (-1x192x56x56xf16, 1x192x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_1, parameter_3)

        # pd_op.relu_: (-1x192x56x56xf16) <- (-1x192x56x56xf16)
        relu__1 = paddle._C_ops.relu_(add__1)

        # pd_op.conv2d: (-1x192x56x56xf16) <- (-1x192x56x56xf16, 192x192x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x56x56xf16) <- (-1x192x56x56xf16, 1x192x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_2, parameter_5)

        # pd_op.relu_: (-1x192x56x56xf16) <- (-1x192x56x56xf16)
        relu__2 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (-1x192x56x56xf16) <- (-1x192x56x56xf16, 192x192x3x3xf16)
        conv2d_3 = paddle._C_ops.conv2d(relu__2, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x56x56xf16) <- (-1x192x56x56xf16, 1x192x1x1xf16)
        add__3 = paddle._C_ops.add_(conv2d_3, parameter_7)

        # pd_op.relu_: (-1x192x56x56xf16) <- (-1x192x56x56xf16)
        relu__3 = paddle._C_ops.relu_(add__3)

        # pd_op.conv2d: (-1x192x56x56xf16) <- (-1x192x56x56xf16, 192x192x3x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__3, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x56x56xf16) <- (-1x192x56x56xf16, 1x192x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_4, parameter_9)

        # pd_op.relu_: (-1x192x56x56xf16) <- (-1x192x56x56xf16)
        relu__4 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (-1x384x28x28xf16) <- (-1x192x56x56xf16, 384x192x3x3xf16)
        conv2d_5 = paddle._C_ops.conv2d(relu__4, parameter_10, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 1x384x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_5, parameter_11)

        # pd_op.relu_: (-1x384x28x28xf16) <- (-1x384x28x28xf16)
        relu__5 = paddle._C_ops.relu_(add__5)

        # pd_op.conv2d: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 384x384x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(relu__5, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 1x384x1x1xf16)
        add__6 = paddle._C_ops.add_(conv2d_6, parameter_13)

        # pd_op.relu_: (-1x384x28x28xf16) <- (-1x384x28x28xf16)
        relu__6 = paddle._C_ops.relu_(add__6)

        # pd_op.conv2d: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 384x384x3x3xf16)
        conv2d_7 = paddle._C_ops.conv2d(relu__6, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 1x384x1x1xf16)
        add__7 = paddle._C_ops.add_(conv2d_7, parameter_15)

        # pd_op.relu_: (-1x384x28x28xf16) <- (-1x384x28x28xf16)
        relu__7 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 384x384x3x3xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu__7, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 1x384x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_8, parameter_17)

        # pd_op.relu_: (-1x384x28x28xf16) <- (-1x384x28x28xf16)
        relu__8 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 384x384x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__8, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 1x384x1x1xf16)
        add__9 = paddle._C_ops.add_(conv2d_9, parameter_19)

        # pd_op.relu_: (-1x384x28x28xf16) <- (-1x384x28x28xf16)
        relu__9 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 384x384x3x3xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu__9, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x28x28xf16) <- (-1x384x28x28xf16, 1x384x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_10, parameter_21)

        # pd_op.relu_: (-1x384x28x28xf16) <- (-1x384x28x28xf16)
        relu__10 = paddle._C_ops.relu_(add__10)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x384x28x28xf16, 768x384x3x3xf16)
        conv2d_11 = paddle._C_ops.conv2d(relu__10, parameter_22, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__11 = paddle._C_ops.add_(conv2d_11, parameter_23)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__11 = paddle._C_ops.relu_(add__11)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu__11, parameter_24, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__12 = paddle._C_ops.add_(conv2d_12, parameter_25)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__12 = paddle._C_ops.relu_(add__12)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu__12, parameter_26, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__13 = paddle._C_ops.add_(conv2d_13, parameter_27)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__13 = paddle._C_ops.relu_(add__13)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu__13, parameter_28, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_14, parameter_29)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__14 = paddle._C_ops.relu_(add__14)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu__14, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__15 = paddle._C_ops.add_(conv2d_15, parameter_31)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__15 = paddle._C_ops.relu_(add__15)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(relu__15, parameter_32, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__16 = paddle._C_ops.add_(conv2d_16, parameter_33)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__16 = paddle._C_ops.relu_(add__16)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu__16, parameter_34, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_17, parameter_35)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__17 = paddle._C_ops.relu_(add__17)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_18 = paddle._C_ops.conv2d(relu__17, parameter_36, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__18 = paddle._C_ops.add_(conv2d_18, parameter_37)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__18 = paddle._C_ops.relu_(add__18)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu__18, parameter_38, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__19 = paddle._C_ops.add_(conv2d_19, parameter_39)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__19 = paddle._C_ops.relu_(add__19)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_20 = paddle._C_ops.conv2d(relu__19, parameter_40, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__20 = paddle._C_ops.add_(conv2d_20, parameter_41)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__20 = paddle._C_ops.relu_(add__20)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu__20, parameter_42, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__21 = paddle._C_ops.add_(conv2d_21, parameter_43)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__21 = paddle._C_ops.relu_(add__21)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(relu__21, parameter_44, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__22 = paddle._C_ops.add_(conv2d_22, parameter_45)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__22 = paddle._C_ops.relu_(add__22)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu__22, parameter_46, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__23 = paddle._C_ops.add_(conv2d_23, parameter_47)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__23 = paddle._C_ops.relu_(add__23)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_24 = paddle._C_ops.conv2d(relu__23, parameter_48, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__24 = paddle._C_ops.add_(conv2d_24, parameter_49)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__24 = paddle._C_ops.relu_(add__24)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__24, parameter_50, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__25 = paddle._C_ops.add_(conv2d_25, parameter_51)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__25 = paddle._C_ops.relu_(add__25)

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 768x768x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(relu__25, parameter_52, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__26 = paddle._C_ops.add_(conv2d_26, parameter_53)

        # pd_op.relu_: (-1x768x14x14xf16) <- (-1x768x14x14xf16)
        relu__26 = paddle._C_ops.relu_(add__26)

        # pd_op.conv2d: (-1x2560x7x7xf16) <- (-1x768x14x14xf16, 2560x768x3x3xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu__26, parameter_54, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x2560x7x7xf16) <- (-1x2560x7x7xf16, 1x2560x1x1xf16)
        add__27 = paddle._C_ops.add_(conv2d_27, parameter_55)

        # pd_op.relu_: (-1x2560x7x7xf16) <- (-1x2560x7x7xf16)
        relu__27 = paddle._C_ops.relu_(add__27)

        # pd_op.pool2d: (-1x2560x1x1xf16) <- (-1x2560x7x7xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__27, constant_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.flatten_: (-1x2560xf16, None) <- (-1x2560x1x1xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(pool2d_0, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x2560xf16, 2560x1000xf16)
        matmul_0 = paddle.matmul(flatten__0, parameter_56, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__28 = paddle._C_ops.add_(matmul_0, parameter_57)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__0 = paddle._C_ops.softmax_(add__28, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__0, paddle.float32)
        return cast_1



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

    def forward(self, constant_0, parameter_55, parameter_53, parameter_51, parameter_49, parameter_47, parameter_45, parameter_43, parameter_41, parameter_39, parameter_37, parameter_35, parameter_33, parameter_31, parameter_29, parameter_27, parameter_25, parameter_23, parameter_21, parameter_19, parameter_17, parameter_15, parameter_13, parameter_11, parameter_9, parameter_7, parameter_5, parameter_3, parameter_1, parameter_0, parameter_2, parameter_4, parameter_6, parameter_8, parameter_10, parameter_12, parameter_14, parameter_16, parameter_18, parameter_20, parameter_22, parameter_24, parameter_26, parameter_28, parameter_30, parameter_32, parameter_34, parameter_36, parameter_38, parameter_40, parameter_42, parameter_44, parameter_46, parameter_48, parameter_50, parameter_52, parameter_54, parameter_56, parameter_57, feed_0):
        return self.builtin_module_647_0_0(constant_0, parameter_55, parameter_53, parameter_51, parameter_49, parameter_47, parameter_45, parameter_43, parameter_41, parameter_39, parameter_37, parameter_35, parameter_33, parameter_31, parameter_29, parameter_27, parameter_25, parameter_23, parameter_21, parameter_19, parameter_17, parameter_15, parameter_13, parameter_11, parameter_9, parameter_7, parameter_5, parameter_3, parameter_1, parameter_0, parameter_2, parameter_4, parameter_6, parameter_8, parameter_10, parameter_12, parameter_14, parameter_16, parameter_18, parameter_20, parameter_22, parameter_24, parameter_26, parameter_28, parameter_30, parameter_32, parameter_34, parameter_36, parameter_38, parameter_40, parameter_42, parameter_44, parameter_46, parameter_48, parameter_50, parameter_52, parameter_54, parameter_56, parameter_57, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_647_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_0
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # parameter_55
            paddle.uniform([1, 2560, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_47
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_25
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_17
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_7
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_5
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_3
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_0
            paddle.uniform([64, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_2
            paddle.uniform([192, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([192, 192, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_6
            paddle.uniform([192, 192, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_8
            paddle.uniform([192, 192, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_10
            paddle.uniform([384, 192, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_12
            paddle.uniform([384, 384, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([384, 384, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_16
            paddle.uniform([384, 384, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([384, 384, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([384, 384, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([768, 384, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_26
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_30
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_32
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_36
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_40
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_42
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_46
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_48
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([768, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([2560, 768, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([2560, 1000], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_0
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_55
            paddle.static.InputSpec(shape=[1, 2560, 1, 1], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_47
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_25
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_17
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float16'),
            # parameter_7
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float16'),
            # parameter_5
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float16'),
            # parameter_3
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float16'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float16'),
            # parameter_0
            paddle.static.InputSpec(shape=[64, 3, 3, 3], dtype='float16'),
            # parameter_2
            paddle.static.InputSpec(shape=[192, 64, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float16'),
            # parameter_6
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float16'),
            # parameter_8
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float16'),
            # parameter_10
            paddle.static.InputSpec(shape=[384, 192, 3, 3], dtype='float16'),
            # parameter_12
            paddle.static.InputSpec(shape=[384, 384, 3, 3], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[384, 384, 3, 3], dtype='float16'),
            # parameter_16
            paddle.static.InputSpec(shape=[384, 384, 3, 3], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[384, 384, 3, 3], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[384, 384, 3, 3], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[768, 384, 3, 3], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_26
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_30
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_32
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_36
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_40
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_42
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_46
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_48
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[768, 768, 3, 3], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[2560, 768, 3, 3], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[2560, 1000], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[1000], dtype='float16'),
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