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

last_cinn_stage_exit_code = None
def LastCINNStageFailed():
    global last_cinn_stage_exit_code
    if last_cinn_stage_exit_code is not None:
        return last_cinn_stage_exit_code != 0
    last_stage = GetPrevCinnStage(GetCurrentCinnStage())
    if last_stage is None:
        return False
    env_vars = dict(
        PADDLE_DEBUG_CINN_STAGE_NAME=last_stage.name,
        PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
    )
    env_vars_str = " ".join(
        f"{env_var}={value}"
        for env_var, value in env_vars.items()
    )
    last_cinn_stage_exit_code = os.system(
        f"{env_vars_str} {sys.executable} {__file__} > /dev/null 2>&1"
    )
    return last_cinn_stage_exit_code != 0

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

last_stage_failed = (IsCinnStageEnableDiff() and LastCINNStageFailed())

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
    return [114][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, parameter_27, parameter_26, parameter_38, parameter_0, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_1, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, data_0):

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x3x-1x-1xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xi64) <- (64xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_0), None

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x64x1x1xf32, 0x64xi64) <- (64xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(parameter_3, full_int_array_0), None

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(add_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_1

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_1

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_1

        # pd_op.pool2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu_1, full_int_array_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_0, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_5, full_int_array_0), None

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu_2, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_6, reshape_7 = paddle.reshape(parameter_7, full_int_array_0), None

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_3 = conv2d_3 + reshape_6

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(add_3)

        # pd_op.pool2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu_3, assign_2, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_1, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_8, reshape_9 = paddle.reshape(parameter_9, full_int_array_0), None

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_4 = conv2d_4 + reshape_8

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu_4, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_10, reshape_11 = paddle.reshape(parameter_11, full_int_array_0), None

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_5 = conv2d_5 + reshape_10

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(add_5)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu_5, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_12, reshape_13 = paddle.reshape(parameter_13, full_int_array_0), None

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_6 = conv2d_6 + reshape_12

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(add_6)

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu_6, assign_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(pool2d_2, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_14, reshape_15 = paddle.reshape(parameter_15, full_int_array_0), None

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_7 = conv2d_7 + reshape_14

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(add_7)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu_7, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_16, reshape_17 = paddle.reshape(parameter_17, full_int_array_0), None

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_8 = conv2d_8 + reshape_16

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(add_8)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu_8, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_18, reshape_19 = paddle.reshape(parameter_19, full_int_array_0), None

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_9 = conv2d_9 + reshape_18

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(add_9)

        # pd_op.pool2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu_9, assign_0, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(pool2d_3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_20, reshape_21 = paddle.reshape(parameter_21, full_int_array_0), None

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_10 = conv2d_10 + reshape_20

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(add_10)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu_10, parameter_22, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_22, reshape_23 = paddle.reshape(parameter_23, full_int_array_0), None

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_11 = conv2d_11 + reshape_22

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(add_11)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu_11, parameter_24, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_24, reshape_25 = paddle.reshape(parameter_25, full_int_array_0), None

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_12 = conv2d_12 + reshape_24

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(add_12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [3, 3]

        # pd_op.pool2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu_12, full_int_array_2, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x512x-1x-1xf32, 1024x512x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(pool2d_4, parameter_26, [1, 1], [6, 6], 'EXPLICIT', [6, 6], 1, 'NCHW')

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xi64) <- (1024xf32, 4xi64)
        reshape_26, reshape_27 = paddle.reshape(parameter_27, full_int_array_0), None

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1x1024x1x1xf32)
        add_13 = conv2d_13 + reshape_26

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(add_13)

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x1024x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu_13, parameter_28, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xi64) <- (1024xf32, 4xi64)
        reshape_28, reshape_29 = paddle.reshape(parameter_29, full_int_array_0), None

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1x1024x1x1xf32)
        add_14 = conv2d_14 + reshape_28

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(add_14)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu_14, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_30, reshape_31 = paddle.reshape(parameter_31, full_int_array_0), None

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_15 = conv2d_15 + reshape_30

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(add_15)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu_15, parameter_32, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_32, reshape_33 = paddle.reshape(parameter_33, full_int_array_0), None

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_16 = conv2d_16 + reshape_32

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(add_16)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x512x-1x-1xf32, 128x512x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu_16, parameter_34, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_34, reshape_35 = paddle.reshape(parameter_35, full_int_array_0), None

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_17 = conv2d_17 + reshape_34

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(add_17)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu_17, parameter_36, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_36, reshape_37 = paddle.reshape(parameter_37, full_int_array_0), None

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_18 = conv2d_18 + reshape_36

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(add_18)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x256x-1x-1xf32, 128x256x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu_18, parameter_38, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_38, reshape_39 = paddle.reshape(parameter_39, full_int_array_0), None

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_19 = conv2d_19 + reshape_38

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(add_19)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu_19, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_40, reshape_41 = paddle.reshape(parameter_41, full_int_array_0), None

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_20 = conv2d_20 + reshape_40

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_20 = paddle._C_ops.relu(add_20)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x256x-1x-1xf32, 128x256x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu_20, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_42, reshape_43 = paddle.reshape(parameter_43, full_int_array_0), None

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_21 = conv2d_21 + reshape_42

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(add_21)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu_21, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_44, reshape_45 = paddle.reshape(parameter_45, full_int_array_0), None

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_22 = conv2d_22 + reshape_44

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(add_22)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 1e-10, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.p_norm: (-1x1x-1x-1xf32) <- (-1x512x-1x-1xf32)
        p_norm_0 = paddle._C_ops.p_norm(relu_9, 2, 1, 1e-10, True, False)

        # pd_op.maximum: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, 1xf32)
        maximum_0 = paddle.maximum(p_norm_0, full_0)

        # pd_op.divide: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x1x-1x-1xf32)
        divide_0 = relu_9 / maximum_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.unsqueeze: (1x512xf32, 0x512xf32) <- (512xf32, 1xi64)
        unsqueeze_0, unsqueeze_1 = paddle.unsqueeze(parameter_46, full_int_array_3), None

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.unsqueeze: (1x512x1xf32, 0x1x512xf32) <- (1x512xf32, 1xi64)
        unsqueeze_2, unsqueeze_3 = paddle.unsqueeze(unsqueeze_0, full_int_array_4), None

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.unsqueeze: (1x512x1x1xf32, 0x1x512x1xf32) <- (1x512x1xf32, 1xi64)
        unsqueeze_4, unsqueeze_5 = paddle.unsqueeze(unsqueeze_2, full_int_array_5), None

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (1x512x1x1xf32, -1x512x-1x-1xf32)
        multiply_0 = unsqueeze_4 * divide_0
        return conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, assign_2, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, assign_1, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, assign_0, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_1, full_int_array_4, unsqueeze_3, full_int_array_5, unsqueeze_4, unsqueeze_5, multiply_0, relu_14, relu_16, relu_18, relu_20, relu_22



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

    def test_train(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

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

    def forward(self, parameter_27, parameter_26, parameter_38, parameter_0, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_1, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, data_0):
        args = [parameter_27, parameter_26, parameter_38, parameter_0, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_1, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, data_0]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_conv2d_0,
            self.op_full_int_array_0,
            self.op_reshape_0,
            self.op_add_0,
            self.op_relu_0,
            self.op_conv2d_1,
            self.op_reshape_1,
            self.op_add_1,
            self.op_relu_1,
            self.op_full_int_array_1,
            self.op_assign_0,
            self.op_assign_1,
            self.op_assign_2,
            self.op_pool2d_0,
            self.op_conv2d_2,
            self.op_reshape_2,
            self.op_add_2,
            self.op_relu_2,
            self.op_conv2d_3,
            self.op_reshape_3,
            self.op_add_3,
            self.op_relu_3,
            self.op_pool2d_1,
            self.op_conv2d_4,
            self.op_reshape_4,
            self.op_add_4,
            self.op_relu_4,
            self.op_conv2d_5,
            self.op_reshape_5,
            self.op_add_5,
            self.op_relu_5,
            self.op_conv2d_6,
            self.op_reshape_6,
            self.op_add_6,
            self.op_relu_6,
            self.op_pool2d_2,
            self.op_conv2d_7,
            self.op_reshape_7,
            self.op_add_7,
            self.op_relu_7,
            self.op_conv2d_8,
            self.op_reshape_8,
            self.op_add_8,
            self.op_relu_8,
            self.op_conv2d_9,
            self.op_reshape_9,
            self.op_add_9,
            self.op_relu_9,
            self.op_pool2d_3,
            self.op_conv2d_10,
            self.op_reshape_10,
            self.op_add_10,
            self.op_relu_10,
            self.op_conv2d_11,
            self.op_reshape_11,
            self.op_add_11,
            self.op_relu_11,
            self.op_conv2d_12,
            self.op_reshape_12,
            self.op_add_12,
            self.op_relu_12,
            self.op_full_int_array_2,
            self.op_pool2d_4,
            self.op_conv2d_13,
            self.op_reshape_13,
            self.op_add_13,
            self.op_relu_13,
            self.op_conv2d_14,
            self.op_reshape_14,
            self.op_add_14,
            self.op_relu_14,
            self.op_conv2d_15,
            self.op_reshape_15,
            self.op_add_15,
            self.op_relu_15,
            self.op_conv2d_16,
            self.op_reshape_16,
            self.op_add_16,
            self.op_relu_16,
            self.op_conv2d_17,
            self.op_reshape_17,
            self.op_add_17,
            self.op_relu_17,
            self.op_conv2d_18,
            self.op_reshape_18,
            self.op_add_18,
            self.op_relu_18,
            self.op_conv2d_19,
            self.op_reshape_19,
            self.op_add_19,
            self.op_relu_19,
            self.op_conv2d_20,
            self.op_reshape_20,
            self.op_add_20,
            self.op_relu_20,
            self.op_conv2d_21,
            self.op_reshape_21,
            self.op_add_21,
            self.op_relu_21,
            self.op_conv2d_22,
            self.op_reshape_22,
            self.op_add_22,
            self.op_relu_22,
            self.op_full_0,
            self.op_p_norm_0,
            self.op_maximum_0,
            self.op_divide_0,
            self.op_full_int_array_3,
            self.op_unsqueeze_0,
            self.op_full_int_array_4,
            self.op_unsqueeze_1,
            self.op_full_int_array_5,
            self.op_unsqueeze_2,
            self.op_multiply_0,
        ]

    def op_conv2d_0(self, parameter_27, parameter_26, parameter_38, parameter_0, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_1, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, data_0):
    
        # EarlyReturn(0, 0)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x3x-1x-1xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_1, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0]

    def op_full_int_array_0(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_1, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0):
    
        # EarlyReturn(0, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_1, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0]

    def op_reshape_0(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_1, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0):
    
        # EarlyReturn(0, 2)

        # pd_op.reshape: (1x64x1x1xf32, 0x64xi64) <- (64xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1]

    def op_add_0(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1):
    
        # EarlyReturn(0, 3)

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0]

    def op_relu_0(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0):
    
        # EarlyReturn(0, 4)

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0]

    def op_conv2d_1(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_2, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0):
    
        # EarlyReturn(0, 5)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1]

    def op_reshape_1(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_3, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1):
    
        # EarlyReturn(0, 6)

        # pd_op.reshape: (1x64x1x1xf32, 0x64xi64) <- (64xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(parameter_3, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3]

    def op_add_1(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3):
    
        # EarlyReturn(0, 7)

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, add_1]

    def op_relu_1(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, add_1):
    
        # EarlyReturn(0, 8)

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(add_1)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1]

    def op_full_int_array_1(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1):
    
        # EarlyReturn(0, 9)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1]

    def op_assign_0(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1):
    
        # EarlyReturn(0, 10)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_1

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0]

    def op_assign_1(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0):
    
        # EarlyReturn(0, 11)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_1

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1]

    def op_assign_2(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1):
    
        # EarlyReturn(0, 12)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_1

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2]

    def op_pool2d_0(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2):
    
        # EarlyReturn(0, 13)

        # pd_op.pool2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu_1, full_int_array_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0]

    def op_conv2d_2(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_4, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0):
    
        # EarlyReturn(0, 14)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_0, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2]

    def op_reshape_2(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_5, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2):
    
        # EarlyReturn(0, 15)

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_5, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5]

    def op_add_2(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5):
    
        # EarlyReturn(0, 16)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, add_2]

    def op_relu_2(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, add_2):
    
        # EarlyReturn(0, 17)

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(add_2)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2]

    def op_conv2d_3(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_6, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2):
    
        # EarlyReturn(0, 18)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu_2, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3]

    def op_reshape_3(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_7, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3):
    
        # EarlyReturn(0, 19)

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_6, reshape_7 = paddle.reshape(parameter_7, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7]

    def op_add_3(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7):
    
        # EarlyReturn(0, 20)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_3 = conv2d_3 + reshape_6

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, add_3]

    def op_relu_3(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, add_3):
    
        # EarlyReturn(0, 21)

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(add_3)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3]

    def op_pool2d_1(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3):
    
        # EarlyReturn(0, 22)

        # pd_op.pool2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu_3, assign_2, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1]

    def op_conv2d_4(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_8, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1):
    
        # EarlyReturn(0, 23)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_1, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4]

    def op_reshape_4(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_9, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4):
    
        # EarlyReturn(0, 24)

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_8, reshape_9 = paddle.reshape(parameter_9, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9]

    def op_add_4(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9):
    
        # EarlyReturn(0, 25)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_4 = conv2d_4 + reshape_8

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, add_4]

    def op_relu_4(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, add_4):
    
        # EarlyReturn(0, 26)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(add_4)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4]

    def op_conv2d_5(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_10, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4):
    
        # EarlyReturn(0, 27)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu_4, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5]

    def op_reshape_5(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_11, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5):
    
        # EarlyReturn(0, 28)

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_10, reshape_11 = paddle.reshape(parameter_11, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11]

    def op_add_5(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11):
    
        # EarlyReturn(0, 29)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_5 = conv2d_5 + reshape_10

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, add_5]

    def op_relu_5(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, add_5):
    
        # EarlyReturn(0, 30)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(add_5)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5]

    def op_conv2d_6(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, parameter_12, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5):
    
        # EarlyReturn(0, 31)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu_5, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6]

    def op_reshape_6(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_13, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6):
    
        # EarlyReturn(0, 32)

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_12, reshape_13 = paddle.reshape(parameter_13, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13]

    def op_add_6(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13):
    
        # EarlyReturn(0, 33)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_6 = conv2d_6 + reshape_12

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, add_6]

    def op_relu_6(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, add_6):
    
        # EarlyReturn(0, 34)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(add_6)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6]

    def op_pool2d_2(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6):
    
        # EarlyReturn(0, 35)

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu_6, assign_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2]

    def op_conv2d_7(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_14, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2):
    
        # EarlyReturn(0, 36)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(pool2d_2, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7]

    def op_reshape_7(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_15, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7):
    
        # EarlyReturn(0, 37)

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_14, reshape_15 = paddle.reshape(parameter_15, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15]

    def op_add_7(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15):
    
        # EarlyReturn(0, 38)

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_7 = conv2d_7 + reshape_14

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, add_7]

    def op_relu_7(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, add_7):
    
        # EarlyReturn(0, 39)

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(add_7)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7]

    def op_conv2d_8(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_16, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7):
    
        # EarlyReturn(0, 40)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu_7, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8]

    def op_reshape_8(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_17, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8):
    
        # EarlyReturn(0, 41)

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_16, reshape_17 = paddle.reshape(parameter_17, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17]

    def op_add_8(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17):
    
        # EarlyReturn(0, 42)

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_8 = conv2d_8 + reshape_16

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, add_8]

    def op_relu_8(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, add_8):
    
        # EarlyReturn(0, 43)

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(add_8)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8]

    def op_conv2d_9(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_18, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8):
    
        # EarlyReturn(0, 44)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu_8, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9]

    def op_reshape_9(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_19, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9):
    
        # EarlyReturn(0, 45)

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_18, reshape_19 = paddle.reshape(parameter_19, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19]

    def op_add_9(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19):
    
        # EarlyReturn(0, 46)

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_9 = conv2d_9 + reshape_18

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, add_9]

    def op_relu_9(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, add_9):
    
        # EarlyReturn(0, 47)

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(add_9)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9]

    def op_pool2d_3(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9):
    
        # EarlyReturn(0, 48)

        # pd_op.pool2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu_9, assign_0, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3]

    def op_conv2d_10(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_20, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3):
    
        # EarlyReturn(0, 49)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(pool2d_3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10]

    def op_reshape_10(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_21, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10):
    
        # EarlyReturn(0, 50)

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_20, reshape_21 = paddle.reshape(parameter_21, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21]

    def op_add_10(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21):
    
        # EarlyReturn(0, 51)

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_10 = conv2d_10 + reshape_20

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, add_10]

    def op_relu_10(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, add_10):
    
        # EarlyReturn(0, 52)

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(add_10)

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10]

    def op_conv2d_11(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_22, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10):
    
        # EarlyReturn(0, 53)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu_10, parameter_22, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11]

    def op_reshape_11(self, parameter_27, parameter_26, parameter_38, parameter_23, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11):
    
        # EarlyReturn(0, 54)

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_22, reshape_23 = paddle.reshape(parameter_23, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23]

    def op_add_11(self, parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23):
    
        # EarlyReturn(0, 55)

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_11 = conv2d_11 + reshape_22

        return [parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, add_11]

    def op_relu_11(self, parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, add_11):
    
        # EarlyReturn(0, 56)

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(add_11)

        return [parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11]

    def op_conv2d_12(self, parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_24, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11):
    
        # EarlyReturn(0, 57)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu_11, parameter_24, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12]

    def op_reshape_12(self, parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, parameter_25, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12):
    
        # EarlyReturn(0, 58)

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_24, reshape_25 = paddle.reshape(parameter_25, full_int_array_0), None

        return [parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25]

    def op_add_12(self, parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25):
    
        # EarlyReturn(0, 59)

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_12 = conv2d_12 + reshape_24

        return [parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, add_12]

    def op_relu_12(self, parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, add_12):
    
        # EarlyReturn(0, 60)

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(add_12)

        return [parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12]

    def op_full_int_array_2(self, parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12):
    
        # EarlyReturn(0, 61)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [3, 3]

        return [parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2]

    def op_pool2d_4(self, parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2):
    
        # EarlyReturn(0, 62)

        # pd_op.pool2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu_12, full_int_array_2, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        return [parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4]

    def op_conv2d_13(self, parameter_27, parameter_26, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4):
    
        # EarlyReturn(0, 63)

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x512x-1x-1xf32, 1024x512x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(pool2d_4, parameter_26, [1, 1], [6, 6], 'EXPLICIT', [6, 6], 1, 'NCHW')

        return [parameter_27, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13]

    def op_reshape_13(self, parameter_27, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13):
    
        # EarlyReturn(0, 64)

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xi64) <- (1024xf32, 4xi64)
        reshape_26, reshape_27 = paddle.reshape(parameter_27, full_int_array_0), None

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27]

    def op_add_13(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27):
    
        # EarlyReturn(0, 65)

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1x1024x1x1xf32)
        add_13 = conv2d_13 + reshape_26

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, add_13]

    def op_relu_13(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, add_13):
    
        # EarlyReturn(0, 66)

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(add_13)

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13]

    def op_conv2d_14(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_28, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13):
    
        # EarlyReturn(0, 67)

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x1024x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu_13, parameter_28, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14]

    def op_reshape_14(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_29, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14):
    
        # EarlyReturn(0, 68)

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xi64) <- (1024xf32, 4xi64)
        reshape_28, reshape_29 = paddle.reshape(parameter_29, full_int_array_0), None

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29]

    def op_add_14(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29):
    
        # EarlyReturn(0, 69)

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1x1024x1x1xf32)
        add_14 = conv2d_14 + reshape_28

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, add_14]

    def op_relu_14(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, add_14):
    
        # EarlyReturn(0, 70)

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(add_14)

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14]

    def op_conv2d_15(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_30, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14):
    
        # EarlyReturn(0, 71)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu_14, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15]

    def op_reshape_15(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_32, parameter_31, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15):
    
        # EarlyReturn(0, 72)

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_30, reshape_31 = paddle.reshape(parameter_31, full_int_array_0), None

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_32, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31]

    def op_add_15(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_32, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31):
    
        # EarlyReturn(0, 73)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_15 = conv2d_15 + reshape_30

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_32, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, add_15]

    def op_relu_15(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_32, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, add_15):
    
        # EarlyReturn(0, 74)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(add_15)

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_32, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15]

    def op_conv2d_16(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_32, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15):
    
        # EarlyReturn(0, 75)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu_15, parameter_32, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16]

    def op_reshape_16(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, parameter_33, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16):
    
        # EarlyReturn(0, 76)

        # pd_op.reshape: (1x512x1x1xf32, 0x512xi64) <- (512xf32, 4xi64)
        reshape_32, reshape_33 = paddle.reshape(parameter_33, full_int_array_0), None

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33]

    def op_add_16(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33):
    
        # EarlyReturn(0, 77)

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_16 = conv2d_16 + reshape_32

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, add_16]

    def op_relu_16(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, add_16):
    
        # EarlyReturn(0, 78)

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(add_16)

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16]

    def op_conv2d_17(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_34, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16):
    
        # EarlyReturn(0, 79)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x512x-1x-1xf32, 128x512x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu_16, parameter_34, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_38, parameter_45, parameter_42, parameter_35, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17]

    def op_reshape_17(self, parameter_38, parameter_45, parameter_42, parameter_35, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17):
    
        # EarlyReturn(0, 80)

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_34, reshape_35 = paddle.reshape(parameter_35, full_int_array_0), None

        return [parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35]

    def op_add_17(self, parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35):
    
        # EarlyReturn(0, 81)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_17 = conv2d_17 + reshape_34

        return [parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, add_17]

    def op_relu_17(self, parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, add_17):
    
        # EarlyReturn(0, 82)

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(add_17)

        return [parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17]

    def op_conv2d_18(self, parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, parameter_36, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17):
    
        # EarlyReturn(0, 83)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu_17, parameter_36, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18]

    def op_reshape_18(self, parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_37, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18):
    
        # EarlyReturn(0, 84)

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_36, reshape_37 = paddle.reshape(parameter_37, full_int_array_0), None

        return [parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37]

    def op_add_18(self, parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37):
    
        # EarlyReturn(0, 85)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_18 = conv2d_18 + reshape_36

        return [parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, add_18]

    def op_relu_18(self, parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, add_18):
    
        # EarlyReturn(0, 86)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(add_18)

        return [parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18]

    def op_conv2d_19(self, parameter_38, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18):
    
        # EarlyReturn(0, 87)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x256x-1x-1xf32, 128x256x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu_18, parameter_38, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19]

    def op_reshape_19(self, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_39, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19):
    
        # EarlyReturn(0, 88)

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_38, reshape_39 = paddle.reshape(parameter_39, full_int_array_0), None

        return [parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39]

    def op_add_19(self, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39):
    
        # EarlyReturn(0, 89)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_19 = conv2d_19 + reshape_38

        return [parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, add_19]

    def op_relu_19(self, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, add_19):
    
        # EarlyReturn(0, 90)

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(add_19)

        return [parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19]

    def op_conv2d_20(self, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_40, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19):
    
        # EarlyReturn(0, 91)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu_19, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20]

    def op_reshape_20(self, parameter_45, parameter_42, parameter_46, parameter_41, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20):
    
        # EarlyReturn(0, 92)

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_40, reshape_41 = paddle.reshape(parameter_41, full_int_array_0), None

        return [parameter_45, parameter_42, parameter_46, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41]

    def op_add_20(self, parameter_45, parameter_42, parameter_46, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41):
    
        # EarlyReturn(0, 93)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_20 = conv2d_20 + reshape_40

        return [parameter_45, parameter_42, parameter_46, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, add_20]

    def op_relu_20(self, parameter_45, parameter_42, parameter_46, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, add_20):
    
        # EarlyReturn(0, 94)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_20 = paddle._C_ops.relu(add_20)

        return [parameter_45, parameter_42, parameter_46, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20]

    def op_conv2d_21(self, parameter_45, parameter_42, parameter_46, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20):
    
        # EarlyReturn(0, 95)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x256x-1x-1xf32, 128x256x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu_20, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_45, parameter_46, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21]

    def op_reshape_21(self, parameter_45, parameter_46, parameter_43, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21):
    
        # EarlyReturn(0, 96)

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_42, reshape_43 = paddle.reshape(parameter_43, full_int_array_0), None

        return [parameter_45, parameter_46, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43]

    def op_add_21(self, parameter_45, parameter_46, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43):
    
        # EarlyReturn(0, 97)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_21 = conv2d_21 + reshape_42

        return [parameter_45, parameter_46, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, add_21]

    def op_relu_21(self, parameter_45, parameter_46, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, add_21):
    
        # EarlyReturn(0, 98)

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(add_21)

        return [parameter_45, parameter_46, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21]

    def op_conv2d_22(self, parameter_45, parameter_46, parameter_44, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21):
    
        # EarlyReturn(0, 99)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu_21, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_45, parameter_46, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22]

    def op_reshape_22(self, parameter_45, parameter_46, conv2d_0, full_int_array_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22):
    
        # EarlyReturn(0, 100)

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_44, reshape_45 = paddle.reshape(parameter_45, full_int_array_0), None

        return [parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45]

    def op_add_22(self, parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45):
    
        # EarlyReturn(0, 101)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_22 = conv2d_22 + reshape_44

        return [parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, add_22]

    def op_relu_22(self, parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, add_22):
    
        # EarlyReturn(0, 102)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(add_22)

        return [parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22]

    def op_full_0(self, parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22):
    
        # EarlyReturn(0, 103)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 1e-10, paddle.float32, paddle.framework._current_expected_place())

        return [parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0]

    def op_p_norm_0(self, parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0):
    
        # EarlyReturn(0, 104)

        # pd_op.p_norm: (-1x1x-1x-1xf32) <- (-1x512x-1x-1xf32)
        p_norm_0 = paddle._C_ops.p_norm(relu_9, 2, 1, 1e-10, True, False)

        return [parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0]

    def op_maximum_0(self, parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0):
    
        # EarlyReturn(0, 105)

        # pd_op.maximum: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, 1xf32)
        maximum_0 = paddle.maximum(p_norm_0, full_0)

        return [parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0]

    def op_divide_0(self, parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0):
    
        # EarlyReturn(0, 106)

        # pd_op.divide: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x1x-1x-1xf32)
        divide_0 = relu_9 / maximum_0

        return [parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0]

    def op_full_int_array_3(self, parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0):
    
        # EarlyReturn(0, 107)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        return [parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3]

    def op_unsqueeze_0(self, parameter_46, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3):
    
        # EarlyReturn(0, 108)

        # pd_op.unsqueeze: (1x512xf32, 0x512xf32) <- (512xf32, 1xi64)
        unsqueeze_0, unsqueeze_1 = paddle.unsqueeze(parameter_46, full_int_array_3), None

        return [conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_0, unsqueeze_1]

    def op_full_int_array_4(self, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_0, unsqueeze_1):
    
        # EarlyReturn(0, 109)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        return [conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_0, unsqueeze_1, full_int_array_4]

    def op_unsqueeze_1(self, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_0, unsqueeze_1, full_int_array_4):
    
        # EarlyReturn(0, 110)

        # pd_op.unsqueeze: (1x512x1xf32, 0x1x512xf32) <- (1x512xf32, 1xi64)
        unsqueeze_2, unsqueeze_3 = paddle.unsqueeze(unsqueeze_0, full_int_array_4), None

        return [conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_1, full_int_array_4, unsqueeze_2, unsqueeze_3]

    def op_full_int_array_5(self, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_1, full_int_array_4, unsqueeze_2, unsqueeze_3):
    
        # EarlyReturn(0, 111)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        return [conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_1, full_int_array_4, unsqueeze_2, unsqueeze_3, full_int_array_5]

    def op_unsqueeze_2(self, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_1, full_int_array_4, unsqueeze_2, unsqueeze_3, full_int_array_5):
    
        # EarlyReturn(0, 112)

        # pd_op.unsqueeze: (1x512x1x1xf32, 0x1x512x1xf32) <- (1x512x1xf32, 1xi64)
        unsqueeze_4, unsqueeze_5 = paddle.unsqueeze(unsqueeze_2, full_int_array_5), None

        return [conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_1, full_int_array_4, unsqueeze_3, full_int_array_5, unsqueeze_4, unsqueeze_5]

    def op_multiply_0(self, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, assign_0, assign_1, assign_2, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, relu_14, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, relu_16, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, relu_18, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, relu_20, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, relu_22, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_1, full_int_array_4, unsqueeze_3, full_int_array_5, unsqueeze_4, unsqueeze_5):
    
        # EarlyReturn(0, 113)

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (1x512x1x1xf32, -1x512x-1x-1xf32)
        multiply_0 = unsqueeze_4 * divide_0

        return [conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, full_int_array_1, pool2d_0, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, assign_2, pool2d_1, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, assign_1, pool2d_2, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, relu_8, conv2d_9, reshape_18, reshape_19, relu_9, assign_0, pool2d_3, conv2d_10, reshape_20, reshape_21, relu_10, conv2d_11, reshape_22, reshape_23, relu_11, conv2d_12, reshape_24, reshape_25, relu_12, full_int_array_2, pool2d_4, conv2d_13, reshape_26, reshape_27, relu_13, conv2d_14, reshape_28, reshape_29, conv2d_15, reshape_30, reshape_31, relu_15, conv2d_16, reshape_32, reshape_33, conv2d_17, reshape_34, reshape_35, relu_17, conv2d_18, reshape_36, reshape_37, conv2d_19, reshape_38, reshape_39, relu_19, conv2d_20, reshape_40, reshape_41, conv2d_21, reshape_42, reshape_43, relu_21, conv2d_22, reshape_44, reshape_45, full_0, p_norm_0, maximum_0, divide_0, full_int_array_3, unsqueeze_1, full_int_array_4, unsqueeze_3, full_int_array_5, unsqueeze_4, unsqueeze_5, multiply_0, relu_14, relu_16, relu_18, relu_20, relu_22]

is_module_block_and_last_stage_passed = (
    True and not last_stage_failed
)
@unittest.skipIf(not is_module_block_and_last_stage_passed, "last stage failed")
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_27
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([1024, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([64, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([1024, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1, 3, 300, 300], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_27
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[1024, 512, 3, 3], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[64, 3, 3, 3], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[1024, 1024, 1, 1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = Block_builtin_module_0_0_0()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        paddle.seed(2024)
        out = net(*self.inputs)
        return out

if __name__ == '__main__':
    unittest.main()