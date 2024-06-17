import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'

import unittest
import numpy as np
import paddle

def NumCurrentUnittestOperations():
    return 177 # number-of-ops

def GetPaddleDebugNumAllowedOps():
    try:
        return int(os.getenv('PADDLE_DEBUG_NUM_ALLOWED_OPS'))
    except:
        return None

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


paddle_debug_num_allowed_ops = GetPaddleDebugNumAllowedOps()

def FastReturn(i):
    return (
        type(paddle_debug_num_allowed_ops) is int
        and i >= paddle_debug_num_allowed_ops
    )

class GroupOp(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, matmul_0, parameter_0):

        if FastReturn(0):
            return matmul_0, parameter_0

        #  type: (1x1280xf16) <- (1x1280xf16, 1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280], [1280])
        #  data: (None) <- (None, None)
        add_0 = matmul_0 + parameter_0

        if FastReturn(1):
            return add_0

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_0 = paddle.cast(add_0, dtype='float32')

        if FastReturn(2):
            return add_0, cast_0

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_0 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(3):
            return add_0, cast_0, full_0

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_0 = cast_0 * -1 + 0

        if FastReturn(4):
            return add_0, cast_0, full_0, scale_0

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_0 = paddle.exp(scale_0)

        if FastReturn(5):
            return add_0, cast_0, full_0, exp_0

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_1 = full_0 + exp_0

        if FastReturn(6):
            return add_0, cast_0, full_0, add_1

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_0 = full_0 / add_1

        if FastReturn(7):
            return add_0, cast_0, divide_0

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_0 = cast_0 * divide_0

        if FastReturn(8):
            return add_0, multiply_0

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_1 = paddle.cast(multiply_0, dtype='float16')

        if FastReturn(9):
            return add_0, cast_1

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_2 = paddle.cast(add_0, dtype='float32')

        if FastReturn(10):
            return add_0, cast_1, cast_2

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_1 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(11):
            return add_0, cast_1, cast_2, full_1

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_1 = cast_2 * -1 + 0

        if FastReturn(12):
            return add_0, cast_1, cast_2, full_1, scale_1

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_1 = paddle.exp(scale_1)

        if FastReturn(13):
            return add_0, cast_1, cast_2, full_1, exp_1

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_2 = full_1 + exp_1

        if FastReturn(14):
            return add_0, cast_1, cast_2, full_1, add_2

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_1 = full_1 / add_2

        if FastReturn(15):
            return add_0, cast_1, cast_2, divide_1

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_1 = cast_2 * divide_1

        if FastReturn(16):
            return add_0, cast_1, multiply_1

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_3 = paddle.cast(multiply_1, dtype='float16')

        if FastReturn(17):
            return add_0, cast_1, cast_3

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_4 = paddle.cast(add_0, dtype='float32')

        if FastReturn(18):
            return add_0, cast_1, cast_3, cast_4

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_2 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(19):
            return add_0, cast_1, cast_3, cast_4, full_2

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_2 = cast_4 * -1 + 0

        if FastReturn(20):
            return add_0, cast_1, cast_3, cast_4, full_2, scale_2

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_2 = paddle.exp(scale_2)

        if FastReturn(21):
            return add_0, cast_1, cast_3, cast_4, full_2, exp_2

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_3 = full_2 + exp_2

        if FastReturn(22):
            return add_0, cast_1, cast_3, cast_4, full_2, add_3

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_2 = full_2 / add_3

        if FastReturn(23):
            return add_0, cast_1, cast_3, cast_4, divide_2

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_2 = cast_4 * divide_2

        if FastReturn(24):
            return add_0, cast_1, cast_3, multiply_2

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_5 = paddle.cast(multiply_2, dtype='float16')

        if FastReturn(25):
            return add_0, cast_1, cast_3, cast_5

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_6 = paddle.cast(add_0, dtype='float32')

        if FastReturn(26):
            return add_0, cast_1, cast_3, cast_5, cast_6

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_3 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(27):
            return add_0, cast_1, cast_3, cast_5, cast_6, full_3

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_3 = cast_6 * -1 + 0

        if FastReturn(28):
            return add_0, cast_1, cast_3, cast_5, cast_6, full_3, scale_3

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_3 = paddle.exp(scale_3)

        if FastReturn(29):
            return add_0, cast_1, cast_3, cast_5, cast_6, full_3, exp_3

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_4 = full_3 + exp_3

        if FastReturn(30):
            return add_0, cast_1, cast_3, cast_5, cast_6, full_3, add_4

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_3 = full_3 / add_4

        if FastReturn(31):
            return add_0, cast_1, cast_3, cast_5, cast_6, divide_3

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_3 = cast_6 * divide_3

        if FastReturn(32):
            return add_0, cast_1, cast_3, cast_5, multiply_3

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_7 = paddle.cast(multiply_3, dtype='float16')

        if FastReturn(33):
            return add_0, cast_1, cast_3, cast_5, cast_7

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_8 = paddle.cast(add_0, dtype='float32')

        if FastReturn(34):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_8

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_4 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(35):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_8, full_4

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_4 = cast_8 * -1 + 0

        if FastReturn(36):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_8, full_4, scale_4

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_4 = paddle.exp(scale_4)

        if FastReturn(37):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_8, full_4, exp_4

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_5 = full_4 + exp_4

        if FastReturn(38):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_8, full_4, add_5

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_4 = full_4 / add_5

        if FastReturn(39):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_8, divide_4

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_4 = cast_8 * divide_4

        if FastReturn(40):
            return add_0, cast_1, cast_3, cast_5, cast_7, multiply_4

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_9 = paddle.cast(multiply_4, dtype='float16')

        if FastReturn(41):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_10 = paddle.cast(add_0, dtype='float32')

        if FastReturn(42):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_10

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_5 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(43):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_10, full_5

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_5 = cast_10 * -1 + 0

        if FastReturn(44):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_10, full_5, scale_5

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_5 = paddle.exp(scale_5)

        if FastReturn(45):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_10, full_5, exp_5

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_6 = full_5 + exp_5

        if FastReturn(46):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_10, full_5, add_6

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_5 = full_5 / add_6

        if FastReturn(47):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_10, divide_5

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_5 = cast_10 * divide_5

        if FastReturn(48):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, multiply_5

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_11 = paddle.cast(multiply_5, dtype='float16')

        if FastReturn(49):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_12 = paddle.cast(add_0, dtype='float32')

        if FastReturn(50):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_12

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_6 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(51):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_12, full_6

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_6 = cast_12 * -1 + 0

        if FastReturn(52):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_12, full_6, scale_6

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_6 = paddle.exp(scale_6)

        if FastReturn(53):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_12, full_6, exp_6

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_7 = full_6 + exp_6

        if FastReturn(54):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_12, full_6, add_7

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_6 = full_6 / add_7

        if FastReturn(55):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_12, divide_6

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_6 = cast_12 * divide_6

        if FastReturn(56):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, multiply_6

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_13 = paddle.cast(multiply_6, dtype='float16')

        if FastReturn(57):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_14 = paddle.cast(add_0, dtype='float32')

        if FastReturn(58):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_14

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_7 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(59):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_14, full_7

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_7 = cast_14 * -1 + 0

        if FastReturn(60):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_14, full_7, scale_7

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_7 = paddle.exp(scale_7)

        if FastReturn(61):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_14, full_7, exp_7

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_8 = full_7 + exp_7

        if FastReturn(62):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_14, full_7, add_8

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_7 = full_7 / add_8

        if FastReturn(63):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_14, divide_7

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_7 = cast_14 * divide_7

        if FastReturn(64):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, multiply_7

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_15 = paddle.cast(multiply_7, dtype='float16')

        if FastReturn(65):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_16 = paddle.cast(add_0, dtype='float32')

        if FastReturn(66):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_16

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_8 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(67):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_16, full_8

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_8 = cast_16 * -1 + 0

        if FastReturn(68):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_16, full_8, scale_8

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_8 = paddle.exp(scale_8)

        if FastReturn(69):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_16, full_8, exp_8

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_9 = full_8 + exp_8

        if FastReturn(70):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_16, full_8, add_9

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_8 = full_8 / add_9

        if FastReturn(71):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_16, divide_8

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_8 = cast_16 * divide_8

        if FastReturn(72):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, multiply_8

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_17 = paddle.cast(multiply_8, dtype='float16')

        if FastReturn(73):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_18 = paddle.cast(add_0, dtype='float32')

        if FastReturn(74):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_18

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_9 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(75):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_18, full_9

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_9 = cast_18 * -1 + 0

        if FastReturn(76):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_18, full_9, scale_9

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_9 = paddle.exp(scale_9)

        if FastReturn(77):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_18, full_9, exp_9

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_10 = full_9 + exp_9

        if FastReturn(78):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_18, full_9, add_10

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_9 = full_9 / add_10

        if FastReturn(79):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_18, divide_9

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_9 = cast_18 * divide_9

        if FastReturn(80):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, multiply_9

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_19 = paddle.cast(multiply_9, dtype='float16')

        if FastReturn(81):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_20 = paddle.cast(add_0, dtype='float32')

        if FastReturn(82):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_20

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_10 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(83):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_20, full_10

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_10 = cast_20 * -1 + 0

        if FastReturn(84):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_20, full_10, scale_10

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_10 = paddle.exp(scale_10)

        if FastReturn(85):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_20, full_10, exp_10

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_11 = full_10 + exp_10

        if FastReturn(86):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_20, full_10, add_11

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_10 = full_10 / add_11

        if FastReturn(87):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_20, divide_10

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_10 = cast_20 * divide_10

        if FastReturn(88):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, multiply_10

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_21 = paddle.cast(multiply_10, dtype='float16')

        if FastReturn(89):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_22 = paddle.cast(add_0, dtype='float32')

        if FastReturn(90):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_22

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_11 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(91):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_22, full_11

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_11 = cast_22 * -1 + 0

        if FastReturn(92):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_22, full_11, scale_11

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_11 = paddle.exp(scale_11)

        if FastReturn(93):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_22, full_11, exp_11

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_12 = full_11 + exp_11

        if FastReturn(94):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_22, full_11, add_12

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_11 = full_11 / add_12

        if FastReturn(95):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_22, divide_11

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_11 = cast_22 * divide_11

        if FastReturn(96):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, multiply_11

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_23 = paddle.cast(multiply_11, dtype='float16')

        if FastReturn(97):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_24 = paddle.cast(add_0, dtype='float32')

        if FastReturn(98):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_24

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_12 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(99):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_24, full_12

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_12 = cast_24 * -1 + 0

        if FastReturn(100):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_24, full_12, scale_12

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_12 = paddle.exp(scale_12)

        if FastReturn(101):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_24, full_12, exp_12

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_13 = full_12 + exp_12

        if FastReturn(102):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_24, full_12, add_13

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_12 = full_12 / add_13

        if FastReturn(103):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_24, divide_12

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_12 = cast_24 * divide_12

        if FastReturn(104):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, multiply_12

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_25 = paddle.cast(multiply_12, dtype='float16')

        if FastReturn(105):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_26 = paddle.cast(add_0, dtype='float32')

        if FastReturn(106):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_26

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_13 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(107):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_26, full_13

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_13 = cast_26 * -1 + 0

        if FastReturn(108):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_26, full_13, scale_13

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_13 = paddle.exp(scale_13)

        if FastReturn(109):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_26, full_13, exp_13

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_14 = full_13 + exp_13

        if FastReturn(110):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_26, full_13, add_14

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_13 = full_13 / add_14

        if FastReturn(111):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_26, divide_13

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_13 = cast_26 * divide_13

        if FastReturn(112):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, multiply_13

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_27 = paddle.cast(multiply_13, dtype='float16')

        if FastReturn(113):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_28 = paddle.cast(add_0, dtype='float32')

        if FastReturn(114):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_28

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_14 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(115):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_28, full_14

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_14 = cast_28 * -1 + 0

        if FastReturn(116):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_28, full_14, scale_14

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_14 = paddle.exp(scale_14)

        if FastReturn(117):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_28, full_14, exp_14

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_15 = full_14 + exp_14

        if FastReturn(118):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_28, full_14, add_15

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_14 = full_14 / add_15

        if FastReturn(119):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_28, divide_14

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_14 = cast_28 * divide_14

        if FastReturn(120):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, multiply_14

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_29 = paddle.cast(multiply_14, dtype='float16')

        if FastReturn(121):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_30 = paddle.cast(add_0, dtype='float32')

        if FastReturn(122):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_30

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_15 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(123):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_30, full_15

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_15 = cast_30 * -1 + 0

        if FastReturn(124):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_30, full_15, scale_15

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_15 = paddle.exp(scale_15)

        if FastReturn(125):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_30, full_15, exp_15

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_16 = full_15 + exp_15

        if FastReturn(126):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_30, full_15, add_16

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_15 = full_15 / add_16

        if FastReturn(127):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_30, divide_15

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_15 = cast_30 * divide_15

        if FastReturn(128):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, multiply_15

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_31 = paddle.cast(multiply_15, dtype='float16')

        if FastReturn(129):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_32 = paddle.cast(add_0, dtype='float32')

        if FastReturn(130):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_32

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_16 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(131):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_32, full_16

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_16 = cast_32 * -1 + 0

        if FastReturn(132):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_32, full_16, scale_16

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_16 = paddle.exp(scale_16)

        if FastReturn(133):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_32, full_16, exp_16

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_17 = full_16 + exp_16

        if FastReturn(134):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_32, full_16, add_17

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_16 = full_16 / add_17

        if FastReturn(135):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_32, divide_16

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_16 = cast_32 * divide_16

        if FastReturn(136):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, multiply_16

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_33 = paddle.cast(multiply_16, dtype='float16')

        if FastReturn(137):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_34 = paddle.cast(add_0, dtype='float32')

        if FastReturn(138):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_34

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_17 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(139):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_34, full_17

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_17 = cast_34 * -1 + 0

        if FastReturn(140):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_34, full_17, scale_17

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_17 = paddle.exp(scale_17)

        if FastReturn(141):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_34, full_17, exp_17

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_18 = full_17 + exp_17

        if FastReturn(142):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_34, full_17, add_18

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_17 = full_17 / add_18

        if FastReturn(143):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_34, divide_17

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_17 = cast_34 * divide_17

        if FastReturn(144):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, multiply_17

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_35 = paddle.cast(multiply_17, dtype='float16')

        if FastReturn(145):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_36 = paddle.cast(add_0, dtype='float32')

        if FastReturn(146):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_36

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_18 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(147):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_36, full_18

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_18 = cast_36 * -1 + 0

        if FastReturn(148):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_36, full_18, scale_18

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_18 = paddle.exp(scale_18)

        if FastReturn(149):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_36, full_18, exp_18

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_19 = full_18 + exp_18

        if FastReturn(150):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_36, full_18, add_19

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_18 = full_18 / add_19

        if FastReturn(151):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_36, divide_18

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_18 = cast_36 * divide_18

        if FastReturn(152):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, multiply_18

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_37 = paddle.cast(multiply_18, dtype='float16')

        if FastReturn(153):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_38 = paddle.cast(add_0, dtype='float32')

        if FastReturn(154):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_38

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_19 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(155):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_38, full_19

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_19 = cast_38 * -1 + 0

        if FastReturn(156):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_38, full_19, scale_19

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_19 = paddle.exp(scale_19)

        if FastReturn(157):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_38, full_19, exp_19

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_20 = full_19 + exp_19

        if FastReturn(158):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_38, full_19, add_20

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_19 = full_19 / add_20

        if FastReturn(159):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_38, divide_19

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_19 = cast_38 * divide_19

        if FastReturn(160):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, multiply_19

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_39 = paddle.cast(multiply_19, dtype='float16')

        if FastReturn(161):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_40 = paddle.cast(add_0, dtype='float32')

        if FastReturn(162):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_40

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_20 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(163):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_40, full_20

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_20 = cast_40 * -1 + 0

        if FastReturn(164):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_40, full_20, scale_20

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_20 = paddle.exp(scale_20)

        if FastReturn(165):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_40, full_20, exp_20

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_21 = full_20 + exp_20

        if FastReturn(166):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_40, full_20, add_21

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_20 = full_20 / add_21

        if FastReturn(167):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_40, divide_20

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_20 = cast_40 * divide_20

        if FastReturn(168):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, multiply_20

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_41 = paddle.cast(multiply_20, dtype='float16')

        if FastReturn(169):
            return add_0, cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_41

        #  type: (1x1280xf32) <- (1x1280xf16)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_42 = paddle.cast(add_0, dtype='float32')

        if FastReturn(170):
            return cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_41, cast_42

        #  type: (1xf32) <- ()
        # shape: ([1]) <- ()
        #  data: ([1]) <- ()
        full_21 = paddle.full(shape=[1], dtype='float32', fill_value=1)

        if FastReturn(171):
            return cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_41, cast_42, full_21

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        scale_21 = cast_42 * -1 + 0

        if FastReturn(172):
            return cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_41, cast_42, full_21, scale_21

        #  type: (1x1280xf32) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        exp_21 = paddle.exp(scale_21)

        if FastReturn(173):
            return cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_41, cast_42, full_21, exp_21

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        add_22 = full_21 + exp_21

        if FastReturn(174):
            return cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_41, cast_42, full_21, add_22

        #  type: (1x1280xf32) <- (1xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1], [1, 1280])
        #  data: (None) <- ([1], None)
        divide_21 = full_21 / add_22

        if FastReturn(175):
            return cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_41, cast_42, divide_21

        #  type: (1x1280xf32) <- (1x1280xf32, 1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280], [1, 1280])
        #  data: (None) <- (None, None)
        multiply_21 = cast_42 * divide_21

        if FastReturn(176):
            return cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_41, multiply_21

        #  type: (1x1280xf16) <- (1x1280xf32)
        # shape: ([1, 1280]) <- ([1, 1280])
        #  data: (None) <- (None)
        cast_43 = paddle.cast(multiply_21, dtype='float16')

        #  type: () <- (1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16, 1x1280xf16)
        # shape: () <- ([1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280], [1, 1280])
        #  data: () <- (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        return cast_1, cast_3, cast_5, cast_7, cast_9, cast_11, cast_13, cast_15, cast_17, cast_19, cast_21, cast_23, cast_25, cast_27, cast_29, cast_31, cast_33, cast_35, cast_37, cast_39, cast_41, cast_43


class TestGroupOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1280], dtype='float16', min=-0.5, max=0.5),
            paddle.uniform([1280], dtype='float16', min=-0.5, max=0.5),
        ]
        for input in self.inputs:
          input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1280], dtype='float16'),
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = GroupOp()
        net.eval()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

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
            np.testing.assert_allclose(x.numpy(), y.numpy(), atol=1e-6)
        else:
            assert x == y


if __name__ == '__main__':
    unittest.main()