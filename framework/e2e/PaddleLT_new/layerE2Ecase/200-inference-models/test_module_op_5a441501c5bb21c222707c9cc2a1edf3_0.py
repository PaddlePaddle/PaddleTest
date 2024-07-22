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
    return [1289][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2299_0_0(self, constant_22, constant_21, parameter_309, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, parameter_33, constant_11, constant_10, constant_9, parameter_32, parameter_31, constant_8, parameter_30, parameter_29, constant_7, constant_6, parameter_28, constant_5, parameter_27, parameter_26, constant_12, constant_4, parameter_24, parameter_25, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_21, parameter_22, parameter_23, parameter_34, parameter_35, parameter_36, parameter_40, parameter_37, parameter_39, parameter_38, parameter_41, parameter_45, parameter_42, parameter_44, parameter_43, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_55, parameter_52, parameter_54, parameter_53, parameter_56, parameter_60, parameter_57, parameter_59, parameter_58, parameter_61, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_70, parameter_67, parameter_69, parameter_68, parameter_71, parameter_75, parameter_72, parameter_74, parameter_73, parameter_76, parameter_80, parameter_77, parameter_79, parameter_78, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_135, parameter_132, parameter_134, parameter_133, parameter_136, parameter_140, parameter_137, parameter_139, parameter_138, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_160, parameter_157, parameter_159, parameter_158, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_175, parameter_172, parameter_174, parameter_173, parameter_176, parameter_180, parameter_177, parameter_179, parameter_178, parameter_181, parameter_185, parameter_182, parameter_184, parameter_183, parameter_186, parameter_190, parameter_187, parameter_189, parameter_188, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_230, parameter_227, parameter_229, parameter_228, parameter_231, parameter_235, parameter_232, parameter_234, parameter_233, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_260, parameter_257, parameter_259, parameter_258, parameter_261, parameter_265, parameter_262, parameter_264, parameter_263, parameter_266, parameter_270, parameter_267, parameter_269, parameter_268, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_280, parameter_277, parameter_279, parameter_278, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_305, parameter_302, parameter_304, parameter_303, parameter_306, parameter_307, parameter_308, parameter_310, parameter_311, parameter_312, parameter_313, parameter_314, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, feed_0):

        # pd_op.conv2d: (-1x64x32x100xf32) <- (-1x1x32x100xf32, 64x1x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x32x100xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x32x100xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x32x100xf32) <- (-1x64x32x100xf32)
        relu__0 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.pool2d: (-1x64x16x50xf32) <- (-1x64x32x100xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__0, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x16x50xf32) <- (-1x64x16x50xf32, 128x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(pool2d_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x16x50xf32) <- (-1x128x16x50xf32)
        relu__1 = paddle._C_ops.relu(batch_norm__6)

        # pd_op.pool2d: (-1x128x8x25xf32) <- (-1x128x16x50xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__1, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x128x8x25xf32, 256x128x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__2 = paddle._C_ops.relu(batch_norm__12)

        # pd_op.pool2d: (-1x256x4x12xf32) <- (-1x256x8x25xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu__2, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x4x12xf32) <- (-1x256x4x12xf32, 512x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_2, parameter_15, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x12xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x12xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x12xf32) <- (-1x512x4x12xf32)
        relu__3 = paddle._C_ops.relu(batch_norm__18)

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x4x12xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu__3, constant_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.squeeze_: (-1x512x1xf32, None) <- (-1x512x1x1xf32, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze(pool2d_3, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x512x1xf32, 1xi64)
        squeeze__2, squeeze__3 = (lambda x, f: f(x))(paddle._C_ops.squeeze(squeeze__0, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256xf32) <- (-1x512xf32, 512x256xf32)
        matmul_0 = paddle.matmul(squeeze__2, parameter_20, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__0 = paddle._C_ops.add(matmul_0, parameter_21)

        # pd_op.relu_: (-1x256xf32) <- (-1x256xf32)
        relu__4 = paddle._C_ops.relu(add__0)

        # pd_op.matmul: (-1x40xf32) <- (-1x256xf32, 256x40xf32)
        matmul_1 = paddle.matmul(relu__4, parameter_22, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x40xf32) <- (-1x40xf32, 40xf32)
        add__1 = paddle._C_ops.add(matmul_1, parameter_23)

        # pd_op.reshape_: (-1x20x2xf32, 0x-1x40xf32) <- (-1x40xf32, 3xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__1, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full([1], float('-1'), paddle.float64, paddle.core.CPUPlace())

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.float64, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.linspace: (10xf64) <- (1xf64, 1xf64, 1xi32)
        linspace_0 = paddle._C_ops.linspace(full_0, full_1, full_2, paddle.float64, paddle.framework._current_expected_place())

        # builtin.combine: ([10xf64, 10xf64]) <- (10xf64, 10xf64)
        combine_0 = [linspace_0, parameter_24]

        # pd_op.stack: (10x2xf64) <- ([10xf64, 10xf64])
        stack_0 = paddle._C_ops.stack(combine_0, 1)

        # builtin.combine: ([10xf64, 10xf64]) <- (10xf64, 10xf64)
        combine_1 = [linspace_0, parameter_25]

        # pd_op.stack: (10x2xf64) <- ([10xf64, 10xf64])
        stack_1 = paddle._C_ops.stack(combine_1, 1)

        # builtin.combine: ([10x2xf64, 10x2xf64]) <- (10x2xf64, 10x2xf64)
        combine_2 = [stack_0, stack_1]

        # pd_op.concat: (20x2xf64) <- ([10x2xf64, 10x2xf64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, constant_4)

        # builtin.combine: ([100xf64, 32xf64]) <- (100xf64, 32xf64)
        combine_3 = [parameter_26, parameter_27]

        # pd_op.meshgrid: ([100x32xf64, 100x32xf64]) <- ([100xf64, 32xf64])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_3)

        # builtin.slice: (100x32xf64) <- ([100x32xf64, 100x32xf64])
        slice_0 = meshgrid_0[0]

        # builtin.slice: (100x32xf64) <- ([100x32xf64, 100x32xf64])
        slice_1 = meshgrid_0[1]

        # builtin.combine: ([100x32xf64, 100x32xf64]) <- (100x32xf64, 100x32xf64)
        combine_4 = [slice_0, slice_1]

        # pd_op.stack: (100x32x2xf64) <- ([100x32xf64, 100x32xf64])
        stack_2 = paddle._C_ops.stack(combine_4, 2)

        # pd_op.transpose: (32x100x2xf64) <- (100x32x2xf64)
        transpose_0 = paddle._C_ops.transpose(stack_2, [1, 0, 2])

        # pd_op.reshape_: (3200x2xf64, 0x32x100x2xf64) <- (32x100x2xf64, 2xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_0, constant_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.reshape: (1x20x2xf64, 0x20x2xf64) <- (20x2xf64, 3xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(concat_0, constant_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.reshape: (20x1x2xf64, 0x20x2xf64) <- (20x2xf64, 3xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(concat_0, constant_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.subtract: (20x20x2xf64) <- (1x20x2xf64, 20x1x2xf64)
        subtract_0 = reshape_0 - reshape_2

        # pd_op.p_norm: (20x20xf64) <- (20x20x2xf64)
        p_norm_0 = paddle._C_ops.p_norm(subtract_0, float('2'), 2, float('1e-12'), False, False)

        # pd_op.add_: (20x20xf64) <- (20x20xf64, 20x20xf64)
        add__2 = paddle._C_ops.add(p_norm_0, parameter_28)

        # pd_op.elementwise_pow: (20x20xf64) <- (20x20xf64, xf64)
        elementwise_pow_0 = paddle.pow(add__2, parameter_29)

        # pd_op.log_: (20x20xf64) <- (20x20xf64)
        log__0 = paddle._C_ops.log(add__2)

        # pd_op.multiply_: (20x20xf64) <- (20x20xf64, 20x20xf64)
        multiply__0 = paddle._C_ops.multiply(elementwise_pow_0, log__0)

        # builtin.combine: ([20x1xf64, 20x2xf64, 20x20xf64]) <- (20x1xf64, 20x2xf64, 20x20xf64)
        combine_5 = [parameter_30, concat_0, multiply__0]

        # pd_op.concat: (20x23xf64) <- ([20x1xf64, 20x2xf64, 20x20xf64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_5, constant_8)

        # pd_op.transpose: (2x20xf64) <- (20x2xf64)
        transpose_1 = paddle._C_ops.transpose(concat_0, [1, 0])

        # builtin.combine: ([2x3xf64, 2x20xf64]) <- (2x3xf64, 2x20xf64)
        combine_6 = [parameter_31, transpose_1]

        # pd_op.concat: (2x23xf64) <- ([2x3xf64, 2x20xf64], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_6, constant_8)

        # builtin.combine: ([20x23xf64, 2x23xf64, 1x23xf64]) <- (20x23xf64, 2x23xf64, 1x23xf64)
        combine_7 = [concat_1, concat_2, parameter_32]

        # pd_op.concat: (23x23xf64) <- ([20x23xf64, 2x23xf64, 1x23xf64], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_7, constant_4)

        # pd_op.inverse: (23x23xf64) <- (23x23xf64)
        inverse_0 = paddle._C_ops.inverse(concat_3)

        # pd_op.cast: (23x23xf32) <- (23x23xf64)
        cast_0 = paddle._C_ops.cast(inverse_0, paddle.float32)

        # pd_op.unsqueeze: (3200x1x2xf64, None) <- (3200x2xf64, 1xi64)
        unsqueeze_0, unsqueeze_1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(reshape__2, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.tile: (3200x20x2xf64) <- (3200x1x2xf64, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, constant_10)

        # pd_op.unsqueeze_: (1x20x2xf64, None) <- (20x2xf64, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(concat_0, constant_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.subtract_: (3200x20x2xf64) <- (3200x20x2xf64, 1x20x2xf64)
        subtract__0 = paddle._C_ops.subtract(tile_0, unsqueeze__0)

        # pd_op.p_norm: (3200x20xf64) <- (3200x20x2xf64)
        p_norm_1 = paddle._C_ops.p_norm(subtract__0, float('2'), 2, float('1e-12'), False, False)

        # pd_op.square: (3200x20xf64) <- (3200x20xf64)
        square_0 = paddle._C_ops.square(p_norm_1)

        # pd_op.scale_: (3200x20xf64) <- (3200x20xf64, 1xf32)
        scale__0 = paddle._C_ops.scale(p_norm_1, constant_12, float('1e-06'), True)

        # pd_op.log_: (3200x20xf64) <- (3200x20xf64)
        log__1 = paddle._C_ops.log(scale__0)

        # pd_op.multiply_: (3200x20xf64) <- (3200x20xf64, 3200x20xf64)
        multiply__1 = paddle._C_ops.multiply(square_0, log__1)

        # builtin.combine: ([3200x1xf64, 3200x2xf64, 3200x20xf64]) <- (3200x1xf64, 3200x2xf64, 3200x20xf64)
        combine_8 = [parameter_33, reshape__2, multiply__1]

        # pd_op.concat: (3200x23xf64) <- ([3200x1xf64, 3200x2xf64, 3200x20xf64], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_8, constant_8)

        # pd_op.cast: (3200x23xf32) <- (3200x23xf64)
        cast_1 = paddle._C_ops.cast(concat_4, paddle.float32)

        # pd_op.shape: (3xi32) <- (-1x20x2xf32)
        shape_0 = paddle._C_ops.shape(reshape__0)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_0, [0], constant_11, constant_9, [1], [0])

        # builtin.combine: ([xi32, 1xi32]) <- (xi32, 1xi32)
        combine_9 = [slice_2, constant_13]

        # pd_op.reshape: (-1x40xf32, 0x-1x20x2xf32) <- (-1x20x2xf32, [xi32, 1xi32])
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__0, combine_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x6xf32) <- (-1x40xf32, 40x6xf32)
        matmul_2 = paddle.matmul(reshape_4, parameter_34, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x6xf32) <- (-1x6xf32, 6xf32)
        add__3 = paddle._C_ops.add(matmul_2, parameter_35)

        # pd_op.reshape_: (-1x3x2xf32, 0x-1x6xf32) <- (-1x6xf32, 3xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__3, constant_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x20x2xf32, -1x3x2xf32]) <- (-1x20x2xf32, -1x3x2xf32)
        combine_10 = [reshape__0, reshape__4]

        # pd_op.concat: (-1x23x2xf32) <- ([-1x20x2xf32, -1x3x2xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_10, constant_8)

        # pd_op.matmul: (-1x23x2xf32) <- (23x23xf32, -1x23x2xf32)
        matmul_3 = paddle.matmul(cast_0, concat_5, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x3200x2xf32) <- (3200x23xf32, -1x23x2xf32)
        matmul_4 = paddle.matmul(cast_1, matmul_3, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (-1x32x100x2xf32, 0x-1x3200x2xf32) <- (-1x3200x2xf32, 4xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_4, constant_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.grid_sample: (-1x1x32x100xf32) <- (-1x1x32x100xf32, -1x32x100x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(feed_0, reshape__6, 'bilinear', 'zeros', True)

        # pd_op.conv2d: (-1x32x32x100xf32) <- (-1x1x32x100xf32, 32x1x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(grid_sample_0, parameter_36, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x32x100xf32, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x32x100xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_37, parameter_38, parameter_39, parameter_40, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x32x100xf32) <- (-1x32x32x100xf32)
        relu__5 = paddle._C_ops.relu(batch_norm__24)

        # pd_op.conv2d: (-1x64x32x100xf32) <- (-1x32x32x100xf32, 64x32x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu__5, parameter_41, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x32x100xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x32x100xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_42, parameter_43, parameter_44, parameter_45, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x32x100xf32) <- (-1x64x32x100xf32)
        relu__6 = paddle._C_ops.relu(batch_norm__30)

        # pd_op.pool2d: (-1x64x16x50xf32) <- (-1x64x32x100xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu__6, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x16x50xf32) <- (-1x64x16x50xf32, 128x64x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(pool2d_4, parameter_46, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_47, parameter_48, parameter_49, parameter_50, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x16x50xf32) <- (-1x128x16x50xf32)
        relu__7 = paddle._C_ops.relu(batch_norm__36)

        # pd_op.conv2d: (-1x128x16x50xf32) <- (-1x128x16x50xf32, 128x128x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu__7, parameter_51, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_52, parameter_53, parameter_54, parameter_55, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x16x50xf32) <- (-1x64x16x50xf32, 128x64x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(pool2d_4, parameter_56, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_57, parameter_58, parameter_59, parameter_60, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x16x50xf32) <- (-1x128x16x50xf32, -1x128x16x50xf32)
        add__4 = paddle._C_ops.add(batch_norm__42, batch_norm__48)

        # pd_op.relu_: (-1x128x16x50xf32) <- (-1x128x16x50xf32)
        relu__8 = paddle._C_ops.relu(add__4)

        # pd_op.conv2d: (-1x128x16x50xf32) <- (-1x128x16x50xf32, 128x128x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__8, parameter_61, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x16x50xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_62, parameter_63, parameter_64, parameter_65, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x16x50xf32) <- (-1x128x16x50xf32)
        relu__9 = paddle._C_ops.relu(batch_norm__54)

        # pd_op.pool2d: (-1x128x8x25xf32) <- (-1x128x16x50xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(relu__9, constant_0, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x128x8x25xf32, 256x128x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(pool2d_5, parameter_66, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_67, parameter_68, parameter_69, parameter_70, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__10 = paddle._C_ops.relu(batch_norm__60)

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x256x8x25xf32, 256x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu__10, parameter_71, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_72, parameter_73, parameter_74, parameter_75, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x128x8x25xf32, 256x128x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(pool2d_5, parameter_76, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_77, parameter_78, parameter_79, parameter_80, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x25xf32) <- (-1x256x8x25xf32, -1x256x8x25xf32)
        add__5 = paddle._C_ops.add(batch_norm__66, batch_norm__72)

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__11 = paddle._C_ops.relu(add__5)

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x256x8x25xf32, 256x256x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu__11, parameter_81, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_82, parameter_83, parameter_84, parameter_85, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__12 = paddle._C_ops.relu(batch_norm__78)

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x256x8x25xf32, 256x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu__12, parameter_86, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_87, parameter_88, parameter_89, parameter_90, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x8x25xf32) <- (-1x256x8x25xf32, -1x256x8x25xf32)
        add__6 = paddle._C_ops.add(batch_norm__84, relu__11)

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__13 = paddle._C_ops.relu(add__6)

        # pd_op.conv2d: (-1x256x8x25xf32) <- (-1x256x8x25xf32, 256x256x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__13, parameter_91, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x8x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x8x25xf32) <- (-1x256x8x25xf32)
        relu__14 = paddle._C_ops.relu(batch_norm__90)

        # pd_op.pool2d: (-1x256x4x26xf32) <- (-1x256x8x25xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(relu__14, constant_0, [2, 1], [0, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x256x4x26xf32, 512x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(pool2d_6, parameter_96, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_97, parameter_98, parameter_99, parameter_100, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__15 = paddle._C_ops.relu(batch_norm__96)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu__15, parameter_101, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_102, parameter_103, parameter_104, parameter_105, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x256x4x26xf32, 512x256x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(pool2d_6, parameter_106, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_107, parameter_108, parameter_109, parameter_110, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__7 = paddle._C_ops.add(batch_norm__102, batch_norm__108)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__16 = paddle._C_ops.relu(add__7)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu__16, parameter_111, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_112, parameter_113, parameter_114, parameter_115, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__17 = paddle._C_ops.relu(batch_norm__114)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu__17, parameter_116, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_117, parameter_118, parameter_119, parameter_120, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__8 = paddle._C_ops.add(batch_norm__120, relu__16)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__18 = paddle._C_ops.relu(add__8)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu__18, parameter_121, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_122, parameter_123, parameter_124, parameter_125, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__19 = paddle._C_ops.relu(batch_norm__126)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu__19, parameter_126, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_127, parameter_128, parameter_129, parameter_130, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__9 = paddle._C_ops.add(batch_norm__132, relu__18)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__20 = paddle._C_ops.relu(add__9)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu__20, parameter_131, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_132, parameter_133, parameter_134, parameter_135, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__21 = paddle._C_ops.relu(batch_norm__138)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__21, parameter_136, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_137, parameter_138, parameter_139, parameter_140, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__10 = paddle._C_ops.add(batch_norm__144, relu__20)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__22 = paddle._C_ops.relu(add__10)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu__22, parameter_141, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_142, parameter_143, parameter_144, parameter_145, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__23 = paddle._C_ops.relu(batch_norm__150)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu__23, parameter_146, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_147, parameter_148, parameter_149, parameter_150, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__11 = paddle._C_ops.add(batch_norm__156, relu__22)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__24 = paddle._C_ops.relu(add__11)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu__24, parameter_151, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_152, parameter_153, parameter_154, parameter_155, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__25 = paddle._C_ops.relu(batch_norm__162)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__25, parameter_156, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_157, parameter_158, parameter_159, parameter_160, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__26 = paddle._C_ops.relu(batch_norm__168)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu__26, parameter_161, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_162, parameter_163, parameter_164, parameter_165, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__12 = paddle._C_ops.add(batch_norm__174, relu__25)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__27 = paddle._C_ops.relu(add__12)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__27, parameter_166, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_167, parameter_168, parameter_169, parameter_170, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__28 = paddle._C_ops.relu(batch_norm__180)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu__28, parameter_171, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_172, parameter_173, parameter_174, parameter_175, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__13 = paddle._C_ops.add(batch_norm__186, relu__27)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__29 = paddle._C_ops.relu(add__13)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu__29, parameter_176, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_177, parameter_178, parameter_179, parameter_180, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__30 = paddle._C_ops.relu(batch_norm__192)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(relu__30, parameter_181, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_182, parameter_183, parameter_184, parameter_185, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__14 = paddle._C_ops.add(batch_norm__198, relu__29)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__31 = paddle._C_ops.relu(add__14)

        # pd_op.conv2d: (-1x512x2x27xf32) <- (-1x512x4x26xf32, 512x512x2x2xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu__31, parameter_186, [2, 1], [0, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x27xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x2x27xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_187, parameter_188, parameter_189, parameter_190, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x2x27xf32) <- (-1x512x2x27xf32)
        relu__32 = paddle._C_ops.relu(batch_norm__204)

        # pd_op.conv2d: (-1x512x1x26xf32) <- (-1x512x2x27xf32, 512x512x2x2xf32)
        conv2d_35 = paddle._C_ops.conv2d(relu__32, parameter_191, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x1x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x1x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_192, parameter_193, parameter_194, parameter_195, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x1x26xf32) <- (-1x512x1x26xf32)
        relu__33 = paddle._C_ops.relu(batch_norm__210)

        # pd_op.pool2d: (-1x256x4x26xf32) <- (-1x256x8x25xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(relu__14, constant_0, [2, 1], [0, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x256x4x26xf32, 512x256x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(pool2d_7, parameter_196, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_197, parameter_198, parameter_199, parameter_200, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__34 = paddle._C_ops.relu(batch_norm__216)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(relu__34, parameter_201, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_202, parameter_203, parameter_204, parameter_205, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x256x4x26xf32, 512x256x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(pool2d_7, parameter_206, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_207, parameter_208, parameter_209, parameter_210, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__15 = paddle._C_ops.add(batch_norm__222, batch_norm__228)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__35 = paddle._C_ops.relu(add__15)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(relu__35, parameter_211, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_212, parameter_213, parameter_214, parameter_215, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__36 = paddle._C_ops.relu(batch_norm__234)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(relu__36, parameter_216, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_217, parameter_218, parameter_219, parameter_220, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__16 = paddle._C_ops.add(batch_norm__240, relu__35)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__37 = paddle._C_ops.relu(add__16)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(relu__37, parameter_221, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_222, parameter_223, parameter_224, parameter_225, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__38 = paddle._C_ops.relu(batch_norm__246)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(relu__38, parameter_226, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_227, parameter_228, parameter_229, parameter_230, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__17 = paddle._C_ops.add(batch_norm__252, relu__37)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__39 = paddle._C_ops.relu(add__17)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_43 = paddle._C_ops.conv2d(relu__39, parameter_231, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_232, parameter_233, parameter_234, parameter_235, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__40 = paddle._C_ops.relu(batch_norm__258)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(relu__40, parameter_236, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_237, parameter_238, parameter_239, parameter_240, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__18 = paddle._C_ops.add(batch_norm__264, relu__39)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__41 = paddle._C_ops.relu(add__18)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(relu__41, parameter_241, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_242, parameter_243, parameter_244, parameter_245, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__42 = paddle._C_ops.relu(batch_norm__270)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_46 = paddle._C_ops.conv2d(relu__42, parameter_246, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_247, parameter_248, parameter_249, parameter_250, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__19 = paddle._C_ops.add(batch_norm__276, relu__41)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__43 = paddle._C_ops.relu(add__19)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_47 = paddle._C_ops.conv2d(relu__43, parameter_251, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_252, parameter_253, parameter_254, parameter_255, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__44 = paddle._C_ops.relu(batch_norm__282)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(relu__44, parameter_256, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_257, parameter_258, parameter_259, parameter_260, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__45 = paddle._C_ops.relu(batch_norm__288)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_49 = paddle._C_ops.conv2d(relu__45, parameter_261, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_262, parameter_263, parameter_264, parameter_265, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__20 = paddle._C_ops.add(batch_norm__294, relu__44)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__46 = paddle._C_ops.relu(add__20)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_50 = paddle._C_ops.conv2d(relu__46, parameter_266, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_267, parameter_268, parameter_269, parameter_270, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__47 = paddle._C_ops.relu(batch_norm__300)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_51 = paddle._C_ops.conv2d(relu__47, parameter_271, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_272, parameter_273, parameter_274, parameter_275, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__21 = paddle._C_ops.add(batch_norm__306, relu__46)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__48 = paddle._C_ops.relu(add__21)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_52 = paddle._C_ops.conv2d(relu__48, parameter_276, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_277, parameter_278, parameter_279, parameter_280, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__49 = paddle._C_ops.relu(batch_norm__312)

        # pd_op.conv2d: (-1x512x4x26xf32) <- (-1x512x4x26xf32, 512x512x3x3xf32)
        conv2d_53 = paddle._C_ops.conv2d(relu__49, parameter_281, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x4x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_282, parameter_283, parameter_284, parameter_285, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x4x26xf32) <- (-1x512x4x26xf32, -1x512x4x26xf32)
        add__22 = paddle._C_ops.add(batch_norm__318, relu__48)

        # pd_op.relu_: (-1x512x4x26xf32) <- (-1x512x4x26xf32)
        relu__50 = paddle._C_ops.relu(add__22)

        # pd_op.conv2d: (-1x512x2x27xf32) <- (-1x512x4x26xf32, 512x512x2x2xf32)
        conv2d_54 = paddle._C_ops.conv2d(relu__50, parameter_286, [2, 1], [0, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x27xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x2x27xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_287, parameter_288, parameter_289, parameter_290, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x2x27xf32) <- (-1x512x2x27xf32)
        relu__51 = paddle._C_ops.relu(batch_norm__324)

        # pd_op.conv2d: (-1x512x1x26xf32) <- (-1x512x2x27xf32, 512x512x2x2xf32)
        conv2d_55 = paddle._C_ops.conv2d(relu__51, parameter_291, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x1x26xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x1x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_292, parameter_293, parameter_294, parameter_295, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x1x26xf32) <- (-1x512x1x26xf32)
        relu__52 = paddle._C_ops.relu(batch_norm__330)

        # pd_op.shape: (4xi32) <- (-1x512x1x26xf32)
        shape_1 = paddle._C_ops.shape(relu__33)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_1, [0], constant_11, constant_9, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_3, constant_16, constant_17, constant_18]

        # pd_op.reshape_: (-1x512x1x26xf32, 0x-1x512x1x26xf32) <- (-1x512x1x26xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__33, combine_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.squeeze: (-1x512x26xf32, None) <- (-1x512x1x26xf32, 1xi64)
        squeeze_0, squeeze_1 = (lambda x, f: f(x))(paddle._C_ops.squeeze(reshape__8, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x26x512xf32) <- (-1x512x26xf32)
        transpose_2 = paddle._C_ops.transpose(squeeze_0, [0, 2, 1])

        # pd_op.matmul: (-1x26x512xf32) <- (-1x26x512xf32, 512x512xf32)
        matmul_5 = paddle.matmul(transpose_2, parameter_296, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x26xf32) <- (-1x26x512xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_5, [0, 2, 1])

        # pd_op.batch_norm_: (-1x512x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_3, parameter_297, parameter_298, parameter_299, parameter_300, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x26xf32) <- (-1x512x26xf32)
        relu__53 = paddle._C_ops.relu(batch_norm__336)

        # pd_op.unsqueeze_: (-1x512x1x26xf32, None) <- (-1x512x26xf32, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(relu__53, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x512x1x26xf32) <- (-1x512x1x26xf32, -1x512x1x26xf32)
        multiply_0 = relu__52 * unsqueeze__2

        # pd_op.squeeze: (-1x512x26xf32, None) <- (-1x512x1x26xf32, 1xi64)
        squeeze_2, squeeze_3 = (lambda x, f: f(x))(paddle._C_ops.squeeze(relu__52, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x26x512xf32) <- (-1x512x26xf32)
        transpose_4 = paddle._C_ops.transpose(squeeze_2, [0, 2, 1])

        # pd_op.matmul: (-1x26x512xf32) <- (-1x26x512xf32, 512x512xf32)
        matmul_6 = paddle.matmul(transpose_4, parameter_301, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (-1x512x26xf32) <- (-1x26x512xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_6, [0, 2, 1])

        # pd_op.batch_norm_: (-1x512x26xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x26xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_5, parameter_302, parameter_303, parameter_304, parameter_305, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x26xf32) <- (-1x512x26xf32)
        relu__54 = paddle._C_ops.relu(batch_norm__342)

        # pd_op.unsqueeze_: (-1x512x1x26xf32, None) <- (-1x512x26xf32, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(relu__54, constant_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x512x1x26xf32) <- (-1x512x1x26xf32, -1x512x1x26xf32)
        multiply__2 = paddle._C_ops.multiply(relu__52, unsqueeze__4)

        # pd_op.add_: (-1x512x1x26xf32) <- (-1x512x1x26xf32, -1x512x1x26xf32)
        add__23 = paddle._C_ops.add(reshape__8, multiply__2)

        # pd_op.shape: (4xi32) <- (-1x512x1x26xf32)
        shape_2 = paddle._C_ops.shape(multiply_0)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_2, [0], constant_11, constant_9, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_4, constant_16, constant_17, constant_18]

        # pd_op.reshape_: (-1x512x1x26xf32, 0x-1x512x1x26xf32) <- (-1x512x1x26xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape(multiply_0, combine_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.squeeze_: (-1x512x26xf32, None) <- (-1x512x1x26xf32, 1xi64)
        squeeze__4, squeeze__5 = (lambda x, f: f(x))(paddle._C_ops.squeeze(reshape__10, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x26x512xf32) <- (-1x512x26xf32)
        transpose_6 = paddle._C_ops.transpose(squeeze__4, [0, 2, 1])

        # pd_op.shape: (4xi32) <- (-1x512x1x26xf32)
        shape_3 = paddle._C_ops.shape(add__23)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_3, [0], constant_11, constant_9, [1], [0])

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_13 = [slice_5, constant_16, constant_18]

        # pd_op.reshape_: (-1x512x26xf32, 0x-1x512x1x26xf32) <- (-1x512x1x26xf32, [xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__23, combine_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x26x512xf32) <- (-1x512x26xf32)
        transpose_7 = paddle._C_ops.transpose(reshape__12, [0, 2, 1])

        # pd_op.matmul: (-1x26x512xf32) <- (-1x26x512xf32, 512x512xf32)
        matmul_7 = paddle.matmul(transpose_7, parameter_306, transpose_x=False, transpose_y=False)

        # pd_op.shape: (3xi32) <- (-1x26x512xf32)
        shape_4 = paddle._C_ops.shape(matmul_7)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_4, [0], constant_11, constant_9, [1], [0])

        # builtin.combine: ([xi32, 1xi32]) <- (xi32, 1xi32)
        combine_14 = [slice_6, constant_20]

        # pd_op.reshape_: (-1x13312xf32, 0x-1x26x512xf32) <- (-1x26x512xf32, [xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape(matmul_7, combine_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x38xf32) <- (-1x13312xf32, 13312x38xf32)
        matmul_8 = paddle.matmul(reshape__14, parameter_307, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__24 = paddle._C_ops.add(matmul_8, parameter_308)

        # pd_op.shape: (3xi32) <- (-1x26x512xf32)
        shape_5 = paddle._C_ops.shape(transpose_6)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_5, [0], constant_11, constant_9, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_15 = [slice_7, parameter_309]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(slice_7, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_16 = [memcpy_h2d_0, parameter_309]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_3 = paddle._C_ops.stack(combine_16, 0)

        # pd_op.full_with_tensor: (-1x256xf32) <- (1xf32, 2xi32)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(full_3, stack_3, paddle.float32)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_17 = [slice_7, parameter_309]

        # pd_op.memcpy_h2d: (xi32) <- (xi32)
        memcpy_h2d_1 = paddle._C_ops.memcpy_h2d(slice_7, 1)

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_18 = [memcpy_h2d_1, parameter_309]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_4 = paddle._C_ops.stack(combine_18, 0)

        # pd_op.full_with_tensor: (-1x256xf32) <- (1xf32, 2xi32)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_3, stack_4, paddle.float32)

        # builtin.combine: ([xi32]) <- (xi32)
        combine_19 = [slice_7]

        # pd_op.stack: (1xi32) <- ([xi32])
        stack_5 = paddle._C_ops.stack(combine_19, 0)

        # pd_op.full_with_tensor: (-1xi32) <- (1xf32, 1xi32)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(full_3, stack_5, paddle.int32)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(full_with_tensor_2 % paddle.cast(constant_21, full_with_tensor_2.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_9 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_10 = paddle.matmul(full_with_tensor_0, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__25 = paddle._C_ops.add(matmul_10, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__25, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__26 = paddle._C_ops.add(matmul_9, unsqueeze__6)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__0 = paddle._C_ops.tanh(add__26)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_11 = paddle.matmul(tanh__0, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__0 = paddle._C_ops.softmax(matmul_11, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_8 = paddle._C_ops.transpose(softmax__0, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_12 = paddle.matmul(transpose_8, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__6, squeeze__7 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_12, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_20 = [squeeze__6, one_hot_0]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_20, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_13 = paddle.matmul(concat_6, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__27 = paddle._C_ops.add(matmul_13, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_14 = paddle.matmul(full_with_tensor_0, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__28 = paddle._C_ops.add(add__27, matmul_14)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__29 = paddle._C_ops.add(add__28, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add__29, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_8 = split_with_num_0[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__0 = paddle._C_ops.sigmoid(slice_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_9 = split_with_num_0[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__1 = paddle._C_ops.sigmoid(slice_9)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_10 = split_with_num_0[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__2 = paddle._C_ops.sigmoid(slice_10)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__3 = paddle._C_ops.multiply(sigmoid__1, full_with_tensor_1)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_11 = split_with_num_0[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__1 = paddle._C_ops.tanh(slice_11)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__4 = paddle._C_ops.multiply(sigmoid__0, tanh__1)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__30 = paddle._C_ops.add(multiply__3, multiply__4)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_0 = paddle._C_ops.tanh(add__30)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__5 = paddle._C_ops.multiply(sigmoid__2, tanh_0)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_15 = paddle.matmul(multiply__5, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__31 = paddle._C_ops.add(matmul_15, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_2, unsqueeze_3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__31, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(add__31, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_1 = paddle._C_ops.one_hot(argmax_0 % paddle.cast(constant_21, argmax_0.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_16 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_17 = paddle.matmul(multiply__5, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__32 = paddle._C_ops.add(matmul_17, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__32, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__33 = paddle._C_ops.add(matmul_16, unsqueeze__8)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__2 = paddle._C_ops.tanh(add__33)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_18 = paddle.matmul(tanh__2, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__1 = paddle._C_ops.softmax(matmul_18, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_9 = paddle._C_ops.transpose(softmax__1, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_19 = paddle.matmul(transpose_9, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__8, squeeze__9 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_19, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_21 = [squeeze__8, one_hot_1]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_21, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_20 = paddle.matmul(concat_7, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__34 = paddle._C_ops.add(matmul_20, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_21 = paddle.matmul(multiply__5, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__35 = paddle._C_ops.add(add__34, matmul_21)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__36 = paddle._C_ops.add(add__35, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(add__36, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_12 = split_with_num_1[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__3 = paddle._C_ops.sigmoid(slice_12)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_13 = split_with_num_1[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__4 = paddle._C_ops.sigmoid(slice_13)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_14 = split_with_num_1[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__5 = paddle._C_ops.sigmoid(slice_14)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__6 = paddle._C_ops.multiply(sigmoid__4, add__30)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_15 = split_with_num_1[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__3 = paddle._C_ops.tanh(slice_15)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__7 = paddle._C_ops.multiply(sigmoid__3, tanh__3)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__37 = paddle._C_ops.add(multiply__6, multiply__7)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_1 = paddle._C_ops.tanh(add__37)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__8 = paddle._C_ops.multiply(sigmoid__5, tanh_1)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_22 = paddle.matmul(multiply__8, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__38 = paddle._C_ops.add(matmul_22, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_4, unsqueeze_5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__38, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x1x38xf32, -1x1x38xf32]) <- (-1x1x38xf32, -1x1x38xf32)
        combine_22 = [unsqueeze_2, unsqueeze_4]

        # pd_op.concat: (-1x2x38xf32) <- ([-1x1x38xf32, -1x1x38xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_22, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_1 = paddle._C_ops.argmax(add__38, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_2 = paddle._C_ops.one_hot(argmax_1 % paddle.cast(constant_21, argmax_1.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_23 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_24 = paddle.matmul(multiply__8, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__39 = paddle._C_ops.add(matmul_24, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__10, unsqueeze__11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__39, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__40 = paddle._C_ops.add(matmul_23, unsqueeze__10)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__4 = paddle._C_ops.tanh(add__40)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_25 = paddle.matmul(tanh__4, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__2 = paddle._C_ops.softmax(matmul_25, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_10 = paddle._C_ops.transpose(softmax__2, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_26 = paddle.matmul(transpose_10, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__10, squeeze__11 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_26, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_23 = [squeeze__10, one_hot_2]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_23, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_27 = paddle.matmul(concat_9, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__41 = paddle._C_ops.add(matmul_27, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_28 = paddle.matmul(multiply__8, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__42 = paddle._C_ops.add(add__41, matmul_28)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__43 = paddle._C_ops.add(add__42, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(add__43, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_16 = split_with_num_2[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__6 = paddle._C_ops.sigmoid(slice_16)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_17 = split_with_num_2[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__7 = paddle._C_ops.sigmoid(slice_17)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_18 = split_with_num_2[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__8 = paddle._C_ops.sigmoid(slice_18)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__9 = paddle._C_ops.multiply(sigmoid__7, add__37)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_19 = split_with_num_2[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__5 = paddle._C_ops.tanh(slice_19)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__10 = paddle._C_ops.multiply(sigmoid__6, tanh__5)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__44 = paddle._C_ops.add(multiply__9, multiply__10)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_2 = paddle._C_ops.tanh(add__44)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__11 = paddle._C_ops.multiply(sigmoid__8, tanh_2)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_29 = paddle.matmul(multiply__11, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__45 = paddle._C_ops.add(matmul_29, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_6, unsqueeze_7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__45, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x2x38xf32, -1x1x38xf32]) <- (-1x2x38xf32, -1x1x38xf32)
        combine_24 = [concat_8, unsqueeze_6]

        # pd_op.concat: (-1x3x38xf32) <- ([-1x2x38xf32, -1x1x38xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_24, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_2 = paddle._C_ops.argmax(add__45, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_3 = paddle._C_ops.one_hot(argmax_2 % paddle.cast(constant_21, argmax_2.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_30 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_31 = paddle.matmul(multiply__11, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__46 = paddle._C_ops.add(matmul_31, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__12, unsqueeze__13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__46, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__47 = paddle._C_ops.add(matmul_30, unsqueeze__12)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__6 = paddle._C_ops.tanh(add__47)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_32 = paddle.matmul(tanh__6, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__3 = paddle._C_ops.softmax(matmul_32, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_11 = paddle._C_ops.transpose(softmax__3, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_33 = paddle.matmul(transpose_11, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__12, squeeze__13 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_33, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_25 = [squeeze__12, one_hot_3]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_25, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_34 = paddle.matmul(concat_11, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__48 = paddle._C_ops.add(matmul_34, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_35 = paddle.matmul(multiply__11, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__49 = paddle._C_ops.add(add__48, matmul_35)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__50 = paddle._C_ops.add(add__49, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(add__50, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_20 = split_with_num_3[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__9 = paddle._C_ops.sigmoid(slice_20)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_21 = split_with_num_3[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__10 = paddle._C_ops.sigmoid(slice_21)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_22 = split_with_num_3[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__11 = paddle._C_ops.sigmoid(slice_22)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__12 = paddle._C_ops.multiply(sigmoid__10, add__44)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_23 = split_with_num_3[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__7 = paddle._C_ops.tanh(slice_23)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__13 = paddle._C_ops.multiply(sigmoid__9, tanh__7)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__51 = paddle._C_ops.add(multiply__12, multiply__13)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_3 = paddle._C_ops.tanh(add__51)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__14 = paddle._C_ops.multiply(sigmoid__11, tanh_3)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_36 = paddle.matmul(multiply__14, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__52 = paddle._C_ops.add(matmul_36, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_8, unsqueeze_9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__52, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x3x38xf32, -1x1x38xf32]) <- (-1x3x38xf32, -1x1x38xf32)
        combine_26 = [concat_10, unsqueeze_8]

        # pd_op.concat: (-1x4x38xf32) <- ([-1x3x38xf32, -1x1x38xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_26, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_3 = paddle._C_ops.argmax(add__52, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_4 = paddle._C_ops.one_hot(argmax_3 % paddle.cast(constant_21, argmax_3.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_37 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_38 = paddle.matmul(multiply__14, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__53 = paddle._C_ops.add(matmul_38, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__14, unsqueeze__15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__53, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__54 = paddle._C_ops.add(matmul_37, unsqueeze__14)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__8 = paddle._C_ops.tanh(add__54)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_39 = paddle.matmul(tanh__8, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__4 = paddle._C_ops.softmax(matmul_39, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_12 = paddle._C_ops.transpose(softmax__4, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_40 = paddle.matmul(transpose_12, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__14, squeeze__15 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_40, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_27 = [squeeze__14, one_hot_4]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_27, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_41 = paddle.matmul(concat_13, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__55 = paddle._C_ops.add(matmul_41, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_42 = paddle.matmul(multiply__14, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__56 = paddle._C_ops.add(add__55, matmul_42)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__57 = paddle._C_ops.add(add__56, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(add__57, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_24 = split_with_num_4[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__12 = paddle._C_ops.sigmoid(slice_24)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_25 = split_with_num_4[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__13 = paddle._C_ops.sigmoid(slice_25)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_26 = split_with_num_4[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__14 = paddle._C_ops.sigmoid(slice_26)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__15 = paddle._C_ops.multiply(sigmoid__13, add__51)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_27 = split_with_num_4[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__9 = paddle._C_ops.tanh(slice_27)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__16 = paddle._C_ops.multiply(sigmoid__12, tanh__9)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__58 = paddle._C_ops.add(multiply__15, multiply__16)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_4 = paddle._C_ops.tanh(add__58)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__17 = paddle._C_ops.multiply(sigmoid__14, tanh_4)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_43 = paddle.matmul(multiply__17, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__59 = paddle._C_ops.add(matmul_43, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_10, unsqueeze_11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__59, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x4x38xf32, -1x1x38xf32]) <- (-1x4x38xf32, -1x1x38xf32)
        combine_28 = [concat_12, unsqueeze_10]

        # pd_op.concat: (-1x5x38xf32) <- ([-1x4x38xf32, -1x1x38xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_28, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_4 = paddle._C_ops.argmax(add__59, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_5 = paddle._C_ops.one_hot(argmax_4 % paddle.cast(constant_21, argmax_4.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_44 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_45 = paddle.matmul(multiply__17, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__60 = paddle._C_ops.add(matmul_45, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__16, unsqueeze__17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__60, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__61 = paddle._C_ops.add(matmul_44, unsqueeze__16)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__10 = paddle._C_ops.tanh(add__61)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_46 = paddle.matmul(tanh__10, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__5 = paddle._C_ops.softmax(matmul_46, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_13 = paddle._C_ops.transpose(softmax__5, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_47 = paddle.matmul(transpose_13, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__16, squeeze__17 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_47, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_29 = [squeeze__16, one_hot_5]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_29, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_48 = paddle.matmul(concat_15, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__62 = paddle._C_ops.add(matmul_48, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_49 = paddle.matmul(multiply__17, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__63 = paddle._C_ops.add(add__62, matmul_49)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__64 = paddle._C_ops.add(add__63, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(add__64, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_28 = split_with_num_5[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__15 = paddle._C_ops.sigmoid(slice_28)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_29 = split_with_num_5[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__16 = paddle._C_ops.sigmoid(slice_29)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_30 = split_with_num_5[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__17 = paddle._C_ops.sigmoid(slice_30)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__18 = paddle._C_ops.multiply(sigmoid__16, add__58)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_31 = split_with_num_5[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__11 = paddle._C_ops.tanh(slice_31)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__19 = paddle._C_ops.multiply(sigmoid__15, tanh__11)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__65 = paddle._C_ops.add(multiply__18, multiply__19)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_5 = paddle._C_ops.tanh(add__65)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__20 = paddle._C_ops.multiply(sigmoid__17, tanh_5)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_50 = paddle.matmul(multiply__20, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__66 = paddle._C_ops.add(matmul_50, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_12, unsqueeze_13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__66, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x5x38xf32, -1x1x38xf32]) <- (-1x5x38xf32, -1x1x38xf32)
        combine_30 = [concat_14, unsqueeze_12]

        # pd_op.concat: (-1x6x38xf32) <- ([-1x5x38xf32, -1x1x38xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_30, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_5 = paddle._C_ops.argmax(add__66, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_6 = paddle._C_ops.one_hot(argmax_5 % paddle.cast(constant_21, argmax_5.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_51 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_52 = paddle.matmul(multiply__20, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__67 = paddle._C_ops.add(matmul_52, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__18, unsqueeze__19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__67, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__68 = paddle._C_ops.add(matmul_51, unsqueeze__18)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__12 = paddle._C_ops.tanh(add__68)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_53 = paddle.matmul(tanh__12, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__6 = paddle._C_ops.softmax(matmul_53, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_14 = paddle._C_ops.transpose(softmax__6, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_54 = paddle.matmul(transpose_14, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__18, squeeze__19 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_54, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_31 = [squeeze__18, one_hot_6]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_31, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_55 = paddle.matmul(concat_17, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__69 = paddle._C_ops.add(matmul_55, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_56 = paddle.matmul(multiply__20, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__70 = paddle._C_ops.add(add__69, matmul_56)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__71 = paddle._C_ops.add(add__70, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(add__71, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_32 = split_with_num_6[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__18 = paddle._C_ops.sigmoid(slice_32)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_33 = split_with_num_6[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__19 = paddle._C_ops.sigmoid(slice_33)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_34 = split_with_num_6[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__20 = paddle._C_ops.sigmoid(slice_34)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__21 = paddle._C_ops.multiply(sigmoid__19, add__65)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_35 = split_with_num_6[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__13 = paddle._C_ops.tanh(slice_35)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__22 = paddle._C_ops.multiply(sigmoid__18, tanh__13)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__72 = paddle._C_ops.add(multiply__21, multiply__22)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_6 = paddle._C_ops.tanh(add__72)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__23 = paddle._C_ops.multiply(sigmoid__20, tanh_6)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_57 = paddle.matmul(multiply__23, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__73 = paddle._C_ops.add(matmul_57, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_14, unsqueeze_15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__73, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x6x38xf32, -1x1x38xf32]) <- (-1x6x38xf32, -1x1x38xf32)
        combine_32 = [concat_16, unsqueeze_14]

        # pd_op.concat: (-1x7x38xf32) <- ([-1x6x38xf32, -1x1x38xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_32, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_6 = paddle._C_ops.argmax(add__73, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_7 = paddle._C_ops.one_hot(argmax_6 % paddle.cast(constant_21, argmax_6.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_58 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_59 = paddle.matmul(multiply__23, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__74 = paddle._C_ops.add(matmul_59, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__20, unsqueeze__21 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__74, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__75 = paddle._C_ops.add(matmul_58, unsqueeze__20)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__14 = paddle._C_ops.tanh(add__75)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_60 = paddle.matmul(tanh__14, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__7 = paddle._C_ops.softmax(matmul_60, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_15 = paddle._C_ops.transpose(softmax__7, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_61 = paddle.matmul(transpose_15, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__20, squeeze__21 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_61, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_33 = [squeeze__20, one_hot_7]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_33, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_62 = paddle.matmul(concat_19, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__76 = paddle._C_ops.add(matmul_62, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_63 = paddle.matmul(multiply__23, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__77 = paddle._C_ops.add(add__76, matmul_63)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__78 = paddle._C_ops.add(add__77, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(add__78, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_36 = split_with_num_7[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__21 = paddle._C_ops.sigmoid(slice_36)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_37 = split_with_num_7[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__22 = paddle._C_ops.sigmoid(slice_37)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_38 = split_with_num_7[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__23 = paddle._C_ops.sigmoid(slice_38)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__24 = paddle._C_ops.multiply(sigmoid__22, add__72)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_39 = split_with_num_7[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__15 = paddle._C_ops.tanh(slice_39)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__25 = paddle._C_ops.multiply(sigmoid__21, tanh__15)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__79 = paddle._C_ops.add(multiply__24, multiply__25)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_7 = paddle._C_ops.tanh(add__79)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__26 = paddle._C_ops.multiply(sigmoid__23, tanh_7)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_64 = paddle.matmul(multiply__26, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__80 = paddle._C_ops.add(matmul_64, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_16, unsqueeze_17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__80, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x7x38xf32, -1x1x38xf32]) <- (-1x7x38xf32, -1x1x38xf32)
        combine_34 = [concat_18, unsqueeze_16]

        # pd_op.concat: (-1x8x38xf32) <- ([-1x7x38xf32, -1x1x38xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_34, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_7 = paddle._C_ops.argmax(add__80, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_8 = paddle._C_ops.one_hot(argmax_7 % paddle.cast(constant_21, argmax_7.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_65 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_66 = paddle.matmul(multiply__26, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__81 = paddle._C_ops.add(matmul_66, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__22, unsqueeze__23 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__81, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__82 = paddle._C_ops.add(matmul_65, unsqueeze__22)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__16 = paddle._C_ops.tanh(add__82)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_67 = paddle.matmul(tanh__16, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__8 = paddle._C_ops.softmax(matmul_67, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_16 = paddle._C_ops.transpose(softmax__8, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_68 = paddle.matmul(transpose_16, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__22, squeeze__23 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_68, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_35 = [squeeze__22, one_hot_8]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_35, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_69 = paddle.matmul(concat_21, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__83 = paddle._C_ops.add(matmul_69, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_70 = paddle.matmul(multiply__26, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__84 = paddle._C_ops.add(add__83, matmul_70)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__85 = paddle._C_ops.add(add__84, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(add__85, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_40 = split_with_num_8[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__24 = paddle._C_ops.sigmoid(slice_40)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_41 = split_with_num_8[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__25 = paddle._C_ops.sigmoid(slice_41)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_42 = split_with_num_8[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__26 = paddle._C_ops.sigmoid(slice_42)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__27 = paddle._C_ops.multiply(sigmoid__25, add__79)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_43 = split_with_num_8[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__17 = paddle._C_ops.tanh(slice_43)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__28 = paddle._C_ops.multiply(sigmoid__24, tanh__17)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__86 = paddle._C_ops.add(multiply__27, multiply__28)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_8 = paddle._C_ops.tanh(add__86)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__29 = paddle._C_ops.multiply(sigmoid__26, tanh_8)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_71 = paddle.matmul(multiply__29, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__87 = paddle._C_ops.add(matmul_71, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_18, unsqueeze_19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__87, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x8x38xf32, -1x1x38xf32]) <- (-1x8x38xf32, -1x1x38xf32)
        combine_36 = [concat_20, unsqueeze_18]

        # pd_op.concat: (-1x9x38xf32) <- ([-1x8x38xf32, -1x1x38xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_36, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_8 = paddle._C_ops.argmax(add__87, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_9 = paddle._C_ops.one_hot(argmax_8 % paddle.cast(constant_21, argmax_8.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_72 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_73 = paddle.matmul(multiply__29, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__88 = paddle._C_ops.add(matmul_73, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__24, unsqueeze__25 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__88, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__89 = paddle._C_ops.add(matmul_72, unsqueeze__24)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__18 = paddle._C_ops.tanh(add__89)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_74 = paddle.matmul(tanh__18, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__9 = paddle._C_ops.softmax(matmul_74, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_17 = paddle._C_ops.transpose(softmax__9, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_75 = paddle.matmul(transpose_17, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__24, squeeze__25 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_75, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_37 = [squeeze__24, one_hot_9]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_37, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_76 = paddle.matmul(concat_23, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__90 = paddle._C_ops.add(matmul_76, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_77 = paddle.matmul(multiply__29, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__91 = paddle._C_ops.add(add__90, matmul_77)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__92 = paddle._C_ops.add(add__91, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(add__92, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_44 = split_with_num_9[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__27 = paddle._C_ops.sigmoid(slice_44)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_45 = split_with_num_9[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__28 = paddle._C_ops.sigmoid(slice_45)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_46 = split_with_num_9[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__29 = paddle._C_ops.sigmoid(slice_46)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__30 = paddle._C_ops.multiply(sigmoid__28, add__86)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_47 = split_with_num_9[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__19 = paddle._C_ops.tanh(slice_47)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__31 = paddle._C_ops.multiply(sigmoid__27, tanh__19)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__93 = paddle._C_ops.add(multiply__30, multiply__31)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_9 = paddle._C_ops.tanh(add__93)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__32 = paddle._C_ops.multiply(sigmoid__29, tanh_9)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_78 = paddle.matmul(multiply__32, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__94 = paddle._C_ops.add(matmul_78, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_20, unsqueeze_21 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__94, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x9x38xf32, -1x1x38xf32]) <- (-1x9x38xf32, -1x1x38xf32)
        combine_38 = [concat_22, unsqueeze_20]

        # pd_op.concat: (-1x10x38xf32) <- ([-1x9x38xf32, -1x1x38xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_38, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_9 = paddle._C_ops.argmax(add__94, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_10 = paddle._C_ops.one_hot(argmax_9 % paddle.cast(constant_21, argmax_9.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_79 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_80 = paddle.matmul(multiply__32, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__95 = paddle._C_ops.add(matmul_80, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__26, unsqueeze__27 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__95, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__96 = paddle._C_ops.add(matmul_79, unsqueeze__26)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__20 = paddle._C_ops.tanh(add__96)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_81 = paddle.matmul(tanh__20, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__10 = paddle._C_ops.softmax(matmul_81, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_18 = paddle._C_ops.transpose(softmax__10, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_82 = paddle.matmul(transpose_18, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__26, squeeze__27 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_82, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_39 = [squeeze__26, one_hot_10]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_39, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_83 = paddle.matmul(concat_25, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__97 = paddle._C_ops.add(matmul_83, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_84 = paddle.matmul(multiply__32, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__98 = paddle._C_ops.add(add__97, matmul_84)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__99 = paddle._C_ops.add(add__98, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(add__99, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_48 = split_with_num_10[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__30 = paddle._C_ops.sigmoid(slice_48)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_49 = split_with_num_10[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__31 = paddle._C_ops.sigmoid(slice_49)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_50 = split_with_num_10[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__32 = paddle._C_ops.sigmoid(slice_50)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__33 = paddle._C_ops.multiply(sigmoid__31, add__93)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_51 = split_with_num_10[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__21 = paddle._C_ops.tanh(slice_51)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__34 = paddle._C_ops.multiply(sigmoid__30, tanh__21)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__100 = paddle._C_ops.add(multiply__33, multiply__34)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_10 = paddle._C_ops.tanh(add__100)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__35 = paddle._C_ops.multiply(sigmoid__32, tanh_10)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_85 = paddle.matmul(multiply__35, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__101 = paddle._C_ops.add(matmul_85, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_22, unsqueeze_23 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__101, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x10x38xf32, -1x1x38xf32]) <- (-1x10x38xf32, -1x1x38xf32)
        combine_40 = [concat_24, unsqueeze_22]

        # pd_op.concat: (-1x11x38xf32) <- ([-1x10x38xf32, -1x1x38xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_40, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_10 = paddle._C_ops.argmax(add__101, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_11 = paddle._C_ops.one_hot(argmax_10 % paddle.cast(constant_21, argmax_10.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_86 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_87 = paddle.matmul(multiply__35, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__102 = paddle._C_ops.add(matmul_87, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__28, unsqueeze__29 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__102, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__103 = paddle._C_ops.add(matmul_86, unsqueeze__28)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__22 = paddle._C_ops.tanh(add__103)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_88 = paddle.matmul(tanh__22, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__11 = paddle._C_ops.softmax(matmul_88, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_19 = paddle._C_ops.transpose(softmax__11, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_89 = paddle.matmul(transpose_19, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__28, squeeze__29 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_89, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_41 = [squeeze__28, one_hot_11]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_41, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_90 = paddle.matmul(concat_27, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__104 = paddle._C_ops.add(matmul_90, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_91 = paddle.matmul(multiply__35, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__105 = paddle._C_ops.add(add__104, matmul_91)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__106 = paddle._C_ops.add(add__105, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(add__106, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_52 = split_with_num_11[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__33 = paddle._C_ops.sigmoid(slice_52)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_53 = split_with_num_11[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__34 = paddle._C_ops.sigmoid(slice_53)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_54 = split_with_num_11[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__35 = paddle._C_ops.sigmoid(slice_54)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__36 = paddle._C_ops.multiply(sigmoid__34, add__100)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_55 = split_with_num_11[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__23 = paddle._C_ops.tanh(slice_55)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__37 = paddle._C_ops.multiply(sigmoid__33, tanh__23)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__107 = paddle._C_ops.add(multiply__36, multiply__37)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_11 = paddle._C_ops.tanh(add__107)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__38 = paddle._C_ops.multiply(sigmoid__35, tanh_11)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_92 = paddle.matmul(multiply__38, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__108 = paddle._C_ops.add(matmul_92, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_24, unsqueeze_25 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__108, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x11x38xf32, -1x1x38xf32]) <- (-1x11x38xf32, -1x1x38xf32)
        combine_42 = [concat_26, unsqueeze_24]

        # pd_op.concat: (-1x12x38xf32) <- ([-1x11x38xf32, -1x1x38xf32], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_42, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_11 = paddle._C_ops.argmax(add__108, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_12 = paddle._C_ops.one_hot(argmax_11 % paddle.cast(constant_21, argmax_11.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_93 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_94 = paddle.matmul(multiply__38, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__109 = paddle._C_ops.add(matmul_94, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__30, unsqueeze__31 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__109, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__110 = paddle._C_ops.add(matmul_93, unsqueeze__30)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__24 = paddle._C_ops.tanh(add__110)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_95 = paddle.matmul(tanh__24, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__12 = paddle._C_ops.softmax(matmul_95, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_20 = paddle._C_ops.transpose(softmax__12, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_96 = paddle.matmul(transpose_20, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__30, squeeze__31 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_96, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_43 = [squeeze__30, one_hot_12]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_43, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_97 = paddle.matmul(concat_29, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__111 = paddle._C_ops.add(matmul_97, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_98 = paddle.matmul(multiply__38, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__112 = paddle._C_ops.add(add__111, matmul_98)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__113 = paddle._C_ops.add(add__112, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_12 = paddle._C_ops.split_with_num(add__113, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_56 = split_with_num_12[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__36 = paddle._C_ops.sigmoid(slice_56)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_57 = split_with_num_12[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__37 = paddle._C_ops.sigmoid(slice_57)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_58 = split_with_num_12[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__38 = paddle._C_ops.sigmoid(slice_58)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__39 = paddle._C_ops.multiply(sigmoid__37, add__107)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_59 = split_with_num_12[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__25 = paddle._C_ops.tanh(slice_59)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__40 = paddle._C_ops.multiply(sigmoid__36, tanh__25)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__114 = paddle._C_ops.add(multiply__39, multiply__40)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_12 = paddle._C_ops.tanh(add__114)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__41 = paddle._C_ops.multiply(sigmoid__38, tanh_12)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_99 = paddle.matmul(multiply__41, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__115 = paddle._C_ops.add(matmul_99, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_26, unsqueeze_27 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__115, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x12x38xf32, -1x1x38xf32]) <- (-1x12x38xf32, -1x1x38xf32)
        combine_44 = [concat_28, unsqueeze_26]

        # pd_op.concat: (-1x13x38xf32) <- ([-1x12x38xf32, -1x1x38xf32], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_44, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_12 = paddle._C_ops.argmax(add__115, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_13 = paddle._C_ops.one_hot(argmax_12 % paddle.cast(constant_21, argmax_12.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_100 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_101 = paddle.matmul(multiply__41, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__116 = paddle._C_ops.add(matmul_101, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__32, unsqueeze__33 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__116, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__117 = paddle._C_ops.add(matmul_100, unsqueeze__32)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__26 = paddle._C_ops.tanh(add__117)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_102 = paddle.matmul(tanh__26, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__13 = paddle._C_ops.softmax(matmul_102, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_21 = paddle._C_ops.transpose(softmax__13, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_103 = paddle.matmul(transpose_21, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__32, squeeze__33 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_103, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_45 = [squeeze__32, one_hot_13]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_45, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_104 = paddle.matmul(concat_31, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__118 = paddle._C_ops.add(matmul_104, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_105 = paddle.matmul(multiply__41, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__119 = paddle._C_ops.add(add__118, matmul_105)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__120 = paddle._C_ops.add(add__119, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_13 = paddle._C_ops.split_with_num(add__120, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_60 = split_with_num_13[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__39 = paddle._C_ops.sigmoid(slice_60)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_61 = split_with_num_13[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__40 = paddle._C_ops.sigmoid(slice_61)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_62 = split_with_num_13[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__41 = paddle._C_ops.sigmoid(slice_62)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__42 = paddle._C_ops.multiply(sigmoid__40, add__114)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_63 = split_with_num_13[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__27 = paddle._C_ops.tanh(slice_63)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__43 = paddle._C_ops.multiply(sigmoid__39, tanh__27)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__121 = paddle._C_ops.add(multiply__42, multiply__43)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_13 = paddle._C_ops.tanh(add__121)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__44 = paddle._C_ops.multiply(sigmoid__41, tanh_13)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_106 = paddle.matmul(multiply__44, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__122 = paddle._C_ops.add(matmul_106, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_28, unsqueeze_29 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__122, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x13x38xf32, -1x1x38xf32]) <- (-1x13x38xf32, -1x1x38xf32)
        combine_46 = [concat_30, unsqueeze_28]

        # pd_op.concat: (-1x14x38xf32) <- ([-1x13x38xf32, -1x1x38xf32], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_46, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_13 = paddle._C_ops.argmax(add__122, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_14 = paddle._C_ops.one_hot(argmax_13 % paddle.cast(constant_21, argmax_13.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_107 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_108 = paddle.matmul(multiply__44, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__123 = paddle._C_ops.add(matmul_108, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__34, unsqueeze__35 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__123, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__124 = paddle._C_ops.add(matmul_107, unsqueeze__34)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__28 = paddle._C_ops.tanh(add__124)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_109 = paddle.matmul(tanh__28, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__14 = paddle._C_ops.softmax(matmul_109, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_22 = paddle._C_ops.transpose(softmax__14, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_110 = paddle.matmul(transpose_22, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__34, squeeze__35 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_110, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_47 = [squeeze__34, one_hot_14]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_47, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_111 = paddle.matmul(concat_33, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__125 = paddle._C_ops.add(matmul_111, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_112 = paddle.matmul(multiply__44, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__126 = paddle._C_ops.add(add__125, matmul_112)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__127 = paddle._C_ops.add(add__126, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_14 = paddle._C_ops.split_with_num(add__127, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_64 = split_with_num_14[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__42 = paddle._C_ops.sigmoid(slice_64)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_65 = split_with_num_14[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__43 = paddle._C_ops.sigmoid(slice_65)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_66 = split_with_num_14[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__44 = paddle._C_ops.sigmoid(slice_66)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__45 = paddle._C_ops.multiply(sigmoid__43, add__121)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_67 = split_with_num_14[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__29 = paddle._C_ops.tanh(slice_67)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__46 = paddle._C_ops.multiply(sigmoid__42, tanh__29)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__128 = paddle._C_ops.add(multiply__45, multiply__46)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_14 = paddle._C_ops.tanh(add__128)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__47 = paddle._C_ops.multiply(sigmoid__44, tanh_14)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_113 = paddle.matmul(multiply__47, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__129 = paddle._C_ops.add(matmul_113, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_30, unsqueeze_31 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__129, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x14x38xf32, -1x1x38xf32]) <- (-1x14x38xf32, -1x1x38xf32)
        combine_48 = [concat_32, unsqueeze_30]

        # pd_op.concat: (-1x15x38xf32) <- ([-1x14x38xf32, -1x1x38xf32], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_48, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_14 = paddle._C_ops.argmax(add__129, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_15 = paddle._C_ops.one_hot(argmax_14 % paddle.cast(constant_21, argmax_14.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_114 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_115 = paddle.matmul(multiply__47, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__130 = paddle._C_ops.add(matmul_115, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__36, unsqueeze__37 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__130, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__131 = paddle._C_ops.add(matmul_114, unsqueeze__36)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__30 = paddle._C_ops.tanh(add__131)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_116 = paddle.matmul(tanh__30, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__15 = paddle._C_ops.softmax(matmul_116, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_23 = paddle._C_ops.transpose(softmax__15, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_117 = paddle.matmul(transpose_23, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__36, squeeze__37 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_117, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_49 = [squeeze__36, one_hot_15]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_49, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_118 = paddle.matmul(concat_35, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__132 = paddle._C_ops.add(matmul_118, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_119 = paddle.matmul(multiply__47, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__133 = paddle._C_ops.add(add__132, matmul_119)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__134 = paddle._C_ops.add(add__133, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_15 = paddle._C_ops.split_with_num(add__134, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_68 = split_with_num_15[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__45 = paddle._C_ops.sigmoid(slice_68)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_69 = split_with_num_15[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__46 = paddle._C_ops.sigmoid(slice_69)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_70 = split_with_num_15[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__47 = paddle._C_ops.sigmoid(slice_70)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__48 = paddle._C_ops.multiply(sigmoid__46, add__128)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_71 = split_with_num_15[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__31 = paddle._C_ops.tanh(slice_71)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__49 = paddle._C_ops.multiply(sigmoid__45, tanh__31)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__135 = paddle._C_ops.add(multiply__48, multiply__49)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_15 = paddle._C_ops.tanh(add__135)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__50 = paddle._C_ops.multiply(sigmoid__47, tanh_15)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_120 = paddle.matmul(multiply__50, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__136 = paddle._C_ops.add(matmul_120, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_32, unsqueeze_33 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__136, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x15x38xf32, -1x1x38xf32]) <- (-1x15x38xf32, -1x1x38xf32)
        combine_50 = [concat_34, unsqueeze_32]

        # pd_op.concat: (-1x16x38xf32) <- ([-1x15x38xf32, -1x1x38xf32], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_50, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_15 = paddle._C_ops.argmax(add__136, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_16 = paddle._C_ops.one_hot(argmax_15 % paddle.cast(constant_21, argmax_15.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_121 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_122 = paddle.matmul(multiply__50, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__137 = paddle._C_ops.add(matmul_122, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__38, unsqueeze__39 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__137, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__138 = paddle._C_ops.add(matmul_121, unsqueeze__38)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__32 = paddle._C_ops.tanh(add__138)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_123 = paddle.matmul(tanh__32, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__16 = paddle._C_ops.softmax(matmul_123, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_24 = paddle._C_ops.transpose(softmax__16, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_124 = paddle.matmul(transpose_24, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__38, squeeze__39 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_124, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_51 = [squeeze__38, one_hot_16]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_51, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_125 = paddle.matmul(concat_37, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__139 = paddle._C_ops.add(matmul_125, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_126 = paddle.matmul(multiply__50, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__140 = paddle._C_ops.add(add__139, matmul_126)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__141 = paddle._C_ops.add(add__140, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_16 = paddle._C_ops.split_with_num(add__141, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_72 = split_with_num_16[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__48 = paddle._C_ops.sigmoid(slice_72)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_73 = split_with_num_16[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__49 = paddle._C_ops.sigmoid(slice_73)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_74 = split_with_num_16[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__50 = paddle._C_ops.sigmoid(slice_74)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__51 = paddle._C_ops.multiply(sigmoid__49, add__135)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_75 = split_with_num_16[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__33 = paddle._C_ops.tanh(slice_75)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__52 = paddle._C_ops.multiply(sigmoid__48, tanh__33)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__142 = paddle._C_ops.add(multiply__51, multiply__52)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_16 = paddle._C_ops.tanh(add__142)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__53 = paddle._C_ops.multiply(sigmoid__50, tanh_16)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_127 = paddle.matmul(multiply__53, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__143 = paddle._C_ops.add(matmul_127, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_34, unsqueeze_35 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__143, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x16x38xf32, -1x1x38xf32]) <- (-1x16x38xf32, -1x1x38xf32)
        combine_52 = [concat_36, unsqueeze_34]

        # pd_op.concat: (-1x17x38xf32) <- ([-1x16x38xf32, -1x1x38xf32], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_52, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_16 = paddle._C_ops.argmax(add__143, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_17 = paddle._C_ops.one_hot(argmax_16 % paddle.cast(constant_21, argmax_16.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_128 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_129 = paddle.matmul(multiply__53, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__144 = paddle._C_ops.add(matmul_129, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__40, unsqueeze__41 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__144, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__145 = paddle._C_ops.add(matmul_128, unsqueeze__40)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__34 = paddle._C_ops.tanh(add__145)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_130 = paddle.matmul(tanh__34, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__17 = paddle._C_ops.softmax(matmul_130, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_25 = paddle._C_ops.transpose(softmax__17, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_131 = paddle.matmul(transpose_25, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__40, squeeze__41 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_131, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_53 = [squeeze__40, one_hot_17]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_39 = paddle._C_ops.concat(combine_53, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_132 = paddle.matmul(concat_39, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__146 = paddle._C_ops.add(matmul_132, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_133 = paddle.matmul(multiply__53, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__147 = paddle._C_ops.add(add__146, matmul_133)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__148 = paddle._C_ops.add(add__147, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_17 = paddle._C_ops.split_with_num(add__148, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_76 = split_with_num_17[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__51 = paddle._C_ops.sigmoid(slice_76)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_77 = split_with_num_17[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__52 = paddle._C_ops.sigmoid(slice_77)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_78 = split_with_num_17[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__53 = paddle._C_ops.sigmoid(slice_78)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__54 = paddle._C_ops.multiply(sigmoid__52, add__142)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_79 = split_with_num_17[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__35 = paddle._C_ops.tanh(slice_79)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__55 = paddle._C_ops.multiply(sigmoid__51, tanh__35)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__149 = paddle._C_ops.add(multiply__54, multiply__55)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_17 = paddle._C_ops.tanh(add__149)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__56 = paddle._C_ops.multiply(sigmoid__53, tanh_17)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_134 = paddle.matmul(multiply__56, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__150 = paddle._C_ops.add(matmul_134, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_36, unsqueeze_37 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__150, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x17x38xf32, -1x1x38xf32]) <- (-1x17x38xf32, -1x1x38xf32)
        combine_54 = [concat_38, unsqueeze_36]

        # pd_op.concat: (-1x18x38xf32) <- ([-1x17x38xf32, -1x1x38xf32], 1xi32)
        concat_40 = paddle._C_ops.concat(combine_54, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_17 = paddle._C_ops.argmax(add__150, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_18 = paddle._C_ops.one_hot(argmax_17 % paddle.cast(constant_21, argmax_17.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_135 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_136 = paddle.matmul(multiply__56, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__151 = paddle._C_ops.add(matmul_136, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__42, unsqueeze__43 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__151, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__152 = paddle._C_ops.add(matmul_135, unsqueeze__42)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__36 = paddle._C_ops.tanh(add__152)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_137 = paddle.matmul(tanh__36, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__18 = paddle._C_ops.softmax(matmul_137, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_26 = paddle._C_ops.transpose(softmax__18, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_138 = paddle.matmul(transpose_26, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__42, squeeze__43 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_138, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_55 = [squeeze__42, one_hot_18]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_41 = paddle._C_ops.concat(combine_55, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_139 = paddle.matmul(concat_41, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__153 = paddle._C_ops.add(matmul_139, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_140 = paddle.matmul(multiply__56, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__154 = paddle._C_ops.add(add__153, matmul_140)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__155 = paddle._C_ops.add(add__154, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_18 = paddle._C_ops.split_with_num(add__155, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_80 = split_with_num_18[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__54 = paddle._C_ops.sigmoid(slice_80)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_81 = split_with_num_18[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__55 = paddle._C_ops.sigmoid(slice_81)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_82 = split_with_num_18[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__56 = paddle._C_ops.sigmoid(slice_82)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__57 = paddle._C_ops.multiply(sigmoid__55, add__149)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_83 = split_with_num_18[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__37 = paddle._C_ops.tanh(slice_83)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__58 = paddle._C_ops.multiply(sigmoid__54, tanh__37)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__156 = paddle._C_ops.add(multiply__57, multiply__58)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_18 = paddle._C_ops.tanh(add__156)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__59 = paddle._C_ops.multiply(sigmoid__56, tanh_18)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_141 = paddle.matmul(multiply__59, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__157 = paddle._C_ops.add(matmul_141, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_38, unsqueeze_39 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__157, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x18x38xf32, -1x1x38xf32]) <- (-1x18x38xf32, -1x1x38xf32)
        combine_56 = [concat_40, unsqueeze_38]

        # pd_op.concat: (-1x19x38xf32) <- ([-1x18x38xf32, -1x1x38xf32], 1xi32)
        concat_42 = paddle._C_ops.concat(combine_56, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_18 = paddle._C_ops.argmax(add__157, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_19 = paddle._C_ops.one_hot(argmax_18 % paddle.cast(constant_21, argmax_18.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_142 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_143 = paddle.matmul(multiply__59, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__158 = paddle._C_ops.add(matmul_143, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__44, unsqueeze__45 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__158, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__159 = paddle._C_ops.add(matmul_142, unsqueeze__44)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__38 = paddle._C_ops.tanh(add__159)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_144 = paddle.matmul(tanh__38, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__19 = paddle._C_ops.softmax(matmul_144, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_27 = paddle._C_ops.transpose(softmax__19, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_145 = paddle.matmul(transpose_27, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__44, squeeze__45 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_145, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_57 = [squeeze__44, one_hot_19]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_43 = paddle._C_ops.concat(combine_57, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_146 = paddle.matmul(concat_43, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__160 = paddle._C_ops.add(matmul_146, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_147 = paddle.matmul(multiply__59, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__161 = paddle._C_ops.add(add__160, matmul_147)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__162 = paddle._C_ops.add(add__161, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_19 = paddle._C_ops.split_with_num(add__162, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_84 = split_with_num_19[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__57 = paddle._C_ops.sigmoid(slice_84)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_85 = split_with_num_19[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__58 = paddle._C_ops.sigmoid(slice_85)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_86 = split_with_num_19[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__59 = paddle._C_ops.sigmoid(slice_86)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__60 = paddle._C_ops.multiply(sigmoid__58, add__156)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_87 = split_with_num_19[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__39 = paddle._C_ops.tanh(slice_87)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__61 = paddle._C_ops.multiply(sigmoid__57, tanh__39)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__163 = paddle._C_ops.add(multiply__60, multiply__61)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_19 = paddle._C_ops.tanh(add__163)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__62 = paddle._C_ops.multiply(sigmoid__59, tanh_19)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_148 = paddle.matmul(multiply__62, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__164 = paddle._C_ops.add(matmul_148, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_40, unsqueeze_41 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__164, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x19x38xf32, -1x1x38xf32]) <- (-1x19x38xf32, -1x1x38xf32)
        combine_58 = [concat_42, unsqueeze_40]

        # pd_op.concat: (-1x20x38xf32) <- ([-1x19x38xf32, -1x1x38xf32], 1xi32)
        concat_44 = paddle._C_ops.concat(combine_58, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_19 = paddle._C_ops.argmax(add__164, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_20 = paddle._C_ops.one_hot(argmax_19 % paddle.cast(constant_21, argmax_19.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_149 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_150 = paddle.matmul(multiply__62, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__165 = paddle._C_ops.add(matmul_150, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__46, unsqueeze__47 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__165, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__166 = paddle._C_ops.add(matmul_149, unsqueeze__46)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__40 = paddle._C_ops.tanh(add__166)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_151 = paddle.matmul(tanh__40, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__20 = paddle._C_ops.softmax(matmul_151, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_28 = paddle._C_ops.transpose(softmax__20, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_152 = paddle.matmul(transpose_28, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__46, squeeze__47 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_152, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_59 = [squeeze__46, one_hot_20]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_45 = paddle._C_ops.concat(combine_59, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_153 = paddle.matmul(concat_45, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__167 = paddle._C_ops.add(matmul_153, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_154 = paddle.matmul(multiply__62, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__168 = paddle._C_ops.add(add__167, matmul_154)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__169 = paddle._C_ops.add(add__168, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_20 = paddle._C_ops.split_with_num(add__169, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_88 = split_with_num_20[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__60 = paddle._C_ops.sigmoid(slice_88)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_89 = split_with_num_20[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__61 = paddle._C_ops.sigmoid(slice_89)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_90 = split_with_num_20[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__62 = paddle._C_ops.sigmoid(slice_90)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__63 = paddle._C_ops.multiply(sigmoid__61, add__163)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_91 = split_with_num_20[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__41 = paddle._C_ops.tanh(slice_91)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__64 = paddle._C_ops.multiply(sigmoid__60, tanh__41)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__170 = paddle._C_ops.add(multiply__63, multiply__64)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_20 = paddle._C_ops.tanh(add__170)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__65 = paddle._C_ops.multiply(sigmoid__62, tanh_20)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_155 = paddle.matmul(multiply__65, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__171 = paddle._C_ops.add(matmul_155, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_42, unsqueeze_43 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__171, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x20x38xf32, -1x1x38xf32]) <- (-1x20x38xf32, -1x1x38xf32)
        combine_60 = [concat_44, unsqueeze_42]

        # pd_op.concat: (-1x21x38xf32) <- ([-1x20x38xf32, -1x1x38xf32], 1xi32)
        concat_46 = paddle._C_ops.concat(combine_60, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_20 = paddle._C_ops.argmax(add__171, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_21 = paddle._C_ops.one_hot(argmax_20 % paddle.cast(constant_21, argmax_20.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_156 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_157 = paddle.matmul(multiply__65, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__172 = paddle._C_ops.add(matmul_157, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__48, unsqueeze__49 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__172, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__173 = paddle._C_ops.add(matmul_156, unsqueeze__48)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__42 = paddle._C_ops.tanh(add__173)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_158 = paddle.matmul(tanh__42, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__21 = paddle._C_ops.softmax(matmul_158, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_29 = paddle._C_ops.transpose(softmax__21, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_159 = paddle.matmul(transpose_29, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__48, squeeze__49 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_159, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_61 = [squeeze__48, one_hot_21]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_47 = paddle._C_ops.concat(combine_61, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_160 = paddle.matmul(concat_47, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__174 = paddle._C_ops.add(matmul_160, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_161 = paddle.matmul(multiply__65, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__175 = paddle._C_ops.add(add__174, matmul_161)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__176 = paddle._C_ops.add(add__175, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_21 = paddle._C_ops.split_with_num(add__176, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_92 = split_with_num_21[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__63 = paddle._C_ops.sigmoid(slice_92)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_93 = split_with_num_21[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__64 = paddle._C_ops.sigmoid(slice_93)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_94 = split_with_num_21[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__65 = paddle._C_ops.sigmoid(slice_94)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__66 = paddle._C_ops.multiply(sigmoid__64, add__170)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_95 = split_with_num_21[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__43 = paddle._C_ops.tanh(slice_95)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__67 = paddle._C_ops.multiply(sigmoid__63, tanh__43)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__177 = paddle._C_ops.add(multiply__66, multiply__67)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_21 = paddle._C_ops.tanh(add__177)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__68 = paddle._C_ops.multiply(sigmoid__65, tanh_21)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_162 = paddle.matmul(multiply__68, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__178 = paddle._C_ops.add(matmul_162, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_44, unsqueeze_45 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__178, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x21x38xf32, -1x1x38xf32]) <- (-1x21x38xf32, -1x1x38xf32)
        combine_62 = [concat_46, unsqueeze_44]

        # pd_op.concat: (-1x22x38xf32) <- ([-1x21x38xf32, -1x1x38xf32], 1xi32)
        concat_48 = paddle._C_ops.concat(combine_62, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_21 = paddle._C_ops.argmax(add__178, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_22 = paddle._C_ops.one_hot(argmax_21 % paddle.cast(constant_21, argmax_21.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_163 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_164 = paddle.matmul(multiply__68, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__179 = paddle._C_ops.add(matmul_164, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__50, unsqueeze__51 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__179, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__180 = paddle._C_ops.add(matmul_163, unsqueeze__50)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__44 = paddle._C_ops.tanh(add__180)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_165 = paddle.matmul(tanh__44, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__22 = paddle._C_ops.softmax(matmul_165, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_30 = paddle._C_ops.transpose(softmax__22, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_166 = paddle.matmul(transpose_30, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__50, squeeze__51 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_166, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_63 = [squeeze__50, one_hot_22]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_49 = paddle._C_ops.concat(combine_63, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_167 = paddle.matmul(concat_49, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__181 = paddle._C_ops.add(matmul_167, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_168 = paddle.matmul(multiply__68, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__182 = paddle._C_ops.add(add__181, matmul_168)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__183 = paddle._C_ops.add(add__182, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_22 = paddle._C_ops.split_with_num(add__183, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_96 = split_with_num_22[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__66 = paddle._C_ops.sigmoid(slice_96)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_97 = split_with_num_22[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__67 = paddle._C_ops.sigmoid(slice_97)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_98 = split_with_num_22[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__68 = paddle._C_ops.sigmoid(slice_98)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__69 = paddle._C_ops.multiply(sigmoid__67, add__177)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_99 = split_with_num_22[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__45 = paddle._C_ops.tanh(slice_99)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__70 = paddle._C_ops.multiply(sigmoid__66, tanh__45)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__184 = paddle._C_ops.add(multiply__69, multiply__70)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_22 = paddle._C_ops.tanh(add__184)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__71 = paddle._C_ops.multiply(sigmoid__68, tanh_22)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_169 = paddle.matmul(multiply__71, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__185 = paddle._C_ops.add(matmul_169, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_46, unsqueeze_47 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__185, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x22x38xf32, -1x1x38xf32]) <- (-1x22x38xf32, -1x1x38xf32)
        combine_64 = [concat_48, unsqueeze_46]

        # pd_op.concat: (-1x23x38xf32) <- ([-1x22x38xf32, -1x1x38xf32], 1xi32)
        concat_50 = paddle._C_ops.concat(combine_64, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_22 = paddle._C_ops.argmax(add__185, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_23 = paddle._C_ops.one_hot(argmax_22 % paddle.cast(constant_21, argmax_22.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_170 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_171 = paddle.matmul(multiply__71, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__186 = paddle._C_ops.add(matmul_171, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__52, unsqueeze__53 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__186, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__187 = paddle._C_ops.add(matmul_170, unsqueeze__52)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__46 = paddle._C_ops.tanh(add__187)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_172 = paddle.matmul(tanh__46, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__23 = paddle._C_ops.softmax(matmul_172, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_31 = paddle._C_ops.transpose(softmax__23, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_173 = paddle.matmul(transpose_31, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__52, squeeze__53 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_173, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_65 = [squeeze__52, one_hot_23]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_51 = paddle._C_ops.concat(combine_65, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_174 = paddle.matmul(concat_51, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__188 = paddle._C_ops.add(matmul_174, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_175 = paddle.matmul(multiply__71, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__189 = paddle._C_ops.add(add__188, matmul_175)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__190 = paddle._C_ops.add(add__189, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_23 = paddle._C_ops.split_with_num(add__190, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_100 = split_with_num_23[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__69 = paddle._C_ops.sigmoid(slice_100)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_101 = split_with_num_23[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__70 = paddle._C_ops.sigmoid(slice_101)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_102 = split_with_num_23[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__71 = paddle._C_ops.sigmoid(slice_102)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__72 = paddle._C_ops.multiply(sigmoid__70, add__184)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_103 = split_with_num_23[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__47 = paddle._C_ops.tanh(slice_103)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__73 = paddle._C_ops.multiply(sigmoid__69, tanh__47)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__191 = paddle._C_ops.add(multiply__72, multiply__73)

        # pd_op.tanh: (-1x256xf32) <- (-1x256xf32)
        tanh_23 = paddle._C_ops.tanh(add__191)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__74 = paddle._C_ops.multiply(sigmoid__71, tanh_23)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_176 = paddle.matmul(multiply__74, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__192 = paddle._C_ops.add(matmul_176, parameter_319)

        # pd_op.unsqueeze: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze_48, unsqueeze_49 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__192, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x23x38xf32, -1x1x38xf32]) <- (-1x23x38xf32, -1x1x38xf32)
        combine_66 = [concat_50, unsqueeze_48]

        # pd_op.concat: (-1x24x38xf32) <- ([-1x23x38xf32, -1x1x38xf32], 1xi32)
        concat_52 = paddle._C_ops.concat(combine_66, constant_8)

        # pd_op.argmax: (-1xi64) <- (-1x38xf32, 1xi64)
        argmax_23 = paddle._C_ops.argmax(add__192, constant_22, False, False, paddle.int64)

        # pd_op.one_hot: (-1x38xf32) <- (-1xi64, 1xi32)
        one_hot_24 = paddle._C_ops.one_hot(argmax_23 % paddle.cast(constant_21, argmax_23.dtype), constant_21)

        # pd_op.matmul: (-1x26x256xf32) <- (-1x26x512xf32, 512x256xf32)
        matmul_177 = paddle.matmul(transpose_6, parameter_310, transpose_x=False, transpose_y=False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_178 = paddle.matmul(multiply__74, parameter_311, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__193 = paddle._C_ops.add(matmul_178, parameter_312)

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__54, unsqueeze__55 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__193, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x26x256xf32) <- (-1x26x256xf32, -1x1x256xf32)
        add__194 = paddle._C_ops.add(matmul_177, unsqueeze__54)

        # pd_op.tanh_: (-1x26x256xf32) <- (-1x26x256xf32)
        tanh__48 = paddle._C_ops.tanh(add__194)

        # pd_op.matmul: (-1x26x1xf32) <- (-1x26x256xf32, 256x1xf32)
        matmul_179 = paddle.matmul(tanh__48, parameter_313, transpose_x=False, transpose_y=False)

        # pd_op.softmax_: (-1x26x1xf32) <- (-1x26x1xf32)
        softmax__24 = paddle._C_ops.softmax(matmul_179, 1)

        # pd_op.transpose: (-1x1x26xf32) <- (-1x26x1xf32)
        transpose_32 = paddle._C_ops.transpose(softmax__24, [0, 2, 1])

        # pd_op.matmul: (-1x1x512xf32) <- (-1x1x26xf32, -1x26x512xf32)
        matmul_180 = paddle.matmul(transpose_32, transpose_6, transpose_x=False, transpose_y=False)

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x1x512xf32, 1xi64)
        squeeze__54, squeeze__55 = (lambda x, f: f(x))(paddle._C_ops.squeeze(matmul_180, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x512xf32, -1x38xf32]) <- (-1x512xf32, -1x38xf32)
        combine_67 = [squeeze__54, one_hot_24]

        # pd_op.concat: (-1x550xf32) <- ([-1x512xf32, -1x38xf32], 1xi32)
        concat_53 = paddle._C_ops.concat(combine_67, constant_8)

        # pd_op.matmul: (-1x1024xf32) <- (-1x550xf32, 1024x550xf32)
        matmul_181 = paddle.matmul(concat_53, parameter_314, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__195 = paddle._C_ops.add(matmul_181, parameter_315)

        # pd_op.matmul: (-1x1024xf32) <- (-1x256xf32, 1024x256xf32)
        matmul_182 = paddle.matmul(multiply__74, parameter_316, transpose_x=False, transpose_y=True)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, -1x1024xf32)
        add__196 = paddle._C_ops.add(add__195, matmul_182)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__197 = paddle._C_ops.add(add__196, parameter_317)

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_24 = paddle._C_ops.split_with_num(add__197, 4, constant_8)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_104 = split_with_num_24[0]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__72 = paddle._C_ops.sigmoid(slice_104)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_105 = split_with_num_24[1]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__73 = paddle._C_ops.sigmoid(slice_105)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_106 = split_with_num_24[3]

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__74 = paddle._C_ops.sigmoid(slice_106)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__75 = paddle._C_ops.multiply(sigmoid__73, add__191)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32, -1x256xf32])
        slice_107 = split_with_num_24[2]

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__49 = paddle._C_ops.tanh(slice_107)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__76 = paddle._C_ops.multiply(sigmoid__72, tanh__49)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__198 = paddle._C_ops.add(multiply__75, multiply__76)

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__50 = paddle._C_ops.tanh(add__198)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__77 = paddle._C_ops.multiply(sigmoid__74, tanh__50)

        # pd_op.matmul: (-1x38xf32) <- (-1x256xf32, 256x38xf32)
        matmul_183 = paddle.matmul(multiply__77, parameter_318, transpose_x=False, transpose_y=False)

        # pd_op.add_: (-1x38xf32) <- (-1x38xf32, 38xf32)
        add__199 = paddle._C_ops.add(matmul_183, parameter_319)

        # pd_op.unsqueeze_: (-1x1x38xf32, None) <- (-1x38xf32, 1xi64)
        unsqueeze__56, unsqueeze__57 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add__199, constant_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x24x38xf32, -1x1x38xf32]) <- (-1x24x38xf32, -1x1x38xf32)
        combine_68 = [concat_52, unsqueeze__56]

        # pd_op.concat: (-1x25x38xf32) <- ([-1x24x38xf32, -1x1x38xf32], 1xi32)
        concat_54 = paddle._C_ops.concat(combine_68, constant_8)

        # pd_op.softmax_: (-1x25x38xf32) <- (-1x25x38xf32)
        softmax__25 = paddle._C_ops.softmax(concat_54, 2)

        # pd_op.scale_: (-1x38xf32) <- (-1x38xf32, 1xf32)
        scale__1 = paddle._C_ops.scale(add__24, constant_12, float('0'), True)

        # pd_op.scale_: (-1x25x38xf32) <- (-1x25x38xf32, 1xf32)
        scale__2 = paddle._C_ops.scale(softmax__25, constant_12, float('0'), True)
        return scale__1, scale__2



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

    def forward(self, constant_22, constant_21, parameter_309, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, parameter_33, constant_11, constant_10, constant_9, parameter_32, parameter_31, constant_8, parameter_30, parameter_29, constant_7, constant_6, parameter_28, constant_5, parameter_27, parameter_26, constant_12, constant_4, parameter_24, parameter_25, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_21, parameter_22, parameter_23, parameter_34, parameter_35, parameter_36, parameter_40, parameter_37, parameter_39, parameter_38, parameter_41, parameter_45, parameter_42, parameter_44, parameter_43, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_55, parameter_52, parameter_54, parameter_53, parameter_56, parameter_60, parameter_57, parameter_59, parameter_58, parameter_61, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_70, parameter_67, parameter_69, parameter_68, parameter_71, parameter_75, parameter_72, parameter_74, parameter_73, parameter_76, parameter_80, parameter_77, parameter_79, parameter_78, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_135, parameter_132, parameter_134, parameter_133, parameter_136, parameter_140, parameter_137, parameter_139, parameter_138, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_160, parameter_157, parameter_159, parameter_158, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_175, parameter_172, parameter_174, parameter_173, parameter_176, parameter_180, parameter_177, parameter_179, parameter_178, parameter_181, parameter_185, parameter_182, parameter_184, parameter_183, parameter_186, parameter_190, parameter_187, parameter_189, parameter_188, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_230, parameter_227, parameter_229, parameter_228, parameter_231, parameter_235, parameter_232, parameter_234, parameter_233, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_260, parameter_257, parameter_259, parameter_258, parameter_261, parameter_265, parameter_262, parameter_264, parameter_263, parameter_266, parameter_270, parameter_267, parameter_269, parameter_268, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_280, parameter_277, parameter_279, parameter_278, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_305, parameter_302, parameter_304, parameter_303, parameter_306, parameter_307, parameter_308, parameter_310, parameter_311, parameter_312, parameter_313, parameter_314, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, feed_0):
        return self.builtin_module_2299_0_0(constant_22, constant_21, parameter_309, constant_20, constant_19, constant_18, constant_17, constant_16, constant_15, constant_14, constant_13, parameter_33, constant_11, constant_10, constant_9, parameter_32, parameter_31, constant_8, parameter_30, parameter_29, constant_7, constant_6, parameter_28, constant_5, parameter_27, parameter_26, constant_12, constant_4, parameter_24, parameter_25, constant_3, constant_2, constant_1, constant_0, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_21, parameter_22, parameter_23, parameter_34, parameter_35, parameter_36, parameter_40, parameter_37, parameter_39, parameter_38, parameter_41, parameter_45, parameter_42, parameter_44, parameter_43, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_55, parameter_52, parameter_54, parameter_53, parameter_56, parameter_60, parameter_57, parameter_59, parameter_58, parameter_61, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_70, parameter_67, parameter_69, parameter_68, parameter_71, parameter_75, parameter_72, parameter_74, parameter_73, parameter_76, parameter_80, parameter_77, parameter_79, parameter_78, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_135, parameter_132, parameter_134, parameter_133, parameter_136, parameter_140, parameter_137, parameter_139, parameter_138, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_160, parameter_157, parameter_159, parameter_158, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_175, parameter_172, parameter_174, parameter_173, parameter_176, parameter_180, parameter_177, parameter_179, parameter_178, parameter_181, parameter_185, parameter_182, parameter_184, parameter_183, parameter_186, parameter_190, parameter_187, parameter_189, parameter_188, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_230, parameter_227, parameter_229, parameter_228, parameter_231, parameter_235, parameter_232, parameter_234, parameter_233, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_260, parameter_257, parameter_259, parameter_258, parameter_261, parameter_265, parameter_262, parameter_264, parameter_263, parameter_266, parameter_270, parameter_267, parameter_269, parameter_268, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_280, parameter_277, parameter_279, parameter_278, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_305, parameter_302, parameter_304, parameter_303, parameter_306, parameter_307, parameter_308, parameter_310, parameter_311, parameter_312, parameter_313, parameter_314, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2299_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # constant_22
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # constant_21
            paddle.to_tensor([38], dtype='int32').reshape([1]),
            # parameter_309
            paddle.to_tensor([256], dtype='int32').reshape([]),
            # constant_20
            paddle.to_tensor([13312], dtype='int32').reshape([1]),
            # constant_19
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
            # constant_18
            paddle.to_tensor([26], dtype='int32').reshape([1]),
            # constant_17
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # constant_16
            paddle.to_tensor([512], dtype='int32').reshape([1]),
            # constant_15
            paddle.to_tensor([-1, 32, 100, 2], dtype='int64').reshape([4]),
            # constant_14
            paddle.to_tensor([-1, 3, 2], dtype='int64').reshape([3]),
            # constant_13
            paddle.to_tensor([40], dtype='int32').reshape([1]),
            # parameter_33
            paddle.uniform([3200, 1], dtype='float64', min=0, max=0.5),
            # constant_11
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # constant_10
            paddle.to_tensor([1, 20, 1], dtype='int64').reshape([3]),
            # constant_9
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # parameter_32
            paddle.uniform([1, 23], dtype='float64', min=0, max=0.5),
            # parameter_31
            paddle.uniform([2, 3], dtype='float64', min=0, max=0.5),
            # constant_8
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            # parameter_30
            paddle.uniform([20, 1], dtype='float64', min=0, max=0.5),
            # parameter_29
            paddle.uniform([], dtype='float64', min=0, max=0.5),
            # constant_7
            paddle.to_tensor([20, 1, 2], dtype='int64').reshape([3]),
            # constant_6
            paddle.to_tensor([1, 20, 2], dtype='int64').reshape([3]),
            # parameter_28
            paddle.uniform([20, 20], dtype='float64', min=0, max=0.5),
            # constant_5
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
            # parameter_27
            paddle.uniform([32], dtype='float64', min=0, max=0.5),
            # parameter_26
            paddle.uniform([100], dtype='float64', min=0, max=0.5),
            # constant_12
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # constant_4
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            # parameter_24
            paddle.uniform([10], dtype='float64', min=0, max=0.5),
            # parameter_25
            paddle.uniform([10], dtype='float64', min=0, max=0.5),
            # constant_3
            paddle.to_tensor([-1, 20, 2], dtype='int64').reshape([3]),
            # constant_2
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # constant_1
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            # constant_0
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            # parameter_0
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256, 40], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([40, 6], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([6], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_151
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
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512, 512, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([512, 512, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([512, 512, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([512, 512, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([13312, 38], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([256, 1], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([1024, 550], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([256, 38], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # constant_22
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_21
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_309
            paddle.static.InputSpec(shape=[], dtype='int32'),
            # constant_20
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_19
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_18
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_17
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_16
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # constant_15
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # constant_14
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_13
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_33
            paddle.static.InputSpec(shape=[3200, 1], dtype='float64'),
            # constant_11
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_10
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_9
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_32
            paddle.static.InputSpec(shape=[1, 23], dtype='float64'),
            # parameter_31
            paddle.static.InputSpec(shape=[2, 3], dtype='float64'),
            # constant_8
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_30
            paddle.static.InputSpec(shape=[20, 1], dtype='float64'),
            # parameter_29
            paddle.static.InputSpec(shape=[], dtype='float64'),
            # constant_7
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_6
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # parameter_28
            paddle.static.InputSpec(shape=[20, 20], dtype='float64'),
            # constant_5
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_27
            paddle.static.InputSpec(shape=[32], dtype='float64'),
            # parameter_26
            paddle.static.InputSpec(shape=[100], dtype='float64'),
            # constant_12
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # constant_4
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            # parameter_24
            paddle.static.InputSpec(shape=[10], dtype='float64'),
            # parameter_25
            paddle.static.InputSpec(shape=[10], dtype='float64'),
            # constant_3
            paddle.static.InputSpec(shape=[3], dtype='int64'),
            # constant_2
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # constant_1
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # constant_0
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # parameter_0
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256, 40], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[40, 6], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_151
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
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[512, 512, 2, 2], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[512, 512, 2, 2], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[512, 512, 2, 2], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[512, 512, 2, 2], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[13312, 38], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[38], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[256, 1], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[1024, 550], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[256, 38], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[38], dtype='float32'),
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