"""
test prim2orig
"""
import sys
import pytest
import numpy as np
import paddle
import paddle.incubate.autograd as augograd

sys.path.append("../../utils/")
from interceptor import skip_branch_not_develop


# @skip_branch_not_develop
# @pytest.mark.api_incubate_autograd_prim2orig
@pytest.mark.skip(reason="Skip new AD grad api as the api signature was changed.")
def test_prim2orig():
    """
    test01
    """
    np.random.seed(1)
    x = np.random.rand(2, 2)
    w = np.ones([2, 2])
    paddle.disable_static()
    xp = paddle.to_tensor(x, stop_gradient=False)
    wp = paddle.to_tensor(w)
    y = paddle.tanh(paddle.matmul(xp, wp))
    d_dy_dx = paddle.grad(y, xp)

    paddle.enable_static()
    augograd.enable_prim()
    # Set place and excutor
    place = paddle.CPUPlace()
    if paddle.device.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    # Build program
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        # Set input and parameter
        input_x = paddle.static.data("x", [2, 2], dtype="float64")
        input_x.stop_gradient = False
        params_w = paddle.static.data("w_p", [2, 2], dtype="float64")
        # Build network
        y = paddle.tanh(paddle.matmul(input_x, params_w))
        dy_dx, = paddle.static.gradients([y], [input_x])
        # Do prim2orig transform.
        if augograd.prim_enabled():
            augograd.prim2orig(main.block(0))
    # Run program
    exe.run(startup)
    res = exe.run(main, feed={"x": x, "w_p": w}, fetch_list=dy_dx)

    np.allclose(res[0], d_dy_dx[0])
