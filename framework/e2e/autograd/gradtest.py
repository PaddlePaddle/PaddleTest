"""
grad test
"""

import os
import sys

curPath = os.path.abspath(os.path.dirname("utils"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import copy
from inspect import isclass
import numpy as np
import paddle
import jax
import jax.numpy as jnp
from tool import FrontAPIBase, NPDTYPE
from utils.logger import Logger

logger = Logger("JaxTest")


class JaxTest(object):
    """
    high order autograd test
    compare paddle with jax
    """

    def __init__(self, api):
        """
        initialize
        """
        self.seed = 33
        self.paddle_api = api[0]
        self.jax_api = api[1]
        self.grad_order = 5
        self.paddle_inputs = dict()
        self.paddle_params = dict()
        self.jax_inputs = dict()
        self.jax_params = dict()
        self.ignore_var = []
        self.init_ad = None  # float, int, "random"
        self.logger = logger
        self.logger.get_log().info("start high order autograd test !!!")

    def _set_func(self, api):
        """
        set api
        """
        if not isclass(api):
            return api
        else:
            obj = FrontAPIBase(api)
            return obj.encapsulation

    def run(self, data0, data1):
        """
        run
        """
        self.logger.get_log().info(
            "compare [paddle]{} and [jax]{}".format(str(self.paddle_api.__name__), str(self.jax_api.__name__))
        )
        self._set_inputs(data0, data1)
        self.logger.get_log().info("inputs processing succeeded !!!")

        paddle_func = self._set_func(self.paddle_api)
        paddle_backward_out = self._paddle_backward_ad(paddle_func)
        self.logger.get_log().info("paddle backward ad rslt is:\n {}".format(paddle_backward_out))
        paddle_forward_out = self._paddle_forward_ad(paddle_func)
        self.logger.get_log().info("paddle forward ad rslt is:\n {}".format(paddle_forward_out))

        jax_func = self._set_func(self.jax_api)
        jax_backward_out = self._jax_backward_ad(func=jax_func)
        self.logger.get_log().info("jax backward ad rslt is:\n {}".format(jax_backward_out))
        jax_forward_out = self._jax_forward_ad(func=jax_func)
        self.logger.get_log().info("jax forward ad rslt is:\n {}".format(jax_forward_out))

        self._compare(paddle_backward_out, paddle_forward_out)
        self._compare(paddle_backward_out, jax_backward_out, label="backward")
        self._compare(paddle_forward_out, jax_forward_out, label="forward")

    def _paddle_backward_ad(self, func):
        """
        _paddle_backward_ad
        """
        self.logger.get_log().info("start calculate paddle backward ad")
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()
        paddle.seed(self.seed)
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=mp, startup_program=sp):
                # 1.处理求导的输入
                inputs = {}
                for k, v in self.paddle_inputs.items():
                    inputs[k] = paddle.static.data(name=k, shape=v.shape, dtype=v.dtype)

                # 2.处理非求导输入
                params = copy.deepcopy(self.paddle_params)
                for k, v in params.items():
                    if isinstance(v, np.ndarray):
                        params[k] = paddle.static.data(name=k, shape=v.shape, dtype=v.dtype)

                # 3. 前向
                output = func(**inputs, **params)

                # 4. 反向
                init_feed = {}
                for k, v in inputs.items():  # todo:多输入的微分计算
                    # grad = paddle.static.gradients(output, inputs[k])
                    init_array = paddle.static.data(name=k + "_init", shape=v.shape, dtype=v.dtype)
                    init_feed[k + "_init"] = self._generate_init_ad(
                        v.shape, init=self.init_ad, dtype=NPDTYPE[str(v.dtype)]
                    )
                    grad1 = paddle.incubate.autograd.grad(output, v, grad_outputs=init_array)
                    grad2 = paddle.incubate.autograd.grad(grad1, v, grad_outputs=init_array)
                    grad3 = paddle.incubate.autograd.grad(grad2, v, grad_outputs=init_array)
                    grad4 = paddle.incubate.autograd.grad(grad3, v, grad_outputs=init_array)
                    grad5 = paddle.incubate.autograd.grad(grad4, v, grad_outputs=init_array)
                paddle.incubate.autograd.prim2orig()

        # 5. run
        fetch_list = [grad1, grad2, grad3, grad4, grad5]
        # fetch_list = [output, grad]
        exe = paddle.static.Executor()
        exe.run(sp)
        return exe.run(mp, feed={**self.paddle_inputs, **self.paddle_params, **init_feed}, fetch_list=fetch_list)

    def _paddle_forward_ad(self, func):
        """
        paddle_forward_ad
        """
        self.logger.get_log().info("start calculate paddle forward ad")
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()
        paddle.seed(self.seed)
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=mp, startup_program=sp):
                # 1.处理求导的输入
                inputs = {}
                for k, v in self.paddle_inputs.items():
                    inputs[k] = paddle.static.data(name=k, shape=v.shape, dtype=v.dtype)

                # 2.处理非求导输入
                params = copy.deepcopy(self.paddle_params)
                for k, v in params.items():
                    if isinstance(v, np.ndarray):
                        params[k] = paddle.static.data(name=k, shape=v.shape, dtype=v.dtype)

                # 3. 前向计算
                output = func(**inputs, **params)  # TODO：多输出处理

                # 4. 前向自动微分
                init_feed = {}
                for k, v in inputs.items():  # todo:多输入的微分计算
                    init_array = paddle.static.data(name=k + "_init", shape=v.shape, dtype=v.dtype)
                    init_feed[k + "_init"] = self._generate_init_ad(
                        v.shape, init=self.init_ad, dtype=NPDTYPE[str(v.dtype)]
                    )
                    grad1 = paddle.incubate.autograd.forward_grad(output, v, grad_inputs=init_array)
                    grad2 = paddle.incubate.autograd.forward_grad(grad1, v, grad_inputs=init_array)
                    grad3 = paddle.incubate.autograd.forward_grad(grad2, v, grad_inputs=init_array)
                    grad4 = paddle.incubate.autograd.forward_grad(grad3, v, grad_inputs=init_array)
                    grad5 = paddle.incubate.autograd.forward_grad(grad4, v, grad_inputs=init_array)
                paddle.incubate.autograd.prim2orig()

        # 5. run
        fetch_list = [grad1, grad2, grad3, grad4, grad5]
        # fetch_list = [output, grad]
        exe = paddle.static.Executor()
        exe.run(sp)
        return exe.run(mp, feed={**self.paddle_inputs, **self.paddle_params, **init_feed}, fetch_list=fetch_list)

    def _jax_backward_ad(self, func):
        """
        _jax_backward_ad
        """
        for k, v in self.jax_inputs.items():
            init_array = self._generate_init_ad(v.shape, init=self.init_ad, dtype=v.dtype)
            grad_fn1 = lambda x: jax.vjp(func, x)[1](init_array)[0]
            grad_fn2 = lambda x: jax.vjp(grad_fn1, x)[1](init_array)[0]
            grad_fn3 = lambda x: jax.vjp(grad_fn2, x)[1](init_array)[0]
            grad_fn4 = lambda x: jax.vjp(grad_fn3, x)[1](init_array)[0]
            grad_fn5 = lambda x: jax.vjp(grad_fn4, x)[1](init_array)[0]
            grad1, grad2, grad3, grad4, grad5 = grad_fn1(v), grad_fn2(v), grad_fn3(v), grad_fn4(v), grad_fn5(v)
            # todo: 多输入的反向
            return [grad1.to_py(), grad2.to_py(), grad3.to_py(), grad4.to_py(), grad5.to_py()]

    def _jax_forward_ad(self, func):
        """
        _jax_forward_ad
        """
        for k, v in self.jax_inputs.items():
            init_array = self._generate_init_ad(v.shape, init=self.init_ad, dtype=v.dtype)
            grad_fn1 = lambda x: jax.jvp(func, (x,), (init_array,))[1]
            grad_fn2 = lambda x: jax.jvp(grad_fn1, (x,), (init_array,))[1]
            grad_fn3 = lambda x: jax.jvp(grad_fn2, (x,), (init_array,))[1]
            grad_fn4 = lambda x: jax.jvp(grad_fn3, (x,), (init_array,))[1]
            grad_fn5 = lambda x: jax.jvp(grad_fn4, (x,), (init_array,))[1]
            grad1, grad2, grad3, grad4, grad5 = grad_fn1(v), grad_fn2(v), grad_fn3(v), grad_fn4(v), grad_fn5(v)
            # todo: 多输入的反向
            return [grad1.to_py(), grad2.to_py(), grad3.to_py(), grad4.to_py(), grad5.to_py()]

    def _set_inputs(self, pdata, jdata):
        """
        将求导与不求导的输入分开
        """
        # for k, v in kwargs.items():
        #     if isinstance(v, np.ndarray) and k not in self.ignore_var:
        #         self.inputs[k] = kwargs[k]
        #     else:
        #         self.params[k] = kwargs[k]
        print(pdata)
        self.paddle_inputs, self.paddle_params = pdata.get("inputs"), pdata.get("params")
        self.jax_inputs, self.jax_params = jdata.get("inputs"), jdata.get("params")
        # self.logger.get_log().info("the items requiring gradient: {}".format(self.inputs))
        # self.logger.get_log().info("other parameter items: {}".format(self.params))
        self.logger.get_log().info("ins data set success !")

    def _generate_init_ad(self, shape, init, dtype):
        """
        generate initialize gradient
        """
        np.random.seed(self.seed)
        if isinstance(init, (int, float)):
            array = np.ones(shape).astype(dtype) * init
        else:
            array = np.random.rand(*shape).astype(dtype)
        return array

    def _compare(self, actual, expect, label="mixed"):
        """
        compare
        """

        # todo: 比较多项的梯度
        for k, (i, j) in enumerate(zip(actual, expect)):
            res = np.allclose(i, j, atol=1e-5, rtol=1e-8, equal_nan=True)
            if not res:
                self.logger.get_log().error("the actual {} order gradients is {}".format(k, i))
                self.logger.get_log().error("the expect {} order gradients is {}".format(k, j))
                assert False

        if label == "mixed":
            self.logger.get_log().info("compare paddle backward ad and paddle forward ad success!!!")
        elif label == "backward":
            self.logger.get_log().info("compare paddle backward ad and jax backward ad success!!! ")
        elif label == "forward":
            self.logger.get_log().info("compare paddle forward ad and jax forward ad success!!! ")


if __name__ == "__main__":
    # obj = JaxTest([paddle.sin, jnp.sin])
    # obj.init_ad = 2
    # pdata = {'inputs': {'x': np.array([-8.677039,  8.609215]).astype("float32")}, 'params': {}}
    # jdata = {'inputs': {'x': np.array([-8.677039,  8.609215]).astype("float32")}, 'params': {}}
    # obj.run(pdata, jdata)
    pass
