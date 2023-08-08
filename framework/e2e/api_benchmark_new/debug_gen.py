#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
debug gen
"""
import argparse
import yaml


class DebugCaseGen(object):
    """
    生成单个case的py脚本
    """

    def __init__(self, case, case_name):
        self.case = case
        self.case_name = case_name

        # paddle
        self.paddle = self.case.get("paddle")
        self.paddle_api = self.paddle.get("api_name")
        self.inputs = self.paddle.get("inputs")
        self.params = self.paddle.get("params")

        self.randtool = '''def _randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)
    elif dtype == "int32":
        return np.random.randint(low, high, shape).astype("int32")
    elif dtype == "int64":
        return np.random.randint(low, high, shape).astype("int64")
    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)
    elif dtype == "float16":
        return low + (high - low) * np.random.random(shape).astype("float16")
    elif dtype == "float32":
        return low + (high - low) * np.random.random(shape).astype("float32")
    elif dtype == "float64":
        return low + (high - low) * np.random.random(shape).astype("float64")
    elif dtype == "bfloat16":
        return low + (high - low) * np.random.random(shape).astype("bfloat16")
    elif dtype in ["complex", "complex64", "complex128"]:
        data = low + (high - low) * np.random.random(shape) + (low + (high - low) * np.random.random(shape)) * 1j
        return data if dtype == "complex" or "complex128" else data.astype(np.complex64)
    elif dtype == "bool":
        data = np.random.randint(0, 2, shape).astype("bool")
        return data
    else:
        assert False, "dtype is not supported"'''

        self.caculate = """
inputs = {}
for data, v in all_data.items():
    if isinstance(v, dict):
        if v.get("random"):
            inputs[data] = paddle.to_tensor(_randtool(dtype=v.get("dtype"), low=v.get("range")[0], high=v.get("range")[1], shape=v.get("shape")))
        else:
            inputs[data] = paddle.to_tensor(np.array(v.get("value")), dtype=v.get("dtype"))

def func_def(api, inputs, params):
    eval(api)(**inputs, **params)

def func_class(api, inputs, params):
    obj = eval(api)(**params)
    obj(**inputs)

all_time = []
loops = 50

for i in range(loops):
    if isclass(eval(api)):
        forward_time = timeit.timeit(lambda: func_class(api, inputs, params), number=1000)
        all_time.append(forward_time)
    else:
        forward_time = timeit.timeit(lambda: func_def(api, inputs, params), number=1000)
        all_time.append(forward_time)

head = int(loops / 5)
tail = int(loops - loops / 5)
result = (sum(sorted(all_time)[head:tail]) / (tail - head))
print(result)"""

    def py_gen(self):
        """
        生成可执行py脚本
        :return:
        """
        with open("test_{}.py".format(self.case_name), "w") as f:
            f.write(
                "#!/bin/env python3\n"
                "# -*- coding: utf-8 -*-\n"
                "# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python\n"
                '"""\n'
                "test {}\n"
                '"""\n'
                "import timeit\n"
                "from inspect import isclass\n"
                "import numpy as np\n"
                "import paddle\n"
                "import time\n"
                "\n"
                "\n".format(self.case_name)
            )
            f.write(self.randtool)
            f.write(
                "\n"
                "\n"
                "api = {}\n"
                "all_data = {}\n"
                "params = {}\n".format('"' + self.paddle_api + '"', self.inputs, self.params)
            )
            f.write(self.caculate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=str, help="input the yaml path")
    parser.add_argument("--case_name", type=str, default=None, help="case name")
    args = parser.parse_args()

    with open(args.yaml, encoding="utf-8") as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)

    cases_name = [args.case_name]
    # cases_name = yml.keys()  # 生成全部配置
    for case_name in cases_name:
        case = yml.get(case_name)
        case_gen = DebugCaseGen(case, case_name)
        case_gen.py_gen()
