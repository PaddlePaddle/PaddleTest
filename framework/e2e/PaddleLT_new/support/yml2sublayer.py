#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yaml转写成sublayer子图代码
"""

import os
from inspect import isclass
import paddle
import yaml


class YamlLoader(object):
    """
    yaml_loader
    """

    def __init__(self, yml):
        """initialize"""
        try:
            with open(yml, encoding="utf-8") as f:
                self.yml = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print(e)

    def __str__(self):
        """str"""
        return str(self.yml)

    def get_case_info(self, case_name):
        """
        get case info
        """
        return {"info": self.yml.get(case_name), "name": case_name}

    def get_all_case_name(self):
        """
        get all case name
        """
        # 获取全部case name
        return self.yml.keys()


yaml_file = "nn.yml"
yaml_loader = YamlLoader(yaml_file)

all_cases = list(yaml_loader.get_all_case_name())
# print("all_cases is: ", len(all_cases))
# exit(0)

case_path = "api_case"  # 生成case文件夹路径
if not os.path.exists(case_path):
    os.system(f"mkdir {case_path}")

for case in all_cases:  # case by case写入子图
    case_info = yaml_loader.get_case_info(case)
    print(case_info)
    # case_info结构dict参考:
    # {'info':
    # {'desc': '嵌入层(Embedding Layer)',
    # 'paddle': {'api_name': 'paddle.nn.functional.embedding',
    # 'inputs': {'x': {'random': True, 'type': 'Tensor', 'dtype': 'int64',
    # 'shape': [3, 1], 'range': [2, 8]}}, 'params': {
    # 'weight': {'random': True, 'type': 'Tensor', 'dtype': 'float32',
    # 'shape': [10, 3], 'range': [-1, 1]}, 'padding_idx': -1, 'sparse': True}}}, 'name': 'embedding_base'}

    func = case_info["info"]["paddle"]["api_name"]

    layer_params = ""
    if "params" in case_info["info"]["paddle"].keys():  # 判断是否包含"params"
        in_params = case_info["info"]["paddle"]["params"]
        for k, v in in_params.items():
            if not isinstance(v, dict):
                layer_params += f"{k}={v}, "
            else:
                if v["random"] and v["type"] == "Tensor":
                    if "range" in v.keys():
                        low = v["range"][0]
                        high = v["range"][1]
                    else:
                        low = -1
                        high = 1
                    if v["dtype"] == "int":
                        layer_params += f"paddle.to_tensor(np.random.randint({low}, {high}, {v['shape']}), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "int32":
                        layer_params += f"paddle.to_tensor(np.random.randint({low}, {high}, {v['shape']}).astype('int32'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "int64":
                        layer_params += f"paddle.to_tensor(np.random.randint({low}, {high}, {v['shape']}).astype('int64'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "float":
                        layer_params += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "float16":
                        layer_params += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}).astype('float16'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "float32":
                        layer_params += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}).astype('float32'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "float64":
                        layer_params += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}).astype('float64'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "bfloat16":
                        layer_params += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}).astype('bfloat16'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "complex64":
                        layer_params += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}) + ({low} + ({high} - {low}) * np.random.random({v['shape']})) * 1j).astype(np.complex64), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "complex":
                        layer_params += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}) + ({low} + ({high} - {low}) * np.random.random({v['shape']})) * 1j, dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] in "complex128":
                        layer_params += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}) + ({low} + ({high} - {low}) * np.random.random({v['shape']})) * 1j, dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "bool":
                        layer_params += f"paddle.to_tensor(np.random.randint(0, 2, {v['shape']}).astype('bool'), dtype={v['dtype']}), stop_gradient=False), "
                elif not v["random"] and v["type"] == "Tensor":
                    layer_params += f"paddle.to_tensor({v['value']}, dtype={v['dtype']}, stop_gradient=False), "

    np_inputs = ""
    tensor_inputs = ""
    inputs_k = ""

    if "inputs" in case_info["info"]["paddle"].keys():  # 判断是否有"inputs"
        for k, v in case_info["info"]["paddle"]["inputs"].items():
            inputs_k += f"{k}, "

            if isinstance(v, dict):
                if v["random"] and v["type"] == "Tensor":
                    if "range" in v.keys():
                        low = v["range"][0]
                        high = v["range"][1]
                    else:
                        low = -1
                        high = 1
                    if v["dtype"] == "int":
                        np_inputs += f"np.random.randint({low}, {high}, {v['shape']}), "
                        tensor_inputs += f"paddle.to_tensor(np.random.randint({low}, {high}, {v['shape']}), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "int32":
                        np_inputs += f"np.random.randint({low}, {high}, {v['shape']}).astype('int32'), "
                        tensor_inputs += f"paddle.to_tensor(np.random.randint({low}, {high}, {v['shape']}).astype('int32'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "int64":
                        np_inputs += f"np.random.randint({low}, {high}, {v['shape']}).astype('int64'), "
                        tensor_inputs += f"paddle.to_tensor(np.random.randint({low}, {high}, {v['shape']}).astype('int64'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "float":
                        np_inputs += f"{low} + ({high} - {low}) * np.random.random({v['shape']}), "
                        tensor_inputs += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "float16":
                        np_inputs += f"{low} + ({high} - {low}) * np.random.random({v['shape']}).astype('float16'), "
                        tensor_inputs += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}).astype('float16'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "float32":
                        np_inputs += f"{low} + ({high} - {low}) * np.random.random({v['shape']}).astype('float32'), "
                        tensor_inputs += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}).astype('float32'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "float64":
                        np_inputs += f"{low} + ({high} - {low}) * np.random.random({v['shape']}).astype('float64'), "
                        tensor_inputs += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}).astype('float64'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "bfloat16":
                        np_inputs += f"{low} + ({high} - {low}) * np.random.random({v['shape']}).astype('bfloat16'), "
                        tensor_inputs += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}).astype('bfloat16'), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "complex64":
                        np_inputs += f"({low} + ({high} - {low}) * np.random.random({v['shape']}) + ({low} + ({high} - {low}) * np.random.random({v['shape']})) * 1j).astype(np.complex64), "
                        tensor_inputs += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}) + ({low} + ({high} - {low}) * np.random.random({v['shape']})) * 1j).astype(np.complex64), dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "complex":
                        np_inputs += f"{low} + ({high} - {low}) * np.random.random({v['shape']}) + ({low} + ({high} - {low}) * np.random.random({v['shape']})) * 1j, "
                        tensor_inputs += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}) + ({low} + ({high} - {low}) * np.random.random({v['shape']})) * 1j, dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] in "complex128":
                        np_inputs += f"{low} + ({high} - {low}) * np.random.random({v['shape']}) + ({low} + ({high} - {low}) * np.random.random({v['shape']})) * 1j, "
                        tensor_inputs += f"paddle.to_tensor({low} + ({high} - {low}) * np.random.random({v['shape']}) + ({low} + ({high} - {low}) * np.random.random({v['shape']})) * 1j, dtype='{v['dtype']}', stop_gradient=False), "
                    elif v["dtype"] == "bool":
                        np_inputs += f"np.random.randint(0, 2, {v['shape']}).astype('bool'), "
                        tensor_inputs += f"paddle.to_tensor(np.random.randint(0, 2, {v['shape']}).astype('bool'), dtype={v['dtype']}), stop_gradient=False), "
                elif not v["random"] and v["type"] == "Tensor":
                    np_inputs += f"paddle.to_tensor({v['value']}, dtype={v['dtype']}, stop_gradient=False), "
            else:
                np_inputs = v

    if isclass(eval(func)):  # 判断paddle api为class或是function
        # 创建一个新的 Python 文件并写入代码
        with open(f"{case_path}/{case}_class.py", "w") as file:
            file.write(
                f"""\
import paddle


class LayerCase(paddle.nn.Layer):
    \"\"\"
    case名称: {case_info["name"]}
    api简介: {case_info["info"]['desc']}
    \"\"\"

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = {func}({layer_params})

    def forward(self, {inputs_k}):
        \"\"\"
        forward
        \"\"\"
        out = self.func({inputs_k})
        return out


def create_tensor_inputs():
    \"\"\"
    paddle tensor
    \"\"\"
    inputs = ({tensor_inputs})
    return inputs


def create_numpy_inputs():
    \"\"\"
    numpy array
    \"\"\"
    inputs = ({np_inputs})
    return inputs

"""
            )

    else:
        # 创建一个新的 Python 文件并写入代码
        with open(f"{case_path}/{case}_func.py", "w") as file:
            file.write(
                f"""\
import paddle


class LayerCase(paddle.nn.Layer):
    \"\"\"
    case名称: {case_info["name"]}
    api简介: {case_info["info"]['desc']}
    \"\"\"

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = {func}({layer_params})

    def forward(self, {inputs_k}):
        \"\"\"
        forward
        \"\"\"
        out = {func}({inputs_k}, {layer_params})
        return out


def create_tensor_inputs():
    \"\"\"
    paddle tensor
    \"\"\"
    inputs = ({tensor_inputs})
    return inputs


def create_numpy_inputs():
    \"\"\"
    numpy array
    \"\"\"
    inputs = ({np_inputs})
    return inputs

"""
            )
