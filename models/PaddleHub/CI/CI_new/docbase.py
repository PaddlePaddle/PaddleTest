#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
docbase.py
"""
import os
import re
import argparse


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--path", type=str, default=None, help="model path for predict.")
parser.add_argument("--name", type=str, default=None, help="model name for predict.")
args = parser.parse_args()


class HubDocTest(object):
    """hub doc test"""

    def __init__(self, md_file, module_name="module"):
        self.module_name = module_name
        self.pwd = os.getcwd()
        self.doc_file = md_file
        self.predict_pattern = re.compile(r"预测代码示例([\s\S]*?)###")
        self.python_pattern = re.compile(r"```python([\s\S]*?)```")

    def read_doc(self, doc_file):
        """read doc"""
        with open(doc_file) as f:
            doc_list = f.readlines()
        return doc_list

    def mk_doc(self, in_str, new_doc_name="doc_tmp.py", del_space=0):
        """make new doc"""
        if os.path.exists(os.path.join(self.pwd, new_doc_name)):
            os.remove(os.path.join(self.pwd, new_doc_name))
        predict_tmp = open(os.path.join(self.pwd, new_doc_name), "w")
        if isinstance(in_str, list):
            for index, str_ in enumerate(in_str):
                in_str[index] = in_str[index][del_space:]
                predict_tmp.write(in_str[index])
        else:
            predict_tmp.write(in_str)
        predict_tmp.close()

    def mk_predict_py(self):
        """main def"""
        doc_list = self.read_doc(self.doc_file)
        doc_str = "".join(doc_list)
        tmp_list = self.predict_pattern.findall(doc_str)
        if len(tmp_list) > 1:
            raise Exception("predict code in README.md is not unique!!!")
        if len(tmp_list) > 1:
            raise Exception("predict code in README.md dose not exit!!!")
        try:
            predict_python = self.python_pattern.findall(tmp_list[0])
        except Exception:
            raise IndexError("README.md format is wrong!!! hub CI cannot catch keywords of predict code.")
        self.mk_doc(predict_python[0], new_doc_name="doc_tmp.py")
        tmp_list = self.read_doc(os.path.join(self.pwd, "doc_tmp.py"))

        for index, str_ in enumerate(tmp_list):
            tmp = re.search("import paddlehub as hub", str_)
            if tmp is None:
                continue
            else:
                del_space = tmp.span()[0]
                break

        self.mk_doc(
            tmp_list, new_doc_name=os.path.join(self.pwd, "test_" + self.module_name + ".py"), del_space=del_space
        )
        os.remove("doc_tmp.py")


if __name__ == "__main__":
    create_py = HubDocTest(md_file=args.path, module_name=args.name)
    create_py.mk_predict_py()
