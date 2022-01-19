#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
pdparams compare tool
"""
import os
import argparse
import pytest
import numpy as np
import paddle


class ParamFileReader(object):
    """
    Read pdparams file
    """

    def __init__(self, params_exp_path, params_res_path):
        """
        init
        :param params_exp_path: exp.pdparams path
        :param params_res_path: res.pdparams path
        """
        self.params_exp_path = params_exp_path
        self.params_res_path = params_res_path
        self.params_exp_list = os.listdir(params_exp_path)
        self.params_res_list = os.listdir(params_res_path)
        self.epoch_num = len(self.params_exp_list)
        file_check = len(self.params_exp_list) == len(self.params_exp_list)
        if file_check is False:
            raise Exception("num of exp pdparams_file is not equal num of res pdparams_file!!")

    def __iter__(self):
        """
        iter
        """
        self.count = 0
        return self

    def __next__(self):
        """
        next
        """
        if self.count < self.epoch_num:
            res_filename = self.params_res_list[self.count]
            res_dict = paddle.load(os.path.join(self.params_res_path, self.params_res_list[self.count]))
            exp_filename = self.params_exp_list[self.count]
            exp_dict = paddle.load(os.path.join(self.params_exp_path, self.params_exp_list[self.count]))
            self.count += 1
            return res_dict, exp_dict, res_filename, exp_filename
        else:
            raise StopIteration


class ParamDictReader(object):
    """
    get key/value in exp.pdparams and res.pdparams
    """

    def __init__(self, params_exp, params_res):
        """
        init
        :param params_exp: exp.pdparams file
        :param params_res: res.pdparams file
        """
        self.params_exp = params_exp
        self.params_res = params_res
        self.exp_key_list = list(self.params_exp.keys())
        self.res_key_list = list(self.params_res.keys())
        self.key_num = len(self.exp_key_list)
        key_num_check = self.key_num == len(list(self.params_res.keys()))
        if key_num_check is False:
            raise Exception("num of exp pdparams_key is not equal num of res pdparams_key!!")

    def __iter__(self):
        """
        iter
        """
        self.count = 0
        return self

    def __next__(self):
        """
        next
        """
        if self.count < self.key_num:
            res_key = self.res_key_list[self.count]
            exp_key = self.exp_key_list[self.count]
            res_value = self.params_res[res_key]
            exp_value = self.params_exp[exp_key]
            self.count += 1
            return res_key, res_value, exp_key, exp_value
        else:
            raise StopIteration


class PdparamsCompareTool(object):
    """
    pdparams file compare tool
    """

    def __init__(self, exp_path, res_path, atol, rtol, debug, bug_interrupt):
        """
        init
        :param exp_path: exp.pdparams path
        :param res_path: res.pdparams path
        """
        self.exp_path = exp_path
        self.res_path = res_path
        self.file_reader = iter(ParamFileReader(self.exp_path, self.res_path))
        self.atol = atol
        self.rtol = rtol
        self.debug = debug
        self.bug_interrupt = bug_interrupt
        self.idx = 0
        self.res = True
        self.fail_file_num = 0
        self.fail_file_list = []
        self.success_file_num = 0
        self.success_file_list = []
        self.res_filename = None
        self.exp_filename = None

    def compare(self, result, expect, atol=1e-6, rtol=1e-5):
        """
        比较函数
        :param result: 测试值
        :param expect: 真值
        :param atol: 误差值
        """
        if isinstance(result, np.ndarray):
            expect = np.array(expect)
            res = np.allclose(result, expect, atol=atol, rtol=rtol, equal_nan=True)
            # 出错打印错误数据
            if res is False:
                if self.debug is True:
                    print("the error result is {}".format(result))
                    print("the true expect is {}".format(expect))
            if self.bug_interrupt is True:
                assert res
                assert result.shape == expect.shape
        elif isinstance(result, (list, tuple)):
            for i, j in enumerate(result):
                if isinstance(j, (np.generic, np.ndarray)):
                    self.compare(j, expect[i], atol, rtol)
                else:
                    self.compare(j.numpy(), expect[i], atol, rtol)
        elif isinstance(result, (int, float, bool, str)):
            res = result == expect
            if res is False:
                if self.debug is True:
                    print("the error result is {}".format(result))
                    print("the true expect is {}".format(expect))
            if self.bug_interrupt is True:
                assert res
        else:
            assert result == pytest.approx(expect, atol)

        return res

    def recur_dict_reader(self, exp_dict, res_dict):
        """
        dict/tuple/list reader
        """
        if isinstance(exp_dict, dict):
            dict_reader = iter(ParamDictReader(exp_dict, res_dict))
            for res_key, res_value, exp_key, exp_value in dict_reader:
                res = self.compare(res_key, exp_key, self.atol, self.rtol)
                if res is True:
                    print("idx.{} {} {} key check Success!!!".format(str(self.idx), self.res_filename, exp_key))
                else:
                    self.res = False
                    print("idx.{} {} {} key check Failed!!!".format(str(self.idx), self.res_filename, exp_key))

                if isinstance(exp_value, (dict, list, tuple)):
                    self.recur_dict_reader(exp_value, res_value)
                elif isinstance(exp_value, (int, float, bool, str)):
                    res = self.compare(res_key, exp_key, self.atol, self.rtol)
                    if res is True:
                        print(
                            "idx.{} {} {} {} check Success!!!".format(
                                str(self.idx), self.res_filename, exp_key, type(exp_value)
                            )
                        )
                    else:
                        self.res = False
                        print(
                            "idx.{} {} {} {} check Failed!!!".format(
                                str(self.idx), self.res_filename, exp_key, type(exp_value)
                            )
                        )
                else:
                    res = self.compare(res_value.numpy(), exp_value.numpy(), self.atol, self.rtol)
                    if res is True:
                        print("idx.{} {} {} value check Success!!!".format(str(self.idx), self.res_filename, exp_key))
                    else:
                        self.res = False
                        print("idx.{} {} {} value check Failed!!!".format(str(self.idx), self.res_filename, exp_key))
        elif isinstance(exp_dict, (list, tuple)):
            len_check = len(exp_dict) == len(res_dict)
            if len_check is False:
                if self.debug is True:
                    print("error exp_dict is: ", exp_dict)
                    print("true res_dict is: ", res_dict)
                    print("exp_dict is not equal to res_dict !!!")
            if self.bug_interrupt is True:
                assert len_check
            step = 0
            for res_value, exp_value in zip(res_dict, exp_dict):
                if isinstance(exp_value, (dict, list, tuple)):
                    self.recur_dict_reader(exp_value, res_value)
                elif isinstance(exp_value, (int, float, bool)):
                    res = self.compare(res_value, exp_value, self.atol, self.rtol)
                    if res is True:
                        print(
                            "idx.{} {} list[{}] value check Success!!!".format(
                                str(self.idx), self.res_filename, str(step)
                            )
                        )
                    else:
                        self.res = False
                        print(
                            "idx.{} {} list[{}] value check Failed!!!".format(
                                str(self.idx), self.res_filename, str(step)
                            )
                        )
                else:
                    res = self.compare(res_value.numpy(), exp_value.numpy(), self.atol, self.rtol)
                    if res is True:
                        print(
                            "idx.{} {} list[{}] value check Success!!!".format(
                                str(self.idx), self.res_filename, str(step)
                            )
                        )
                    else:
                        self.res = False
                        print(
                            "idx.{} {} list[{}] value check Failed!!!".format(
                                str(self.idx), self.res_filename, str(step)
                            )
                        )
                step += 1

    def check_params(self):
        """
        check diff
        """
        self.idx = 0
        for res_dict, exp_dict, res_filename, exp_filename in self.file_reader:
            self.idx += 1
            self.res = True
            self.res_filename = res_filename
            self.exp_filename = exp_filename
            print(
                "idx.{} {} pdparams start testing ===============================>>>>>>>>".format(
                    str(self.idx), self.res_filename
                )
            )
            self.recur_dict_reader(exp_dict, res_dict)
            if self.res is True:
                self.success_file_num += 1
                self.success_file_list.append(self.res_filename)
                print(
                    "idx.{} {} pdparams test completed Success ===============================>>>>>>>>".format(
                        str(self.idx), self.res_filename
                    )
                )
            else:
                self.fail_file_num += 1
                self.fail_file_list.append(self.res_filename)
                print(
                    "idx.{} {} pdparams test completed Failed ===============================>>>>>>>>".format(
                        str(self.idx), self.res_filename
                    )
                )
        return self.success_file_num, self.fail_file_num, self.success_file_list, self.fail_file_list


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--params_exp", type=str, default=None, help="true pdparams path.")
parser.add_argument("--params_res", type=str, default=None, help="test pdparams path.")
parser.add_argument("--atol", type=float, default=1e-20, help="absolute diff acc.")
parser.add_argument("--rtol", type=float, default=1e-20, help="relative diff acc.")
parser.add_argument("--debug", type=bool, default=False, help="whether use debug mode.")
parser.add_argument("--bug_interrupt", type=bool, default=False, help="When there is Bug, stopping testing or not.")
args = parser.parse_args()


if __name__ == "__main__":
    check = PdparamsCompareTool(args.params_exp, args.params_res, args.atol, args.rtol, args.debug, args.bug_interrupt)
    success_file_num, fail_file_num, success_file_list, fail_file_list = check.check_params()
    print("+++++++++++++++++++++++++++++++ final result is here +++++++++++++++++++++++++++++++")
    print("Success file number: ", success_file_num)
    print("Success file list: ", success_file_list)
    print("failed file number: ", fail_file_num)
    print("failed file list: ", fail_file_list)
