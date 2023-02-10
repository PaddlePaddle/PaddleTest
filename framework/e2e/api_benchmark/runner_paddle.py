#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
runner
"""

import argparse
import platform
import sys

import paddle

sys.path.append("..")
from utils.yaml_loader import YamlLoader
from utils.logger import logger
from benchtrans import BenchTrans
from jelly_v2 import Jelly_v2
from db import DB
from tools import delete


SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


def schedule(
    yaml_path,
    framework,
    case_name=None,
    place=None,
    card=None,
    test_index=None,
    enable_backward=True,
):
    """
    例行任务，调试任务二次运行调用。 一定入库，存在数据库操作
    """
    yaml_file = yaml_path
    yaml_loader = YamlLoader(yaml_file)
    if case_name is None:
        cases_name = yaml_loader.get_all_case_name()
        for case_name in cases_name:
            # Skip cases
            if case_name in SKIP_DICT[platform.system()]:
                logger.get_log().warning("skip case -->{}<--".format(case_name))
                continue
            if SPECIAL and case_name not in SKIP_DICT[platform.system()]:
                logger.get_log().warning("case is not in index_dict, skipping...-->{}<--".format(case_name))
                continue
            # if not case_name.endswith("_0"):
            #     logger.get_log().warning("skip case -->{}<--".format(case_name))
            #     continue
            if test_index == "case_0":
                if not case_name.endswith("_0"):
                    logger.get_log().warning("skip case -->{}<--".format(case_name))
                    continue
            if test_index == "case_1":
                if not case_name.endswith("_1"):
                    logger.get_log().warning("skip case -->{}<--".format(case_name))
                    continue
            if test_index == "case_2":
                if not case_name.endswith("_2"):
                    logger.get_log().warning("skip case -->{}<--".format(case_name))
                    continue
            case_info = yaml_loader.get_case_info(case_name)
            try:
                bt = BenchTrans(case_info)
                if enable_backward == "False":
                    enable_backward_trigger = False
                else:
                    enable_backward_trigger = bt.enable_backward()
                if framework == "paddle":
                    api = bt.get_paddle_api()
                    jelly = Jelly_v2(
                        api=api,
                        framework=framework,
                        title=case_name,
                        place=place,
                        card=card,
                        default_dtype="float32",
                        enable_backward=enable_backward_trigger,
                    )
                    jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
                    jelly.set_paddle_method(bt.get_paddle_method())
                    jelly.run_schedule()
            except Exception as e:
                paddle.enable_static()
                paddle.disable_static()
                logger.get_log().warn(e)
    else:
        case_info = yaml_loader.get_case_info(case_name)
        bt = BenchTrans(case_info)
        if framework == "paddle":
            api = bt.get_paddle_api()
            if enable_backward == "False":
                enable_backward_trigger = False
            else:
                enable_backward_trigger = bt.enable_backward()
            jelly = Jelly_v2(
                api=api,
                framework=framework,
                title=case_name,
                place=place,
                card=card,
                enable_backward=enable_backward_trigger,
            )
            jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
            jelly.set_paddle_method(bt.get_paddle_method())
            jelly.run_schedule()


def testing(yaml_path, case_name, framework, place=None, card=None, enable_backward=True):
    """
    testing mode 本地调试用
    """
    yaml_file = yaml_path
    yaml_loader = YamlLoader(yaml_file)
    case_info = yaml_loader.get_case_info(case_name)
    bt = BenchTrans(case_info)
    if framework == "paddle":
        api = bt.get_paddle_api()
        if enable_backward == "False":
            enable_backward_trigger = False
        else:
            enable_backward_trigger = bt.enable_backward()
        jelly = Jelly_v2(
            api=api,
            framework=framework,
            title=case_name,
            place=place,
            card=card,
            enable_backward=enable_backward_trigger,
        )
        jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
        jelly.set_paddle_method(bt.get_paddle_method())
        jelly.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=str, help="input the yaml path")
    parser.add_argument(
        "--mode",
        type=str,
        default="testing",
        help="""choose mode: [schedule] for ce or [testing] for native test or [rerun] for double check.
              [schedule] and [rerun] will write data into database """,
    )
    parser.add_argument("--framework", type=str, default="paddle", help="[paddle] | [torch] | [all]")
    parser.add_argument("--case", type=str, default="Tanh", help="case name for [testing] and [rerun] mode")
    parser.add_argument("--place", type=str, default="cpu", help="[cpu] or [gpu]")
    parser.add_argument("--cuda", type=str, default=None, help="cuda version like v10.2 | v11.2 etc.")
    parser.add_argument("--cudnn", type=str, default=None, help="cudnn version like v7.6.5 etc.")
    parser.add_argument("--card", type=str, default=None, help="card number , default is 0")
    parser.add_argument("--comment", type=str, default=None, help="your comment")
    parser.add_argument("--test_index", type=str, default=None, help="[case_0] or [case_1] or [case_2]")
    parser.add_argument("--enable_backward", type=str, default="True", help="if True, enable backward test")
    parser.add_argument("--storage", type=str, default=None, help="path of storage.yaml")
    args = parser.parse_args()

    # 验证参数组合正确性
    if args.place == "gpu" and (args.cuda is None or args.cudnn is None):
        logger.get_log().error("GPU情况下必须输入cuda和cudnn版本")
        raise AttributeError

    if args.mode == "schedule":
        # 判断log文件是否干净
        delete("./log")
        db = DB(storage=args.storage)
        try:
            db.init_mission(
                framework=args.framework,
                mode=args.mode,
                place=args.place,
                cuda=args.cuda,
                cudnn=args.cudnn,
                card=args.card,
                comment=args.comment,
            )
            schedule(
                yaml_path=args.yaml,
                framework=args.framework,
                place=args.place,
                card=args.card,
                test_index=args.test_index,
                enable_backward=args.enable_backward,
            )
            db.save()
        except Exception as e:
            logger.get_log().error(e)
            db.error()
    elif args.mode == "testing":
        testing(
            args.yaml,
            args.case,
            framework=args.framework,
            place=args.place,
            card=args.card,
            enable_backward=args.enable_backward,
        )
    elif args.mode == "rerun":
        # db = DB()
        try:
            # db.init_mission(
            #     framework=args.framework,
            #     mode=args.mode,
            #     place=args.place,
            #     cuda=args.cuda,
            #     cudnn=args.cudnn,
            #     card=args.card,
            # )
            schedule(
                args.yaml,
                framework=args.framework,
                case_name=args.case,
                place=args.place,
                card=args.card,
                test_index=args.test_index,
                enable_backward=args.enable_backward,
            )
            # db.save()
        except Exception as e:
            logger.get_log().error(e)
            # db.error()
    else:
        raise AttributeError
