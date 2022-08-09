#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
runner
"""

import argparse
import sys

sys.path.append("..")
from utils.yaml_loader import YamlLoader
from utils.logger import logger
from benchtrans import BenchTrans
from jelly import Jelly
from jelly_v2 import Jelly_v2
from db import DB


def schedule(yaml_path, framework, case_name=None, place=None, card=None):
    """
    例行任务，调试任务二次运行调用。 一定入库，存在数据库操作
    """
    yaml_file = yaml_path
    yaml_loader = YamlLoader(yaml_file)
    if case_name is None:
        cases_name = yaml_loader.get_all_case_name()
        for case_name in cases_name:
            case_info = yaml_loader.get_case_info(case_name)
            try:
                bt = BenchTrans(case_info)
                if framework == "paddle":
                    api = bt.get_paddle_api()
                    jelly = Jelly_v2(api=api, framework=framework, title=case_name, place=place, card=card)
                    jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
                    jelly.run_schedule()
                elif framework == "torch":
                    if not bt.check_torch:
                        logger.get_log().info("{} 缺少Torch配置, Skip...".format(case_name))
                        continue
                    api = bt.get_torch_api()
                    jelly = Jelly_v2(api=api, framework=framework, title=case_name, place=place, card=card)
                    jelly.set_torch_param(bt.get_torch_inputs(), bt.get_torch_param())
                    jelly.run_schedule()
            except Exception as e:
                logger.get_log().warn(e)
    else:
        case_info = yaml_loader.get_case_info(case_name)
        bt = BenchTrans(case_info)
        if framework == "paddle":
            api = bt.get_paddle_api()
            jelly = Jelly_v2(api=api, framework=framework, title=case_name, place=place, card=card)
            jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
            jelly.run_schedule()

        elif framework == "torch":
            api = bt.get_torch_api()
            jelly = Jelly_v2(api=api, framework=framework, title=case_name, place=place, card=card)
            jelly.set_torch_param(bt.get_torch_inputs(), bt.get_torch_param())
            jelly.run_schedule()


def testing(yaml_path, case_name, framework, place=None, card=None):
    """
    testing mode 本地调试用
    """
    yaml_file = yaml_path
    yaml_loader = YamlLoader(yaml_file)
    case_info = yaml_loader.get_case_info(case_name)
    bt = BenchTrans(case_info)
    if framework == "all":
        jelly = Jelly(bt.get_paddle_api(), bt.get_torch_api(), title=case_name, place=place, card=card)
        jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
        jelly.set_torch_param(bt.get_torch_inputs(), bt.get_torch_param())
        jelly.run()

    elif framework == "paddle":
        api = bt.get_paddle_api()
        jelly = Jelly_v2(api=api, framework=framework, title=case_name, place=place, card=card)
        jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
        jelly.run()

    elif framework == "torch":
        api = bt.get_torch_api()
        jelly = Jelly_v2(api=api, framework=framework, title=case_name, place=place, card=card)
        jelly.set_torch_param(bt.get_torch_inputs(), bt.get_torch_param())
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
    args = parser.parse_args()

    # 验证参数组合正确性
    if args.place == "gpu" and (args.cuda is None or args.cudnn is None):
        logger.get_log().error("GPU情况下必须输入cuda和cudnn版本")
        raise AttributeError

    if args.mode == "schedule":
        db = DB()
        try:
            db.init_mission(
                framework=args.framework,
                mode=args.mode,
                place=args.place,
                cuda=args.cuda,
                cudnn=args.cudnn,
                card=args.card,
            )
            schedule(yaml_path=args.yaml, framework=args.framework, place=args.place, card=args.card)
            db.save()
        except Exception as e:
            logger.get_log().error(e)
            db.error()
    elif args.mode == "testing":
        testing(args.yaml, args.case, framework=args.framework, place=args.place, card=args.card)
    elif args.mode == "rerun":
        db = DB()
        try:
            db.init_mission(
                framework=args.framework,
                mode=args.mode,
                place=args.place,
                cuda=args.cuda,
                cudnn=args.cudnn,
                card=args.card,
            )
            schedule(args.yaml, framework=args.framework, case_name=args.case, place=args.place, card=args.card)
            db.save()
        except Exception as e:
            logger.get_log().error(e)
            db.error()
    else:
        raise AttributeError
