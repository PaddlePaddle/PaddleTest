#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ci data
"""

import json
from datetime import datetime
import traceback
from db import DB


ACCURACY = "%.6g"


class CIdb(DB):
    """仅针对CI例行任务"""

    def __init__(self, storage):
        super(CIdb, self).__init__(storage)

    def ci_insert_job(
        self,
        commit,
        version,
        hostname,
        place,
        system,
        cuda,
        cudnn,
        snapshot,
        md5_id,
        uid,
        routine,
        ci,
        comment,
        enable_backward,
        python,
        yaml_info,
        wheel_link,
        description,
        create_time,
        update_time,
    ):
        """向job表中录入数据, 表中部分字段后续更新"""
        data = {
            "framework": "paddle",
            "status": "running",
            "mode": "schedule",
            "commit": commit,
            "version": version,
            "hostname": hostname,
            "place": place,
            "system": system,
            "cuda": cuda,
            "cudnn": cudnn,
            "snapshot": snapshot,
            "md5_id": md5_id,
            "uid": uid,
            "routine": routine,
            "ci": ci,
            "comment": comment,
            "enable_backward": enable_backward,
            "python": python,
            "yaml_info": yaml_info,
            "wheel_link": wheel_link,
            "description": description,
            "create_time": create_time,
            "update_time": update_time,
        }
        id = self.insert(table="job", data=data)
        return id

    def ci_insert_case(self, jid, data_dict, create_time):
        """向case表中录入数据"""
        # for case_name, value_dict in data_dict.items():
        #     print('value_dict is: ', value_dict)
        #     value_dict['result']['forward'] = ACCURACY % value_dict['result']['forward']
        #     value_dict['result']['total'] = ACCURACY % value_dict['result']['total']
        #     value_dict['result']['backward'] = ACCURACY % value_dict['result']['backward']
        #     value_dict['result']['best_total'] = ACCURACY % value_dict['result']['best_total']
        #     data = {
        #         "jid": jid,
        #         "case_name": case_name,
        #         "api": value_dict['result']['api'],
        #         "result": json.dumps(value_dict['result']),
        #         "create_time": create_time,
        #     }
        #     # print('data is: ', data)
        #     case_id = self.insert(table="case", data=data)
        #     # print('case_id is: ', case_id)
        #     if case_id == -1:
        #         self.success = False

        # print('data_dict is: ', data_dict)
        data_dict["result"]["forward"] = ACCURACY % data_dict["result"]["forward"]
        data_dict["result"]["total"] = ACCURACY % data_dict["result"]["total"]
        data_dict["result"]["backward"] = ACCURACY % data_dict["result"]["backward"]
        data_dict["result"]["best_total"] = ACCURACY % data_dict["result"]["best_total"]
        data = {
            "jid": jid,
            "case_name": data_dict["case_name"],
            "api": data_dict["result"]["api"],
            "result": json.dumps(data_dict["result"]),
            "create_time": create_time,
        }
        # print('data is: ', data)
        case_id = self.insert(table="case", data=data)
        # print('case_id is: ', case_id)
        # if case_id == -1:
        #     self.success = False
        return case_id

    def ci_update_job(self, id, status, update_time):
        """数据录入完成后更新job表中的部分字段"""
        data = {"status": status, "update_time": update_time}
        self.update_by_id(table="job", data=data, id=id)

    def ci_select_baseline_job(self, comment, routine, ci, md5_id):
        """通过comment字段、ci字段、机器唯一标识码，查找baseline数据"""
        condition_list = [
            "comment = '{}'".format(comment),
            "status = 'done'",
            "routine = '{}'".format(routine),
            "ci = '{}'".format(ci),
            "md5_id = '{}'".format(md5_id),
        ]
        res = self.select(table="job", condition_list=condition_list)
        baseline_job = res[-1]
        job_id = baseline_job["id"]
        return job_id

    def ci_update_job_and_insert_case(self, job_id, job_dict, error_dict):
        """
        用于ci通用的job/case录入方法
        :param job_id: job的id号码
        :param job_dict: 多个case的性能信息组成的dict
        :param error_dict: 多个case的错误信息组成的dict
        """
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if bool(error_dict):
            self.ci_update_job(id=job_id, status="error", update_time=time_now)
            raise Exception("error data should not be inserted to api benchmark db!!")
        else:
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                for case_name, data_dict in job_dict.items():
                    # print('data_dict is: ', data_dict)
                    self.ci_insert_case(jid=job_id, data_dict=data_dict, create_time=time_now)
                self.ci_update_job(id=job_id, status="done", update_time=time_now)
            except Exception as e:
                self.ci_update_job(id=job_id, status="error", update_time=time_now)
                print(traceback.format_exc())
                print(e)
