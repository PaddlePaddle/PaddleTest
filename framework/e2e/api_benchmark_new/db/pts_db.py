#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ci data
"""

import traceback
from db import DB


ACCURACY = "%.6g"


class PTSdb(DB):
    """仅针对CI例行任务"""

    def __init__(self, storage):
        super(PTSdb, self).__init__(storage)

    def pts_update_job(self, id, status, job_dict):
        """

        :param id:
        :return:
        """
        try:
            self.update_by_id(table="job", data=dict({"status": status}, **job_dict), id=id)
        except Exception as e:
            self.update_by_id(table="job", data=dict({"status": "error"}, **job_dict), id=id)
            print(traceback.format_exc())
            print(e)
