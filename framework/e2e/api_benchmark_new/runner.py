#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


"""
1. 安装
实例化Snapshot
    获取md5 id
    打印机器状态
实例化DB
    初始化任务id
    数据库检查
实例化统计策略
    选择合适的计算策略
实例化复验策略
    double check
实例化报警策略
    alarm

2. 任务执行
保存执行快照
    打印机器状态
    paddle版本，执行内容等等元数据
    注意跑case的时候禁用logger和print相关内容
统计策略
    根据统计策略将数据整理成合理的格式
复验策略
    记录异常值传入复验对象，进行复验
报警策略
    对复验不通过的进行报警
入库
    将最终结果json，解析入库

3. 整体任务返回
根据任务执行结果，汇总使用logger打印结果列表。
失败exit 非零数据
"""
