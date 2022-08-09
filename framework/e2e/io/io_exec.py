"""
调度
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname("utils"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.yaml_loader import YamlLoader
from utils.logger import Logger
from io_trans import DataLoaderTrans
from io_reader import GenDataset, SetBatchSampler
from io_loader import GenDataLoader
from io_test import TestDataset

logger = Logger("generate dataloader")


# 1. 解析yaml，生成信息

yam_obj = YamlLoader("dataloader.yml")
cases_name = yam_obj.get_all_case_name()
case1 = yam_obj.get_case_info(list(cases_name)[4])  # TODO: for 执行

trans_obj = DataLoaderTrans(case1)
dataset_info = trans_obj.get_dataset()
# print(dataset_info)
batch_sampler_info = trans_obj.get_batch_sampler()
other_params_info = trans_obj.get_other_params()

# 2. 导入信息，生成dataloader

## （1）执行read_data，生成参数
dataset = GenDataset(dataset_info)  # TODO: num_samples默认是10，可传递
# for i in range(len(dataset)):
#     print(dataset[i])

batch_sampler = None
if batch_sampler_info:
    batch_sampler = SetBatchSampler(dataset, batch_sampler_info)()
    # for batch_indices in batch_sampler:
    #     print(batch_indices)

## （2）执行data_loader，生成data_loader
# print(other_params_info)
obj_data = GenDataLoader(dataset, **other_params_info)
data_loader = obj_data.exec(batch_sampler)

for i in data_loader():
    print(i)
    print(type(i))

print(len(data_loader))


# 3.执行测试
# （1）测dataset
obj_test_dataset = TestDataset(dataset)
# 测试数据单元输入
obj_test_dataset.run([10, dataset_info])  # todo: 应传入num_samples
# 测试地址数据集输入
# obj_test_dataset.run(dataset_info)


# (2) 测dataloader
