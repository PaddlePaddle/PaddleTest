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
from io_test import TestDataset, TestDataLoader

logger = Logger("generate dataloader")


# 1. 解析yaml，生成信息
class GTCase(object):
    """
    Generate and Test case
    """

    def __init__(self, yaml):
        """
        initialize
        """
        self.yml = yaml
        self.obj = YamlLoader(self.yml)
        self.logger = logger

    def _generate_dataset(self, dataset_info):
        """
        generate dataset
        """
        dataset = GenDataset(dataset_info)
        return dataset

    def _generate_dataloader(self, dataset, batch_sampler_info, other_params_info):
        """
        generate dataloader
        """
        batch_sampler = None
        if batch_sampler_info:
            batch_sampler = SetBatchSampler(dataset, batch_sampler_info)()
        obj_data = GenDataLoader(dataset, **other_params_info)
        data_loader = obj_data.exec(batch_sampler)
        return data_loader

    def run_case(self):
        """
        执行dataset和dataloader的自测
        """
        cases_name = self.obj.get_all_case_name()
        for name in cases_name:
            # 1. 解析yaml，生成信息
            case = self.obj.get_case_info(name)
            trans_obj = DataLoaderTrans(case)
            dataset_info = trans_obj.get_dataset()
            batch_sampler_info = trans_obj.get_batch_sampler()
            other_params_info = trans_obj.get_other_params()

            # 2. 导入信息，生成dataloader
            dataset = self._generate_dataset(dataset_info)
            data_loader = self._generate_dataloader(dataset, batch_sampler_info, other_params_info)
            yield dataset, data_loader

            # 3.执行测试
            # (1) 测dataset
            obj_test_dataset = TestDataset(dataset)
            if isinstance(dataset_info, str):
                # 测试数据单元输入
                obj_test_dataset.run(dataset_info)
            else:
                # obj_test_dataset.run(dataset_info)
                obj_test_dataset.run([10, dataset_info])  # todo: 应传入num_samples

            # (2)测试dataset
            dataloader_info = self._gen_dataloader_info(batch_sampler_info, other_params_info)
            test_dl_obj = TestDataLoader(data_loader)
            test_dl_obj.run(dataset, dataloader_info)

    def generate(self, case_name):
        """
        根据case名称产生对应的dataset和dataloader，接layer测试
        """
        case = self.obj.get_case_info(case_name)
        trans_obj = DataLoaderTrans(case)
        dataset_info = trans_obj.get_dataset()
        batch_sampler_info = trans_obj.get_batch_sampler()
        other_params_info = trans_obj.get_other_params()
        dataset = self._generate_dataset(dataset_info)
        data_loader = self._generate_dataloader(dataset, batch_sampler_info, other_params_info)
        return dataset, data_loader

    def _gen_dataloader_info(self, batch_sampler_info, other_params_info):
        """
        生成dataloader-info， 主要获取batch_size、shuffle、drop_last；用于测试
        """
        dataloader_info = {}
        if batch_sampler_info:
            for k in batch_sampler_info.keys():
                dataloader_info[k] = batch_sampler_info[k]
        else:
            for k in other_params_info.keys():
                dataloader_info[k] = other_params_info[k]
        if not dataloader_info:
            self.logger.get_log().error("data-loader info must be set !!!")
            assert ValueError
        return dataloader_info


if __name__ == "__main__":
    obj = GTCase("dataloader.yml")
    obj.run_case()
