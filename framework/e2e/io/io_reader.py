"""
io_reader
"""
import json
import gzip
from collections import Iterator
import paddle
from paddle.io import Dataset
from utils.logger import Logger

import numpy as np

logger = Logger("read_data")


class DataGenerator(object):
    """
    DataGenerator class
    """

    def __init__(self, dir):
        """
        init
        """
        self.datafile = dir

    def __call__(self, mode="train"):
        """
        call function
        """
        data = self._read_json()
        # todo: 其他类型数据格式的读入
        try:
            train_set, val_set, eval_set = data
        except:
            logger.get_log().error("dataset can not be split train, valid and eval")
            raise ValueError
        if mode == "train":
            return train_set
        elif mode == "valid":
            return val_set
        elif mode == "eval":
            return eval_set
        else:
            raise NameError

    def _read_json(self):
        """
        read json file
        """
        data = json.load(gzip.open(self.datafile))
        return data


class GenDataset(Dataset):
    """
    generate dataset class
    """

    def __init__(self, data, num_samples=10):
        """
        init
        """
        if isinstance(data, str):
            data = DataGenerator(data)()
            features, labels = data  # todo: 没有label的数据集？
            assert len(features) == len(labels)  # 校验数据
            self.features = features
            self.labels = labels
            self.num_samples = len(features)
        else:
            # 根据数据集单元（单个数据） 生成数据集
            self.num_samples = num_samples
            self.features = data

    def __getitem__(self, idx):
        """
        get item
        """
        if isinstance(self.features, np.ndarray):  # 自定义数据
            feature = self.features * idx
            label = np.ones(1)
            return feature, label
        else:
            # 导入数据集
            feature = np.array(self.features[idx]).astype("float32")
            label = np.array(self.labels[idx]).astype("float32")
            return feature, label

    def __len__(self):
        """
        len
        """
        return self.num_samples


class ChoseSampler(object):
    """
    chose sampler class
    """

    def __init__(self, dataset):
        """
        init
        """
        self.dataset = dataset

    def __call__(self, type, **kwargs):
        """
        call function
        """
        if type == "Random" or type == "RandomSampler":
            # 需要 传递参数
            return self._generate_random()
        elif type == "Sequence" or type == "SequenceSampler":
            return self._generate_sequence()
        elif type == "WeightedRandom" or type == "WeightedRandomSampler":
            # 需要 传递参数
            return self._generate_weighted_random(**kwargs)
        elif type is None:
            return None

    def _generate_random(self):
        """
        generate random sampler
        """
        return paddle.io.RandomSampler(self.dataset)

    def _generate_sequence(self):
        """
        generate sequence sampler
        """
        return paddle.io.SequenceSampler(self.dataset)

    def _generate_weighted_random(self, **kwargs):
        """
        generate sequence sampler
        """
        return paddle.io.WeightedRandomSampler(**kwargs)


class SetBatchSampler(object):
    """
    set batch_sampler class
    """

    def __init__(self, dataset, bs_info):
        """
        init
        """
        self.dataset = dataset
        self.bs_info = bs_info
        self.logger = logger

    def _get_sampler(self):
        """
        get sampler
        """
        sampler_info = self.bs_info.get("sampler")
        # print(sampler_info)
        if sampler_info:
            sampler = ChoseSampler(self.dataset)(**sampler_info)
            return sampler
        else:
            return None

    def _get_batch_sampler(self):
        """
        get batch_sampler
        """
        sampler = self._get_sampler()
        # print(sampler)
        other_bs_info = dict((k, v) for k, v in self.bs_info.items() if k != "sampler")
        if sampler:
            self.logger.get_log().info("this case sets the {} sampler".format(self.bs_info["sampler"].get("type")))
            batch_sampler = paddle.io.BatchSampler(sampler=sampler, **other_bs_info)
        else:
            self.logger.get_log().info("this case has no sampler !!")
            batch_sampler = paddle.io.BatchSampler(dataset=self.dataset, **other_bs_info)
        return batch_sampler

    def __call__(self, *args, **kwargs):
        """
        call function
        """
        return self._get_batch_sampler()


if __name__ == "__main__":
    data = np.array([1, 2, 3, 4])  # 需要传递
    num_samples = 10  # 需要传递
    dataset = GenDataset(data, num_samples)
    for i in range(10):
        print(dataset[i])
