"""
io_test
"""
import numpy as np
from utils.logger import Logger
from io_reader import DataGenerator

logger = Logger("io_test")

# check dataset
class TestDataset(object):
    """
    test Dataset class
    """

    def __init__(self, dataset):
        """
        init
        """
        self.dataset = dataset
        self.logger = logger
        self.length = len(self.dataset)

    def run(self, test_info):
        """
        run
        """
        if isinstance(test_info, str):
            self._check_load(test_info)
        else:
            self._check_unit(test_info)

    def _check_unit(self, info):
        """
        check unit case
        """
        n, unit = info
        check_len = n == self.length
        if check_len:
            logger.get_log().info("dataset length check success !!!")
        else:
            logger.get_log().error("dataset length is {}, but the input length is {}".format(self.dataset, n))

        for i in range(self.length):
            # print(self.dataset[i])
            # print(unit * i)
            jud = np.array_equal(self.dataset[i][0], unit * i)  # todo: 是否需要label比较（支持iterable（tennsor）类型）
            if not jud:
                logger.get_log().error("Error in unit {}".format(i))
                logger.get_log().error("dataset is {}, but calculation is {}".format(self.dataset[i], unit * i))
                assert False
        logger.get_log().info("dataset check success !!!")

    def _check_load(self, dir):
        """
        check load case
        """
        # todo: 检查下载的数据生成的dataset
        data = DataGenerator(dir)()
        features, labels = data
        check_len = len(features) == len(self.dataset)
        if check_len:
            logger.get_log().info("dataset length check success !!!")
        else:
            logger.get_log().error(
                "dataset length is {}, but the input length is {}".format(self.dataset, len(features))
            )

        # 抽样check
        step = len(self.dataset) // 100
        for i in range(0, len(self.dataset), step):
            jud_f = np.allclose(features[i], self.dataset[i][0])
            jud_l = np.allclose(labels[i], self.dataset[i][1])
            if not jud_f:
                logger.get_log().error("Feature Error in iter {}".format(i))
                logger.get_log().error("dataset is {}, but calculation is {}".format(self.dataset[i][0], features[i]))
                assert False
            if not jud_l:
                logger.get_log().error("Feature Error in iter {}".format(i))
                logger.get_log().error("dataset is {}, but calculation is {}".format(self.dataset[i][1], labels[i]))
                assert False
        logger.get_log().info("dataset check success !!!")


# check dataloader
class TestDataLoader(object):
    """
    test DataLoader class
    """

    def __init__(self, dataloader):
        """
        init
        """
        self.dataloader = dataloader

    def run(self, dataset):
        """
        run
        """
        check_len = len(dataset) == len(self.dataloader)
        if check_len:
            logger.get_log().info("dataloader length check success !!!")
        else:
            logger.get_log().error("dataset length is {}, but the dataloader is {}".format(dataset, self.dataloader))
