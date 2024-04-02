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
        data = DataGenerator(dir)()
        features, labels = data
        check_len = len(features) == len(self.dataset)
        if check_len:
            logger.get_log().info("dataset length check success !!!")
        else:
            logger.get_log().error(
                "dataset length is {}, but the input length is {}".format(self.dataset, len(features))
            )
            assert False

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
                logger.get_log().error("Label Error in iter {}".format(i))
                logger.get_log().error("dataset is {}, but calculation is {}".format(self.dataset[i][1], labels[i]))
                assert False
        logger.get_log().info("dataset skip check success !!!")


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
        self.length = len(self.dataloader)
        logger.get_log().info("dataloader length is {}".format(self.length))

    def run(self, dataset, dataloader_info):
        """
        run
        """
        self.dataset = dataset
        length = len(self.dataset)
        batch_size, drop_last = dataloader_info.get("batch_size"), dataloader_info.get("drop_last")
        if not drop_last:
            if length % batch_size:
                check_len = length // batch_size + 1 == self.length
            else:
                check_len = length // batch_size == self.length
        else:
            check_len = length // batch_size == self.length

        if check_len:
            logger.get_log().info("dataloader length check success !!!")
        else:
            logger.get_log().error(
                "drop_last: {}, dataset_len: {}, batch_size: {}, but dataloader_len: {}".format(
                    drop_last, length, batch_size, self.length
                )
            )
            assert False

        if self.length <= 100:
            self._obo_check(dataloader_info)
        else:
            self._skip_check(dataloader_info)

    def _obo_check(self, info):
        """
        check data one by one
        """
        batch_size, shuffle = info.get("batch_size"), info.get("shuffle")
        sample_type = ""
        if info.get("sampler"):
            sample_type = info.get("sampler").get("type")

        if not shuffle and (sample_type != "RandomSampler" and "WeightedRandomSampler"):
            for i, item in enumerate(self.dataloader):
                for j in range(len(item[0])):
                    idx = i * batch_size + j  # 对应的dataset数据的索引
                    jud_f = np.array_equal(self.dataset[idx][0], item[0][j].numpy())  # 检查数据集特征
                    jud_l = np.array_equal(self.dataset[idx][0], item[0][j].numpy())  # 检查数据集标签
                    if not jud_f:
                        logger.get_log().error("Feature error occurred at {} iteration, {} unit".format(i, j))
                        logger.get_log().error(
                            "dataset is {}, but dataloader is {}".format(self.dataset[idx][0], item[0][j].numpy())
                        )
                        assert False
                    if not jud_l:
                        logger.get_log().error("Label error occurred at {} iteration, {} unit".format(i, j))
                        logger.get_log().error(
                            "dataset is {}, but dataloader is {}".format(self.dataset[idx][0], item[0][j].numpy())
                        )
                        assert False
            logger.get_log().info("dataloader check success !!!")

        else:
            # TODO: 引入随机或shuffle之后的dataloader检查
            pass

    def _skip_check(self, info):
        """
        check data skip
        """
        batch_size, shuffle = info.get("batch_size"), info.get("shuffle")
        sample_type = ""
        if info.get("sampler"):
            sample_type = info.get("sampler").get("type")
        # 抽样check
        step = self.length // 100
        if not shuffle and (sample_type != "RandomSampler" and "WeightedRandomSampler"):
            for i, item in enumerate(self.dataloader):
                if i % step == 0:
                    for j in range(len(item[0])):
                        idx = i * batch_size + j  # 对应的dataset数据的索引
                        jud_f = np.array_equal(self.dataset[idx][0], item[0][j].numpy())  # 检查数据集特征
                        jud_l = np.array_equal(self.dataset[idx][0], item[0][j].numpy())  # 检查数据集标签
                        if not jud_f:
                            logger.get_log().error("Feature error occurred at {} iteration, {} unit".format(i, j))
                            logger.get_log().error(
                                "dataset is {}, but dataloader is {}".format(self.dataset[idx][0], item[0][j].numpy())
                            )
                            assert False
                        if not jud_l:
                            logger.get_log().error("Label error occurred at {} iteration, {} unit".format(i, j))
                            logger.get_log().error(
                                "dataset is {}, but dataloader is {}".format(self.dataset[idx][0], item[0][j].numpy())
                            )
                            assert False
            logger.get_log().info("dataloader skip check success !!!")

        else:
            pass
