"""
io_trans
"""
import copy
from utils.weaktrans import WeakTrans, Framework
from utils.logger import Logger
import numpy as np
import paddle

logger = Logger("data_info_trans")


class DataLoaderTrans(WeakTrans):
    """
    DataLoader transform
    """

    def __init__(self, case):
        """initialize"""
        super().__init__(case)
        self.eval_str = None
        self.logger = logger
        self.case = case["info"]
        self.dataset = self.case.get("dataset")
        self.params = self.case.get("params")
        self.seed = self.case.get("seed")
        if self.seed:
            np.random.seed(self.seed)
            paddle.seed(self.seed)
            self.logger.get_log().info("set random seed: {}".format(self.seed))

    def get_dataset(self):
        """
        get dataset
        """
        if self.dataset.get("generate_way"):
            self.logger.get_log().info("loading dataset from {}".format(self.dataset.get("generate_way")))
            return self.dataset.get("generate_way")
        elif self.dataset.get("data"):
            params = self.dataset.get("data")
            self.logger.get_log().info("dataset unit info: {}".format(params))
            return self._reload_randtools(params)

    def get_batch_sampler(self):
        """
        get batch_sampler
        """
        batch_sampler_info = self.params.get("batch_sampler")
        self.logger.get_log().info("batch_sampler info: {}".format(batch_sampler_info))
        return batch_sampler_info

    def get_other_params(self):
        """
        get other params
        """
        params = copy.deepcopy(self.params)
        if params.get("batch_sampler"):
            params.pop("batch_sampler")
        self.logger.get_log().info("other dataloader info: {}".format(params))
        return params

    def _reload_randtools(self, params):
        """
        randtools
        """
        if params.get("type"):
            params.pop("type")
        if params.get("range"):
            params["low"] = params["range"][0]
            params["high"] = params["range"][1]
            params.pop("range")
        else:
            params["low"] = 0
            params["high"] = 1

        return super(DataLoaderTrans, self)._randtool(**params)
