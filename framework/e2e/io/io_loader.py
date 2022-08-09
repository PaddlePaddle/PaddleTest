"""
io_loader
"""
import paddle
from utils.logger import Logger

logger = Logger("data_loader")


class GenDataLoader(object):
    """
    generate DataLoader class
    """

    def __init__(self, dataset, **kwargs):
        """
        init
        """
        self.dataset = dataset  # 可能会对dataset做处理
        self.kwargs = kwargs
        self.logger = logger

    def exec(self, batch_sampler):
        """
        execute
        """
        if batch_sampler:
            self.logger.get_log().info("this case sets the batch_sampler")
        else:
            self.logger.get_log().info("this case has no batch_sampler")
        dataloader = paddle.io.DataLoader(self.dataset, batch_sampler=batch_sampler, **self.kwargs)
        return dataloader
