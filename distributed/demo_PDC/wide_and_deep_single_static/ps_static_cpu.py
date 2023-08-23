"""
PS static cpu demo
export CPU_NUM=2
单机server_num:1, worker_num:1
python -m paddle.distributed.launch --server_num=1 --log_dir=pserver_log --worker_num=1 ps_static_cpu.py
单机server_num:1, worker_num:2
python -m paddle.distributed.launch --server_num=1 --log_dir=pserver_log --worker_num=2 ps_static_cpu.py
单机server_num:2, worker_num:2
python -m paddle.distributed.launch --server_num=2 --log_dir=pserver_log --worker_num=2 ps_static_cpu.py
单机server_num:2, worker_num:1
python -m paddle.distributed.launch --server_num=2 --log_dir=pserver_log --worker_num=1 ps_static_cpu.py
多机server_num:2, worker_num:2
python -m paddle.distributed.launch --servers="xx.xx.xx.xx:port1,yy.yy.yy.yy:port2" \
--workers="xx.xx.xx.xx,yy.yy.yy.yy" ps_static_cpu.py
"""
import os
import sys
import time
import paddle
import paddle.nn as nn
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
from model import WideDeepModel
from reader import WideDeepDataset

paddle.enable_static()
role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)


def distributed_training(exe, model, place):
    """
    distributed_training
    """
    # 数据加载
    data_dir = "./data"
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    dataset = WideDeepDataset(file_list)
    loader = paddle.io.DataLoader(dataset, batch_size=1, places=place, drop_last=True)

    input_data_names = [var.name for var in model.inputs]
    epochs = 1
    print_interval = 1
    # 开始训练
    for epoch_id in range(epochs):
        for batch_id, batch in enumerate(loader()):
            fetch_batch_var = exe.run(
                program=paddle.static.default_main_program(),
                feed=dict(zip(input_data_names, batch)),
                fetch_list=[model.loss.name],
            )
            if batch_id % print_interval == 0:
                print("[Epoch %d, batch %d] loss: %.5f" % (epoch_id, batch_id, fetch_batch_var[0]))


if __name__ == "__main__":
    fleet.init(is_collective=False)
    model = WideDeepModel()
    model.net(is_train=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.0001)
    strategy = fleet.DistributedStrategy()
    strategy.a_sync = True
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(model.loss)
    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
    if fleet.is_worker():
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        fleet.init_worker()
        distributed_training(exe, model, place)
        fleet.stop_worker()
    fleet.util.barrier()
