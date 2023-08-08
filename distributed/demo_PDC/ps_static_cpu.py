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
import math
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker

paddle.enable_static()
role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)
cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000000
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class WideDeepLayer(nn.Layer):
    """
    WideDeepLayer
    """

    def __init__(self, sparse_feature_number, sparse_feature_dim, dense_feature_dim, num_field, layer_sizes):
        """
        __init__
        """
        super(WideDeepLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes
        self.wide_part = paddle.nn.Linear(
            in_features=self.dense_feature_dim,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=1.0 / math.sqrt(self.dense_feature_dim))
            ),
        )
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(name="SparseFeatFactors", initializer=paddle.nn.initializer.Uniform()),
        )
        sizes = [sparse_feature_dim * num_field + dense_feature_dim] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(sizes[i]))),
            )
            self.add_sublayer("linear_%d" % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == "relu":
                act = paddle.nn.ReLU()
                self.add_sublayer("act_%d" % i, act)
                self._mlp_layers.append(act)

    def forward(self, sparse_inputs, dense_inputs):
        """
        forward
        """
        # wide part
        wide_output = self.wide_part(dense_inputs)
        # deep part
        sparse_embs = []
        for s_input in sparse_inputs:
            emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            sparse_embs.append(emb)
        deep_output = paddle.concat(x=sparse_embs + [dense_inputs], axis=1)
        for n_layer in self._mlp_layers:
            deep_output = n_layer(deep_output)
        prediction = paddle.add(x=wide_output, y=deep_output)
        pred = F.sigmoid(prediction)
        return pred


class WideDeepModel:
    """
    WideDeepModel
    """

    def __init__(
        self,
        sparse_feature_number=1000001,
        sparse_inputs_slots=27,
        sparse_feature_dim=10,
        dense_input_dim=13,
        fc_sizes=[400, 400, 400],
    ):
        """
        __init__
        """
        self.sparse_feature_number = sparse_feature_number
        self.sparse_inputs_slots = sparse_inputs_slots
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_input_dim = dense_input_dim
        self.fc_sizes = fc_sizes
        self._metrics = {}

    def net(self, is_train=True):
        """
        net
        """
        dense_input = paddle.static.data(name="dense_input", shape=[None, self.dense_input_dim], dtype="float32")
        sparse_inputs = [
            paddle.static.data(name="C" + str(i), shape=[None, 1], lod_level=1, dtype="int64")
            for i in range(1, self.sparse_inputs_slots)
        ]
        label_input = paddle.static.data(name="label", shape=[None, 1], dtype="int64")
        self.inputs = [dense_input] + sparse_inputs + [label_input]
        self.loader = paddle.fluid.reader.DataLoader.from_generator(feed_list=self.inputs, capacity=64, iterable=False)
        wide_deep_model = WideDeepLayer(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            self.dense_input_dim,
            self.sparse_inputs_slots - 1,
            self.fc_sizes,
        )
        pred = wide_deep_model.forward(sparse_inputs, dense_input)
        # predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        label_float = paddle.cast(label_input, dtype="float32")
        # loss
        cost = paddle.nn.functional.log_loss(input=pred, label=label_float)
        avg_cost = paddle.mean(x=cost)
        self.cost = avg_cost


class WideDeepDataset:
    """
    WideDeepDataset
    """

    def __init__(self, size):
        """
        __init__
        """
        self.size = size

    def line_process(self, i):
        """
        line_process
        """
        dense_feature = []
        sparse_feature = []
        for idx in continuous_range_:
            dense_feature.append(
                (float(random.randint(cont_min_[idx - 1], cont_max_[idx - 1])) - cont_min_[idx - 1])
                / cont_diff_[idx - 1]
            )
        for idx in categorical_range_:
            sparse_feature.append([random.randint(0, hash_dim_)])
        label = [random.randint(0, 1)]
        return [dense_feature] + sparse_feature + [label]

    def __call__(self):
        """
        __call__
        """
        for i in range(self.size):
            input_data = self.line_process(i)
            yield input_data


def distributed_training(exe, train_model, batch_size=10, epoch_num=100):
    """
    distributed_training
    """
    train_data = WideDeepDataset(400)
    reader = train_model.loader.set_sample_generator(
        train_data, batch_size=batch_size, drop_last=True, places=paddle.CPUPlace()
    )
    for epoch_id in range(epoch_num):
        reader.start()
        try:
            while True:
                loss_val = exe.run(program=paddle.static.default_main_program(), fetch_list=[train_model.cost.name])
                loss_val = np.mean(loss_val)
                print("TRAIN ---> pass: {} loss: {}\n".format(epoch_id, loss_val))
        except paddle.common_ops_import.core.EOFException:
            reader.reset()


fleet.init(is_collective=False)
model = WideDeepModel()
model.net(is_train=True)
optimizer = paddle.optimizer.SGD(learning_rate=0.0001)
strategy = fleet.DistributedStrategy()
strategy.a_sync = True
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(model.cost)
if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
if fleet.is_worker():
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    fleet.init_worker()
    distributed_training(exe, model)
    fleet.stop_worker()
fleet.util.barrier()
