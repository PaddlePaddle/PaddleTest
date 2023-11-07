"""
collective gpu demo
数据准备(数据集放到train_data目录下)
wget https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz
wget https://dataset.bj.bcebos.com/mnist/train-labels-idx1-ubyte.gz
单机单卡:python -m paddle.distributed.launch --gpus 0 paddle_collective_gpu_mnist_demo.py
单机八卡:python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 paddle_collective_gpu_mnist_demo.py
两机两卡:python -m paddle.distributed.launch --ips="xx.xx.xx.xx,xx.xx.xx.xx" --gpus 0 paddle_collective_gpu_mnist_demo.py
两机16卡:python -m paddle.distributed.launch --ips="xx.xx.xx.xx,xx.xx.xx.xx" --gpus 0,1,2,3,4,5,6,7
paddle_collective_gpu_mnist_demo.py
"""
import paddle
import paddle.optimizer as opt
import paddle.distributed as dist
import paddle.nn as nn
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.io import DistributedBatchSampler

BATCH_SIZE = 4


class MNIST_net(nn.Layer):
    """MNIST_net"""

    def __init__(self):
        super(MNIST_net, self).__init__()
        self._conv2d_1 = Conv2D(in_channels=BATCH_SIZE, out_channels=20, kernel_size=5)
        self._pool2d_1 = MaxPool2D(kernel_size=2, stride=2)
        self._conv2d_2 = Conv2D(in_channels=20, out_channels=50, kernel_size=5)
        self._pool2d_2 = MaxPool2D(kernel_size=2, stride=2)
        self._fc = Linear(
            int(800 / BATCH_SIZE),
            10,
        )
        self._softmax = nn.Softmax(axis=1)
        self._ce = nn.CrossEntropyLoss()

    def forward(self, inputs, label):
        """forward"""
        x = self._conv2d_1(inputs)
        x = self._pool2d_1(x)
        x = self._conv2d_2(x)
        x = self._pool2d_2(x)
        x = paddle.reshape(x, shape=[BATCH_SIZE, int(800 / BATCH_SIZE)])
        x = self._fc(x)
        logit = self._softmax(x)
        loss = self._ce(logit, label)
        avg_loss = loss.mean()
        return avg_loss


paddle.set_device("gpu:%d" % paddle.distributed.ParallelEnv().dev_id)
dist.init_parallel_env()
model = MNIST_net()
model = paddle.DataParallel(model, find_unused_parameters=False)
train_data_path = "./train_data/train-images-idx3-ubyte.gz"
train_data_label = "./train_data/train-labels-idx1-ubyte.gz"
paddle.vision.set_image_backend("cv2")
train_data = paddle.vision.datasets.MNIST(image_path=train_data_path, label_path=train_data_label, mode="train")
train_batch_sampler = paddle.io.DistributedBatchSampler(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
train_reader = paddle.io.DataLoader(dataset=train_data, batch_sampler=train_batch_sampler)
adam = opt.Adam(learning_rate=1e-3, parameters=model.parameters())
for step_id, (images, labels) in enumerate(train_reader()):
    images = paddle.reshape(images, shape=[1, BATCH_SIZE, 28, 28])
    images = images / 255
    labels.stop_gradient = True
    avg_loss = model(images, labels)
    avg_loss.backward()
    adam.step()
    model.clear_gradients()
    if step_id % 10 == 0:
        print("step: %d, loss: %f" % (step_id, avg_loss.numpy()))
