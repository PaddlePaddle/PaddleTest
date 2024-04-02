import paddle
from paddle import nn
from paddle.optimizer import Adam
import paddle.nn.functional as F
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=['save', 'load'])
    parser.add_argument("--content")
    args = parser.parse_args()
    return args

class TheModelClass(nn.Layer):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2D(3, 6, 5)
        self.pool = nn.MaxPool2D(2, 2)
        self.conv2 = nn.Conv2D(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def save_net(save_type=None):
    pwd = sys.path[0]
    model = TheModelClass()
    adam = Adam(learning_rate=0.001, parameters=model.parameters())
    if save_type == 'net':
        obj_net = {'model': model.state_dict()}
        paddle.save(obj_net, 'paddle_net.pdparams')
        print(' save path : %s/paddle_net.pdparams  ' %{pwd})
    elif save_type == 'params':
        obj_params = {'opt': adam.state_dict(), 'epoch': 100}
        paddle.save(obj_params, 'paddle_params.pdparams')
        print(' save path : %s/paddle_params.pdparams  ' %{pwd})
    elif save_type == 'model':
        obj_model = {'model': model.state_dict(),'opt': adam.state_dict(), 'epoch': 100}
        paddle.save(obj_model, 'paddle_model.pdparams')
        print(' save path : %s/paddle_model.pdparams  ' %{pwd})
    else:
        print('### edit code load your own model , your save_type= ',save_type)

def load_net(save_type=None):
    pwd = sys.path[0]
    if save_type == 'net':
        load_main = paddle.load("paddle_net.pdparams")
        print(' load path : %s/paddle_net.pdparams  ' %{pwd})
    elif save_type == 'params':
        load_main = paddle.load("paddle_params.pdparams")
        print(' load path : %s/paddle_params.pdparams  ' %{pwd})
    elif save_type == 'model':
        load_main = paddle.load("paddle_model.pdparams")
        print(' load path : %s/paddle_model.pdparams  ' %{pwd})
    else:
        print('### edit code load your own model , your save_type= ',save_type)
    # print('load_main')
    # paddle.summary(load_main, (-1, 3, 224, 224))


if __name__ == '__main__':
    args = parse_args()
    if args.action == 'save':
            save_net(args.content)

    if args.action == 'load':
            load_net(args.content)
