import torch
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from torchsummary import summary
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=['save', 'load'])
    parser.add_argument("--content")
    args = parser.parse_args()
    return args

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
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
    sgd = SGD(model.parameters(), lr=0.001, momentum=0.9)
    if save_type == 'net':
        obj_net = {'model': model.state_dict()}
        torch.save(obj_net, 'torch_net.pdparams')
        print(' save path : %s/torch_net.pdparams  ' %(pwd))
    elif save_type == 'params':
        obj_params = {'opt': sgd.state_dict(), 'epoch': 100}
        torch.save(obj_params, 'torch_params.pdparams')
        print(' save path : %s/torch_params.pdparams  ' %(pwd))
    elif save_type == 'model':
        obj_model = {'model': model.state_dict(),'opt': sgd.state_dict(), 'epoch': 100}
        torch.save(obj_model, 'torch_model.pdparams')
        print(' save path : %s/torch_model.pdparams  ' %(pwd))
    else:
        print('### edit code load your own model , your save_type= ',save_type)

def load_net(save_type=None):
    pwd = sys.path[0]
    if save_type == 'net':
        load_main = torch.load("torch_net.pdparams")
        print(' load path : %s/torch_net.pdparams  ' %(pwd))
    elif save_type == 'params':
        load_main = torch.load("torch_params.pdparams")
        print(' load path : %s/torch_params.pdparams  ' %(pwd))
    elif save_type == 'model':
        load_main = torch.load("torch_model.pdparams")
        print(' load path : %s/torch_model.pdparams  ' %(pwd))
    else:
        print('### edit code load your own model , your save_type= ',save_type)
    # print(load_main)
    # summary(load_main, (3, 224, 224))


if __name__ == '__main__':
    args = parse_args()
    if args.action == 'save':
            save_net(args.content)

    if args.action == 'load':
            load_net(args.content)
