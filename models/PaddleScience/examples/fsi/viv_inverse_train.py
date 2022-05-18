"""
fsi train ce test
"""


import copy
import time
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

import paddle
import paddle.nn as nn
import paddle.distributed as dist
import paddlescience as psci
import paddlescience.module.fsi.viv_pinn_solver as psolver

from dataset import Dataset


paddle.disable_static()


def train(net_params=None):
    """
    train
    """
    Neta = 100
    N_train = 100
    t_range = [0.0625, 10]

    data = Dataset(t_range, Neta, N_train)

    # inputdata
    t_eta, eta, t_f, f, tmin, tmax = data.build_data()

    PINN = psolver.PysicsInformedNeuralNetwork(
        layers=6,
        hidden_size=30,
        num_ins=1,
        num_outs=1,
        t_max=tmax,
        t_min=tmin,
        N_f=f.shape[0],
        checkpoint_path="./checkpoint/",
        net_params=net_params,
    )
    PINN.set_eta_data(X=(t_eta, eta))
    PINN.set_f_data(X=(t_f, f))

    # Training
    batchsize = 150
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=1e-3, step_size=20000, gamma=0.9)
    adm_opt = paddle.optimizer.Adam(scheduler, weight_decay=None, parameters=PINN.net.parameters())
    PINN.train(num_epoch=100, batchsize=batchsize, optimizer=adm_opt, scheduler=scheduler)
    adm_opt = psci.optimizer.Adam(learning_rate=1e-5, weight_decay=None, parameters=PINN.net.parameters())
    PINN.train(num_epoch=100, batchsize=batchsize, optimizer=adm_opt)


if __name__ == "__main__":
    net_params = None
    train(net_params=net_params)
