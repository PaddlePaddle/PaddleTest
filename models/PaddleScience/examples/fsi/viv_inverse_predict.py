"""
fsi predict ce test
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


np.random.seed(1234)
paddle.disable_static()


def predict(net_params=None):
    """
    train
    """
    Neta = 100
    N_train = 150
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
        mode="predict",
    )

    PINN.set_eta_data(X=(t_eta, eta))
    PINN.set_f_data(X=(t_f, f))
    eta_pred, f_pred = PINN.predict((-4.0, 0.0))

    error_f = np.linalg.norm(f.reshape([-1]) - f_pred.numpy().reshape([-1]), 2) / np.linalg.norm(f, 2)
    error_eta = np.linalg.norm(eta.reshape([-1]) - eta_pred.numpy().reshape([-1]), 2) / np.linalg.norm(eta, 2)
    print("------------------------")
    print("Error f: %e" % (error_f))
    print("Error eta: %e" % (error_eta))
    print("------------------------")


if __name__ == "__main__":
    net_params = "./checkpoint/net_params_100000"
    predict(net_params=net_params)
