"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import pytest
import paddlescience as psci
import numpy as np
import paddle


# Analytical solution
def DarcyRecSolution(x, y):
    """
    Dirichlet boundary condition
    """
    return np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


# Generate analytical Solution using Geometry points
def GenSolution(xy, bc_index):
    """
    GenSolution
    """
    sol = np.zeros((len(xy), 1)).astype(np.float32)
    bc_value = np.zeros((len(bc_index), 1)).astype(np.float32)
    length1 = len(xy)
    for i in range(length1):
        sol[i][0] = DarcyRecSolution(xy[i][0], xy[i][1])

    length2 = len(bc_index)
    for i in range(length2):
        bc_value[i][0] = sol[bc_index[i]]

    return [sol, bc_value]


# right-hand side
def RighthandBatch(xy):
    """
    RighthandBatch
    """
    return [8.0 * 3.1415926 * 3.1415926 * paddle.sin(2.0 * np.pi * xy[:, 0]) * paddle.cos(2.0 * np.pi * xy[:, 1])]


# Geometry
geo = psci.geometry.Rectangular(space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))

# PDE Laplace
pdes = psci.pde.Laplace2D()

# Discretization
pdes, geo = psci.discretize(pdes, geo, space_nsteps=(4, 4))

# bc value
golden, bc_value = GenSolution(geo.space_domain, geo.bc_index)
pdes.set_bc_value(bc_value=bc_value)
# psci.visu.save_vtk(geo, golden, 'golden_darcy_2d')
# np.save('./golden_darcy_2d.npy', golden)

# Network
net = psci.network.FCNet(num_ins=2, num_outs=1, num_layers=2, hidden_size=1, dtype="float32", activation="tanh")

net._parameters["w_0"].set_value(paddle.to_tensor([[1], [1]]).astype("float32"))
net._parameters["w_1"].set_value(paddle.to_tensor([[1]]).astype("float32"))
# print(net.w_0)
# print(net._parameters)

# Loss, TO rename
loss = psci.loss.L2(pdes=pdes, geo=geo, aux_func=RighthandBatch)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=10)

# Use solution
rslt = solution(geo)
# print(rslt)
# psci.visu.save_vtk(geo, rslt, 'rslt_darcy_2d')
# np.save('./rslt_darcy_2d.npy', rslt)

# Calculate diff and l2 relative error
diff = rslt - golden
# psci.visu.save_vtk(geo, diff, 'diff_darcy_2d')
# np.save('./diff_darcy_2d.npy', diff)
root_square_error = np.linalg.norm(diff, ord=2)
mean_square_error = root_square_error * root_square_error / geo.get_domain_size()
# print("mean_sqeare_error: ", mean_square_error)


@pytest.mark.ldc2d
def test_darcy2D():
    """
    test darcy2d
    """
    expect = np.array(
        [
            [-0.02001287],
            [0.29630685],
            [0.5558747],
            [0.73548657],
            [0.2963115],
            [0.5558783],
            [0.7354888],
            [0.84560883],
            [0.55588174],
            [0.7354911],
            [0.84561014],
            [0.9081895],
            [0.7354933],
            [0.8456115],
            [0.90819025],
            [0.94222784],
        ]
    )
    diff_expect = 0.750401496887207
    assert np.allclose(rslt, expect)
    assert np.isclose(diff_expect, mean_square_error)
