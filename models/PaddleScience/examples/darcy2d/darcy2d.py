"""
darcy2d example test
"""

import paddlescience as psci
import numpy as np
import paddle


# Analytical solution
def DarcyRecSolution(x, y):
    """
    DarcyRecSolution
    """
    return np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


# Generate analytical Solution using Geometry points
def GenSolution(xy, bc_index):
    """
    GenSolution
    """
    sol = np.zeros((len(xy), 1)).astype(np.float32)
    bc_value = np.zeros((len(bc_index), 1)).astype(np.float32)
    len1 = len(xy)
    for i in range(len1):
        sol[i][0] = DarcyRecSolution(xy[i][0], xy[i][1])

    len2 = len(bc_index)
    for i in range(len2):
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
pdes, geo = psci.discretize(pdes, geo, space_nsteps=(11, 11))

# bc value
golden, bc_value = GenSolution(geo.get_space_domain(), geo.get_bc_index())
pdes.set_bc_value(bc_value=bc_value)
psci.visu.save_vtk(geo, golden, "golden_darcy_2d")
np.save("./golden_darcy_2d.npy", golden)

# Network
net = psci.network.FCNet(num_ins=2, num_outs=1, num_layers=5, hidden_size=20, dtype="float32", activation="tanh")

# Loss, TO rename
loss = psci.loss.L2(pdes=pdes, geo=geo, aux_func=RighthandBatch)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=30000)

# Use solution
rslt = solution(geo)
psci.visu.save_vtk(geo, rslt, "rslt_darcy_2d")
np.save("./rslt_darcy_2d.npy", rslt)

# Calculate diff and l2 relative error
diff = rslt - golden

psci.visu.save_vtk(geo, diff, "diff_darcy_2d")
np.save("./diff_darcy_2d.npy", diff)
root_square_error = np.linalg.norm(diff, ord=2)
mean_square_error = root_square_error * root_square_error / geo.get_domain_size()
print("mean_sqeare_error: ", mean_square_error)

assert mean_square_error < 0.00085, (
    "The accuracy of mean_square_error is not enough;" "\n mean_square_error: %f" % mean_square_error
)
