"""
laplace2d example test
"""

import paddlescience as psci
import numpy as np
import paddle

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()
# paddle.disable_static()

# analytical solution
ref_sol = lambda x, y: np.cos(x) * np.cosh(y)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
geo.add_boundary(name="around", criteria=lambda x, y: (y == 1.0) | (y == 0.0) | (x == 0.0) | (x == 1.0))

# discretize geometry
npoints = 10201
geo_disc = geo.discretize(npoints=npoints, method="uniform")

# Laplace
pde = psci.pde.Laplace(dim=2)

# set bounday condition
bc_around = psci.bc.Dirichlet("u", rhs=ref_sol)

# add bounday and boundary condition
pde.add_bc("around", bc_around)

# discretization pde
pde_disc = pde.discretize(geo_disc=geo_disc)

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(num_ins=2, num_outs=1, num_layers=5, hidden_size=20, activation="tanh")

# Loss
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=10000)

psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)

# MSE
# TODO: solution array to dict: interior, bc
cord = pde_disc.geometry.interior
ref = ref_sol(cord[:, 0], cord[:, 1])
mse2 = np.linalg.norm(solution[0][:, 0] - ref, ord=2) ** 2

n = 1
for cord in pde_disc.geometry.boundary.values():
    ref = ref_sol(cord[:, 0], cord[:, 1])
    mse2 += np.linalg.norm(solution[n][:, 0] - ref, ord=2) ** 2
    n += 1

mse = mse2 / npoints

print("MSE is: ", mse)

assert mse < 0.001, "The accuracy of mean_square_error is not enough;" "\n mse: %f" % mse
