"""
darcy2d example test
"""
import paddlescience as psci
import numpy as np
import paddle

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()
# paddle.disable_static()

psci.config.set_dtype("float32")

# ref solution
ref_sol = lambda x, y: np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

# ref rhs
ref_rhs = lambda x, y: 8.0 * np.pi ** 2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))

geo.add_boundary(name="top", criteria=lambda x, y: y == 1.0)
geo.add_boundary(name="down", criteria=lambda x, y: y == 0.0)
geo.add_boundary(name="left", criteria=lambda x, y: x == 0.0)
geo.add_boundary(name="right", criteria=lambda x, y: x == 1.0)

# discretize geometry
npoints = 10201
geo_disc = geo.discretize(npoints=npoints, method="uniform")

# Poisson
pde = psci.pde.Poisson(dim=2, rhs=ref_rhs)

# set bounday condition
bc_top = psci.bc.Dirichlet("u", rhs=ref_sol)
bc_down = psci.bc.Dirichlet("u", rhs=ref_sol)
bc_left = psci.bc.Dirichlet("u", rhs=ref_sol)
bc_right = psci.bc.Dirichlet("u", rhs=ref_sol)

# add bounday and boundary condition
pde.add_bc("top", bc_top)
pde.add_bc("down", bc_down)
pde.add_bc("left", bc_left)
pde.add_bc("right", bc_right)

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

assert mse < 3, "The accuracy of mse is not enough " % mse
