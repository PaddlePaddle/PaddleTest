"""
kovasznay ce test
"""
import paddlescience as psci
import numpy as np
import paddle

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()
# paddle.disable_static()

# constants
Re = 40.0
r = Re / 2 - np.sqrt(Re ** 2 / 4.0 + 4.0 * np.pi ** 2)

# Kovasznay solution
ref_sol_u = lambda x, y: 1.0 - np.exp(r * x) * np.cos(2.0 * np.pi * y)
ref_sol_v = lambda x, y: r / (2 * np.pi) * np.exp(r * x) * np.sin(2.0 * np.pi * y)
ref_sol_p = lambda x, y: 1.0 / 2.0 - 1.0 / 2.0 * np.exp(2.0 * r * x)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(-0.5, -0.5), extent=(1.5, 1.5))
geo.add_boundary(name="boarder", criteria=lambda x, y: (x == -0.5) | (x == 1.5) | (y == -0.5) | (y == 1.5))

# discretization
npoints = 2601
geo_disc = geo.discretize(npoints=npoints, method="uniform")

# N-S equation
pde = psci.pde.NavierStokes(nu=1.0 / Re, rho=1.0, dim=2)

# set boundary condition
bc_border_u = psci.bc.Dirichlet("u", ref_sol_u)
bc_border_v = psci.bc.Dirichlet("v", ref_sol_v)
bc_border_p = psci.bc.Dirichlet("p", ref_sol_p)

# add bounday and boundary condition
pde.add_bc("boarder", bc_border_u)
pde.add_bc("boarder", bc_border_v)
pde.add_bc("boarder", bc_border_p)

# discretization pde
pde_disc = pde.discretize(geo_disc=geo_disc)

# Network
net = psci.network.FCNet(num_ins=2, num_outs=3, num_layers=10, hidden_size=50, activation="tanh")

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
