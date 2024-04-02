"""
test laplace
"""
import sys
import paddlescience as psci
import numpy as np
import paddle
import pytest
from tool import verify_converge


def run(static=False):
    """
    run
    """
    if static is True:
        paddle.enable_static()
    elif static is False:
        paddle.disable_static()

    paddle.seed(1)
    np.random.seed(1)

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
    solution = solver.solve(num_epoch=25, checkpoint_freq=20, checkpoint_path="./data/checkpoint/laplace2d_d/")

    psci.visu.save_vtk(filename="./data/vtk/laplace2d_d", geo_disc=pde_disc.geometry, data=solution)

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
    return solution


@pytest.mark.laplace2d
@pytest.mark.skipif(paddle.distributed.get_world_size() != 1, reason="skip serial case")
def test_laplace2d_0():
    """
    test laplace2d
    """
    solution = run()
    verify_converge("/home/sun/ts/pipeline/CE_GPU_ALL/laplace2d_dynamic.npz", solution)


@pytest.mark.laplace2d
@pytest.mark.skipif(paddle.distributed.get_world_size() != 1, reason="skip serial case")
def test_laplace2d_static():
    """
    test laplace2d
    """
    solution = run(static=True)
    verify_converge("/home/sun/ts/pipeline/CE_GPU_ALL/laplace2d_dynamic.npz", solution)


@pytest.mark.laplace2d
@pytest.mark.skipif(paddle.distributed.get_world_size() != 2, reason="skip serial case")
def test_laplace2d_distribute():
    """
    test laplace2d
    """
    solution = run(static=True)
    verify_converge("/home/sun/ts/pipeline/CE_GPU_ALL/laplace2d_distribute.npz", solution)


if __name__ == "__main__":
    code = pytest.main(["-sv", sys.argv[0]])
    sys.exit(code)
