"""
TOOL
"""
import numpy as np


def verify_converge(file, rslt, threshold=1e-5):
    """
    verify convergence
    """
    data = np.load(file, allow_pickle=True)
    solution, mse = data.get("rslt"), data.get("mse")
    print(mse)
    length = len(solution)
    assert length == len(rslt)
    standard_value, run_value = solution[0], rslt[0]
    for i in range(1, length):
        standard_value = np.vstack((standard_value, solution[i]))
        run_value = np.vstack((run_value, rslt[i]))

    diff = standard_value - run_value
    norm = np.linalg.norm(diff, ord=2)
    error = norm / standard_value.shape[0]

    assert error <= threshold
