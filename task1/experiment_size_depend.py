import numpy as np
import scipy

from matplotlib import pyplot as plt
from datetime import datetime

import optimization
import oracles
from plot_trajectory_2d import plot_levels, plot_trajectory

np.random.seed(42)
seeds = np.random.randint(1, 20000, 15)

options = [
    (10,    'r', 'n = 10'),
    (100,   'b', 'n = 100'),
    (1000,  'g', 'n = 1000'),
    (10000, 'm', 'n = 10000')
]
k_vars = np.arange(1, 1001, 10)
start_time = datetime.now()

plt.xlabel('Condition number')
plt.ylabel('Iterations until convergence')

for opt_set in options:
    T_global = np.zeros(len(k_vars))

    for seed in seeds:
        np.random.seed(seed)
        T_curr = []

        for k in k_vars:
            diag = np.random.uniform(1, k, opt_set[0])
            diag[0] = 1
            diag[-1] = k
            A, b = scipy.sparse.diags(diag, 0), np.random.rand(opt_set[0])
            oracle = oracles.QuadraticOracle(A, b)
            _, _, history = optimization.gradient_descent(
                oracle,
                np.zeros(opt_set[0]),
                trace=True
            )
            T_curr.append(len(history['grad_norm']))

        T_global += np.array(T_curr)
        plt.plot(k_vars, T_curr, opt_set[1] + ':', alpha=0.2)

    plt.plot(k_vars, T_global / (1. * len(seeds)), opt_set[1], linewidth=1.3, label=opt_set[2])

print('end , time:', (datetime.now() - start_time))
plt.legend(bbox_to_anchor=(0.33, 1))
plt.savefig('exp2/convergence_iters.png')
