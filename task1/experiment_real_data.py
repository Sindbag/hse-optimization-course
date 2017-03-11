import numpy as np
from matplotlib import pyplot as plt
import scipy
import time

import optimization
import oracles

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


def compare_speed(svm_file, save_to_file):
    A, b = load_svmlight_file(svm_file)
    A, _, b, _ = train_test_split(A, b, test_size=0.7)
    rc = 1.0 / b.size
    oracle = oracles.create_log_reg_oracle(A, b, rc)
    _, _, history_gd = optimization.gradient_descent(oracle,
                                                     np.zeros(A.shape[1]),
                                                     trace=True)

    plt.xlabel('Time, sec')
    plt.ylabel('Value')

    plt.plot(history_gd['time'],
             history_gd['func'],
             label="Gradient Descent")

    _, _, history_n = optimization.newton(oracle,
                                          np.zeros(A.shape[1]),
                                          display=True,
                                          trace=True)
    plt.plot(history_n['time'],
             history_n['func'],
             label="Newton")

    plt.legend()
    plt.savefig(save_to_file + '_func.png')
    plt.clf()

    grad = oracle.grad(np.zeros(A.shape[1]))
    grad_0_norm = 1.0 / grad.dot(grad)

    plt.plot(history_gd['time'],
             np.square(np.array(history_gd['grad_norm'])) * grad_0_norm,
             label="Gradient Descent")
    plt.plot(history_n['time'],
             np.square(np.array(history_n['grad_norm'])) * grad_0_norm,
             label="Newton")
    plt.yscale('log')
    plt.xlabel('Time, sec')
    plt.ylabel(r'$log  \| \nabla f(x_k) \|_2^2 / \| \nabla f(x_0) \|_2^2$')
    plt.legend()
    plt.savefig(save_to_file + '_grad.png')
    plt.clf()


# compare_speed('sources/w8a', 'exp3/w8a_res')
# compare_speed('sources/gisette_scale.bz2', 'exp3/gisette_scale_res')
compare_speed('sources/real-sim.bz2', 'exp3/real-sim_res')
