import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from IMC import IMC
from dirtyIMC import dirtyIMC
from Utils import *
import matplotlib.pyplot as plt


def do_test_imc(k = 20, d1 = 40, d2 = 40, n1 = 200, n2 = 200, sparsity = 0.3, noisy_level = 0.2):

    seed = 1
    lamb = 1e-3
    maxiter = 100
    # print 'Generating random data...'
    X, Y, A, R = generate_synthesized_noisy_data(k, d1, d2, n1, n2, sparsity, noisy_level, seed)
    W0 = randn(d1, k)
    H0 = randn(d2, k)

# Run IMC
    W, H, losses = IMC(R, X, Y, k, lamb, maxiter, W0, H0)

    #relerr = norm(W.T.dot(H) - Z, 'fro')**2 / norm(Z, 'fro')**2 * 100
    relerr = norm(X.dot(W.T).dot(H).dot(Y.T) - A, 'fro') ** 2 / norm(A, 'fro') ** 2 * 100
    print 'IMC RelErr = %g'%(relerr)
    # plt.plot(losses)
    # plt.yscale('log')
    # plt.show()

    return relerr

def do_test_dirty_imc(k1 = 20, k2 = 20, d1 = 40, d2 = 40, n1 = 200, n2 = 200, sparsity = 0.3, noisy_level = 0.2):

    seed = 1
    lamb1 = 1e-3
    lamb2 = 1e-3
    maxiter = 100
    # print 'Generating random data...'
    X, Y, A, R = generate_synthesized_noisy_data(k1, d1, d2, n1, n2, sparsity, noisy_level, seed)
    W0 = randn(d1, k1)
    H0 = randn(d2, k1)
    U0 = randn(n1, k2)
    V0 = randn(n2, k2)

# Run IMC
    W, H, U, V, losses = dirtyIMC(R, X, Y, k1, k2, lamb1, lamb2, maxiter, W0, H0, U0, V0)

    Diff = X.dot(W.T).dot(H).dot(Y.T) + U.T.dot(V) - A;
    # Diff[R==0] = 0

    relerr = norm(Diff, 'fro')**2 / norm(A, 'fro')**2 * 100
    print 'dirtyIMC RelErr = %g'%(relerr)
    # plt.plot(losses)
    # plt.yscale('log')
    # plt.show()

    return relerr


do_test_imc()
do_test_dirty_imc()

