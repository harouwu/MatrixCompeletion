import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from IMC import IMC
from Utils import generate_synthesized_data
import matplotlib.pyplot as plt


def do_test_imc(k = 5, d1 = 50, d2 = 80, n1 = 120, n2 = 100, m = 1000):

    seed = 1
    lamb = 1e-3
    maxiter = 100
    print 'Generating random data...'
    X, Y, A, Z = generate_synthesized_data(k, d1, d2, n1, n2, m, seed)
    W0 = randn(d1, k)
    H0 = randn(d2, k)

# Run IMC
    W, H, losses = IMC(A, X, Y, k, lamb, maxiter, W0, H0)
    relerr = norm(W.T.dot(H) - Z, 'fro')**2 / norm(Z, 'fro')**2 * 100
    print 'RelErr = %g'%(relerr)
    plt.plot(losses)
    plt.yscale('log')
    plt.show()

    return relerr


do_test_imc()

