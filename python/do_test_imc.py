import numpy as np
from numpy import random
from numpy.random import randn
from numpy.linalg import qr
from numpy.linalg import norm
from IMC import IMC

def generate_synthesized_data(k, d1, d2, n1, n2, m, seed = None):
    """
    Input:
        k:           rank
        d1,d2:       dimension of latent space
        n1,n2:       user, movie sizes
        m:           number of samples
        seed:        random seed
    
    Output:
        X,Y:         features
        W:           ground truth low rank matrix
        G:           sparse n1xn2 matrix, which non-zero entries are observed
                     G = (X*W*Y')_\Omega, where \Omega is the sample set with size m
    """
    random.RandomState(seed)
    W = randn(d1, k).dot(randn(k, d2))

    X = randn(n1, d1)
    Y = randn(n2, d2)

# Orthogonalize X and Y
    X, _ = qr(X)
    Y, _ = qr(Y)

    G = X.dot(W).dot(Y.T)

    # Omega = random.choice(range(n1*n2), m, replace=False)
    # ii = np.zeros((m,))
    # jj = np.zeros((m,))
    # b = np.zeros((m,))
    # for ij in range(m):
    #     i = np.floor(Omega[ij] / n2).astype(int)
    #     ii[ij] = i
    #     j = (Omega[ij] - i * n2).astype(int)
    #     jj[ij] = j
    #     b[ij] = X[i, :].dot(W).dot(Y[j,:].T)

    # G = csr_matrix((b, (ii, jj)), shape=(n1, n2))

    return (X, Y, G, W)

def do_test_imc(k = 5, d1 = 50, d2 = 80, n1 = 120, n2 = 100, m = 1):
    seed = 1
    lamb = 1e-3
    maxiter = 100
    print 'Generating random data...'
    X, Y, A, Z = generate_synthesized_data(k, d1, d2, n1, n2, m, seed)
    W0 = randn(d1, k)
    H0 = randn(d2, k)
    print 'Done!'
    relerr = norm(W0.dot(H0.T) - Z, 'fro') ** 2 / norm(Z, 'fro') ** 2 * 100
    print 'RelErr = %f' % (relerr)

# Run IMC
    W, H, losses = IMC(A, X, Y, k, lamb, maxiter, W0, H0)
    relerr = norm(W.T.dot(H) - Z, 'fro')**2 / norm(Z, 'fro')**2 * 100
    print 'RelErr = %g'%(relerr)
    return relerr


do_test_imc()
