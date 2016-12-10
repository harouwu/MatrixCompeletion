import numpy as np
from numpy import random
from numpy.random import randn
from numpy.linalg import qr
from numpy.linalg import svd


def generate_synthesized_data(k, d1, d2, n1, n2, sparsity, seed=None):
    """
    Input:
        k:           rank
        d1,d2:       dimension of latent space
        n1,n2:       user, movie sizes
        sparsity:    the sparsity of observed matrix
        seed:        random seed

    Output:
        X,Y:         features
        W:           ground truth low rank matrix
        A:           ground truth full matrix
        R:           sparse n1xn2 matrix, which non-zero entries are observed
                     R = (X*W*Y')_\Omega, where \Omega is the sample set with size m
    """

    m = int(round(sparsity * (n1 * n2)))

    random.seed(seed)
    W = randn(d1, k).dot(randn(k, d2))

    X = randn(n1, d1)
    Y = randn(n2, d2)

    # Orthogonalize X and Y
    X, _ = qr(X)
    Y, _ = qr(Y)

    A = X.dot(W).dot(Y.T)

    Omega = random.choice(range(n1 * n2), m, replace=False)
    mask = np.zeros(A.shape)
    for ij in range(m):
        i = np.floor(Omega[ij] / n2).astype(int)
        j = (Omega[ij] - i * n2).astype(int)
        mask[i, j] = 1.0

    R = A * mask

    return X, Y, W, A, R


def generate_synthesized_noisy_data(k, d1, d2, n1, n2, sparsity, noise_level, seed=None):
    """
    Input:
        k:           rank
        d1,d2:       dimension of latent space
        n1,n2:       user, movie sizes
        sparsity:    the sparsity of observed matrix
        seed:        random seed

    Output:
        X,Y:         features
        A:           ground truth full matrix
        R:           sparse n1xn2 matrix, which non-zero entries are observed
                     R = (X*W*Y')_\Omega, where \Omega is the sample set with size m
    """

    m = int(round(sparsity * (n1 * n2)))

    random.seed(seed)
    W = randn(d1, k).dot(randn(k, d2))

    X = randn(n1, d1)
    Y = randn(n2, d2)

    # Orthogonalize X and Y
    X, _ = qr(X)
    Y, _ = qr(Y)

    A = X.dot(W).dot(Y.T)

    Omega = random.choice(range(n1 * n2), m, replace=False)
    mask = np.zeros(A.shape)
    for ij in range(m):
        i = np.floor(Omega[ij] / n2).astype(int)
        j = (Omega[ij] - i * n2).astype(int)
        mask[i, j] = 1.0

    R = A * mask

    return X, Y, A, R