import numpy as np
from numpy import random
from numpy.random import randn
from numpy.linalg import qr


def generate_synthesized_data(k, d1, d2, n1, n2, m, seed=None):
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
    random.seed(seed)
    W = randn(d1, k).dot(randn(k, d2))

    X = randn(n1, d1)
    Y = randn(n2, d2)

    # Orthogonalize X and Y
    X, _ = qr(X)
    Y, _ = qr(Y)

    G = X.dot(W).dot(Y.T)

    Omega = random.choice(range(n1 * n2), m, replace=False)
    mask = np.zeros(G.shape)
    for ij in range(m):
        i = np.floor(Omega[ij] / n2).astype(int)
        j = (Omega[ij] - i * n2).astype(int)
        mask[i, j] = 1.0

    G *= mask

    return X, Y, G, W