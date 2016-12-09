import numpy as np
from numpy import random
from numpy import linalg as la


class CGDParams:
    Omega = None
    lamb = None
    R = None
    P = None
    Q = None
    d = None
    k = None


def CGD(winit, gradw, params, maxiter = 10):
    """
        Conjugate gradient descent for IMC

    :param winit: starting value of w.
    :param gradw: gradient of the objective at winit.
    :param params: information needed for solver.
    :param maxiter: max number of CGD iterations.
    :return w: vectorized version of W.
    """

    r = -gradw;
    d = r.copy();
    tol = 1e-6;
    w = winit.copy();

    for i in xrange(maxiter):
        if la.norm(r) <= tol:
            break

        r2 = r.inner(r)
        S = np.reshape(d, (params.d, params.k))
        U = params.P.dot(S).dot(params.Q)
        U[~params.Omega] = 0
        PUQ = params.P.transpose().dot(U).dot(params.Q.transpose())
        Hd = PUQ.flatten() + params.lamb * d

        alpha = r2 / (d.inner(Hd))
        w = w + alpha * d
        r = r - alpha * Hd
        beta = r.inner(r) / r2
        d = r + beta * d

    return w



def IMC(R, X, Y, k, lamb, maxiter, WInit = None, HInit = None):

    """
        Inductive Matrix Completion using squared loss:
            min_{W,H} \| R - X * W' * H * Y' \|_F^2 + \lambda * (\|W\|_F^2 + \|H\|_F^2)

    :param R: user-item matrix with missing entries. m * n
    :param X: row features. m * d1
    :param Y: column features. n * d2
    :param k: rank of latent matrix Z = WH'
    :param lamb: regularization parameter in the objective.
    :param maxiter: max iterations to run alternating minimization.
    :param WInit: W factor matrix initialization. d1 * k
    :param HInit: H factor matrix initialization. d2 * k
    :
    :return W: factor matrix W. d1 * k
    :return H: factor matrix H. d2 * k
    :return losses: store losses after each update. (2maxiter) * 1
    """

    Omega = R!=0
    m, d1 = X.shape;
    n, d2 = Y.shape;

    if WInit is None:
        W = random.randn([d1, k])
    else:
        W = WInit.copy()

    if HInit is None:
        H = random.randn([k, d2])
    else:
        H = HInit.transpose().copy()

    params = CGDParams()
    params.lamb = lamb
    params.Omega = Omega
    params.R = R

    for i in xrange(maxiter):
        print 'Iter ' + str(i) + '. Updating W. ',
        Q = H.dot(Y.transpose())
        XWQ = X.dot(W).dot(Q)
        XWQ[~Omega] = 0
        D = XWQ - R
        GradW = X.transpose().dot(D).dot(Q.transpose()) + lamb * W
        params.P = X
        params.Q = Q
        params.d = d1
        params.k = k
        w = CGD(W.flatten(), GradW.flatten(), params)
        W = np.reshape(w, (d1, k))

        print 'Updating W.'
        P = X.dot(W)
        PHY = P.dot(H).dot(Y.transpose())
        PHY[~Omega] = 0
        D = PHY - R
        GradH = P.transpose().dot(D).dot(Y) + lamb * H
        params.P = P
        params.Q = Y.transpose()
        params.d = k
        params.k = d2
        h = CGD(H.flattern(), GradH.flattern(), params)
        H = np.reshape(h, (k, d2))

    return W.transpose(), H, 0






