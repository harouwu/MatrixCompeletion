import numpy as np
from numpy import linalg as la


class CGDParams:
    Omega = None
    lamb = None
    P = None
    Q = None
    d = None
    k = None


def CGD(winit, gradw, params, maxiter=10):
    """
        Conjugate gradient descent for IMC

    :param winit: starting value of w.
    :param gradw: gradient of the objective at winit.
    :param params: information needed for solver.
    :param maxiter: max number of CGD iterations.
    :return w: vectorized version of W.
    """

    r = -gradw
    d = r.copy()
    tol = 1e-6
    w = winit.copy()

    for i in xrange(maxiter):
        if la.norm(r) <= tol:
            break

        r2 = r.T.dot(r)
        S = np.reshape(d, (params.d, params.k))
        U = params.P.dot(S).dot(params.Q)
        U[~params.Omega] = 0
        PUQ = params.P.transpose().dot(U).dot(params.Q.transpose())
        Hd = PUQ.flatten() + params.lamb * d

        alpha = r2 / (d.T.dot(Hd))
        w = w + alpha * d
        r = r - alpha * Hd
        beta = r.T.dot(r) / r2
        d = r + beta * d

    return w
