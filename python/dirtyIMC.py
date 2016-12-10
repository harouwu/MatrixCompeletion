import numpy as np
from numpy import random
from numpy import linalg as la
from CGD import *


def computeLoss(R, X, W, H, Y, U, V, lamb1, lamb2, Omega):
    P = X.dot(W).dot(H).dot(Y.T) + U.dot(V)
    P[~Omega] = 0
    loss = la.norm(P - R, 'fro')**2 + lamb1 * la.norm(W, 'fro')**2 + lamb2 * la.norm(H, 'fro')**2
    return loss


def dirtyIMC(R, X, Y, k1, k2, lamb1, lamb2, maxiter, WInit = None, HInit = None, UInit = None, VInit = None):

    """
        Dirty Inductive Matrix Completion using squared loss:
            min_{W,H} \| R - X * W' * H * Y' - U' * V \|_F^2 + \lambda_1 * (\|W\|_F^2 + \|H\|_F^2) + \lambda_2 * (\|U\|_F^2 + \|V\|_F^2)

    :param R: user-item matrix with missing entries. m * n
    :param X: row features. m * d1
    :param Y: column features. n * d2
    :param k: rank of latent matrix Z = WH'
    :param lamb: regularization parameter in the objective.
    :param maxiter: max iterations to run alternating minimization.
    :param WInit: W factor matrix initialization. d1 * k1
    :param HInit: H factor matrix initialization. d2 * k1
    :param UInit: U factor matrix initialization. m * k2
    :param VInit: V factor matrix initialization. n * k2
    :
    :return W: factor matrix W. d1 * k
    :return H: factor matrix H. d2 * k
    :return U: factor matrix U. m * k2
    :return V: factor matrix V. n * k2
    :return losses: store losses after each update. (4maxiter) * 1
    """

    Omega = R!=0
    m, d1 = X.shape;
    n, d2 = Y.shape;

    if WInit is None:
        W = random.randn([d1, k1])
    else:
        W = WInit.copy()

    if HInit is None:
        H = random.randn([k1, d2])
    else:
        H = HInit.T.copy()

    if UInit is None:
        U = random.randn([m, k2])
    else:
        U = UInit.copy()

    if VInit is None:
        V = random.randn([k2, n])
    else:
        V = VInit.T.copy()

    params = CGDParams()
    params.Omega = Omega

    losses = np.zeros(4 * maxiter)

    for i in xrange(maxiter):

        # print 'Iter ' + str(i) + '. Updating W. ',
        UV = U.dot(V)
        Q = H.dot(Y.T)
        XWQ = X.dot(W).dot(Q) + UV
        XWQ[~Omega] = 0
        D = XWQ - R
        GradW = X.T.dot(D).dot(Q.T) + lamb1 * W
        params.P = X
        params.Q = Q
        params.d = d1
        params.k = k1
        params.lamb = lamb1
        w = CGD(W.flatten(), GradW.flatten(), params)
        W = np.reshape(w, (d1, k1))
        losses[4 * i] = computeLoss(R, X, W, H, Y, U, V, lamb1, lamb2, Omega)

        # print 'Updating H. ',
        P = X.dot(W)
        PHY = P.dot(H).dot(Y.T) + UV
        PHY[~Omega] = 0
        D = PHY - R
        GradH = P.transpose().dot(D).dot(Y) + lamb1 * H
        params.P = P
        params.Q = Y.T
        params.d = k1
        params.k = d2
        h = CGD(H.flatten(), GradH.flatten(), params)
        H = np.reshape(h, (k1, d2))
        losses[4 * i + 1] = computeLoss(R, X, W, H, Y, U, V, lamb1, lamb2, Omega)

        # print 'Updating U. ',
        XWHY = X.dot(W).dot(H).dot(Y.T)
        UV = XWHY + U.dot(V)
        UV[~Omega] = 0
        D = UV - R
        GradU = D.dot(V.T) + lamb2 * U
        params.P = np.eye(m)
        params.Q = V
        params.d = m
        params.k = k2
        params.lamb = lamb2
        u = CGD(U.flatten(), GradU.flatten(), params)
        U = np.reshape(u, (m, k2))
        losses[4 * i + 2] = computeLoss(R, X, W, H, Y, U, V, lamb1, lamb2, Omega)

        # print 'Updating V.'
        UV = XWHY + U.dot(V)
        UV[~Omega] = 0
        D = UV - R
        GradV = U.T.dot(D) + lamb2 * V
        params.P = U
        params.Q = np.eye(n)
        params.d = k2
        params.k = n
        v = CGD(V.flatten(), GradV.flatten(), params)
        V = np.reshape(v, (k2, n))
        losses[4 * i + 3] = computeLoss(R, X, W, H, Y, U, V, lamb1, lamb2, Omega)

    return W.transpose(), H, U.transpose(), V, losses






