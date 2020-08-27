import numpy as np
import numpy.matlib
from istft_kkmw import istft_kkmw
from stft_kkmw import stft_kkmw


def ADMM(xk, fs, num_mic):
    Gmin = 0.04
    vareps = 1e-10
    rho = 1
    gamma = 1
    num_ADM_iter = 20
    D = 3  # subband preditction delay
    K = 512  # the number of subbands
    Lc = 20  # subband prediction order
    num_out = 1

    winsize = 320
    winshift = 160
    eta = 0
    eps = np.power(float(2), -52)

    # Signal transformation into STFT domain
    xk = np.moveaxis(stft_kkmw(xk, winsize, K, winshift, fs), 0, 1)

    if np.isscalar(D) == True:
        D = np.matlib.repmat(D, 1, int(K / 2 + 1))

    if np.isscalar(Lc) == True:
        Lc = np.matlib.repmat(Lc, 1, int(K / 2 + 1))

    N = len(xk)
    dk = np.zeros((N, K, num_out), dtype=np.complex_)
    ck = np.zeros([20*num_mic, int(K / 2) + 1], dtype=np.complex_)  #check why 20*num_mic

    for k in range(0, int(K / 2) + 1):
        xk_tmp = np.zeros((N + Lc[0, k], num_mic), dtype=np.complex_)
        xk_tmp[Lc[0, k]:len(xk_tmp)] = (np.squeeze([xk[:, k, :]]))
        x_buf = xk_tmp[Lc[0, k]:len(xk_tmp), 0:num_out]
        X_D = np.zeros([N, num_mic * Lc[0, k]], dtype=np.complex_)
        for ii in range(0, N - D[0, k]):
            x_D = np.transpose(xk_tmp[ii + Lc[0, k]:ii:-1])
            X_D[ii + D[0, k]] = np.hstack(np.transpose(x_D))
        dk[:, k, :] = x_buf[:, 0:num_out]
        c = np.zeros((Lc[0, k] * num_mic, 1), dtype=np.complex_)
        dd = np.inf
        i = 1
        mu = np.zeros((N, 1), dtype=np.complex_) + 1j * np.zeros((N, 1), dtype=np.complex_)
        XXX = np.matmul(np.linalg.pinv(np.matmul(np.conjugate(np.transpose(X_D)), X_D)),
                        np.conjugate(np.transpose(X_D)))  # BLAD
        gl2 = np.matmul(XXX, x_buf)
        d = dk[:, k, :]
        while i < num_ADM_iter + 1 and np.max(np.divide(np.abs(dd), np.abs(d))) > 1e-3:
            d = x_buf - np.matmul(X_D, c) - mu
            d_old = d
            if eta > 0:
                sigma_d2 = nb_smooth(np.square(np.abs(d)), eta)
                w = 1 / (numpy.power(sigma_d2, 0.5))
            elif eta == 0:
                sigma_d2 = abs(d) + vareps
                w = 1 / sigma_d2
            h = d
            sp1 = 1 - (w / rho) / (abs(h) + eps)
            d = np.maximum(sp1, Gmin) * h
            gi = np.matmul(XXX, (d + mu))
            c = gl2 - gi
            dd = (d_old - d)
            mu = mu + gamma * (d + np.matmul(X_D, c) - x_buf)
            i = i + 1

        ck[:, k] = c[:, 0]
        dk[:, k, :] = d
    y = 1
    dk[:, int(K / 2) + 1:len(dk), :] = np.conjugate(dk[:, int(K / 2) - 1:0:-1, :])
    #X=istft_kkmw(np.conjugate(np.transpose(dk)),winsize,K,winshift) - in progress
    return [dk, ck]


def simple_smooth(x, eta):
    xPart1 = np.zeros(len(x), dtype=np.complex_)
    for v in range(1, len(x)):
        xPart1[v] = x[v - 1]
    xPart1 = xPart1 * eta
    x = xPart1 + (1 - eta) * x
    return x


def nb_smooth(x, nb):
    N = len(x)
    x_temp = np.concatenate(np.array([np.zeros(nb, dtype=np.complex_), np.hstack(x), np.zeros(nb, dtype=np.complex_)]))
    Nnb = 2 * nb + 1
    Nd = np.zeros((N, Nnb), dtype=np.complex_)
    for in_p in range(0, N):
        Nd[in_p] = x_temp[in_p:in_p + Nnb]
    weights = np.vstack(np.ones(Nnb) * (1 / Nnb))
    x = np.matmul(Nd, weights)
    return x
