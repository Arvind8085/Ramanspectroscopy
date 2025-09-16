import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

def baseline_als_optimized(y, smth, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = smth * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        x = spsolve(Z, w*y)
        w = p * (y > x) + (1-p) * (y < x)
    return x

def als_baseline(y, smth, p, n=5):
    L = len(y)
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    for _ in range(n):
        W = np.diag(w)
        A = W + smth * D.dot(D.T)
        x = np.linalg.solve(A, w * y)
        w = p * (y > x) + (1 - p) * (y < x)
    return x

def apply_butterworth(data, order, cutoff_frequency):
    B, A = signal.butter(order, cutoff_frequency,  output='ba')
    filtered_data = signal.filtfilt(B, A, data)
    return filtered_data

def calculate_snr(signal, noise):
    signal_rms = np.sqrt(np.mean(np.square(signal)))
    noise_rms = np.sqrt(np.mean(np.square(noise)))
    snr_db = 20 * np.log10(signal_rms / noise_rms)
    return snr_db   

import numpy as np


