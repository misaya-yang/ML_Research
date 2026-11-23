"""Exact cold-start risk for many observables at once, and window envelopes."""

import numpy as np


def dense_kernel(w, q):
    w = np.asarray(w, dtype=float)
    q = np.asarray(q, dtype=float)
    K = q[None, :] * np.minimum(1.0, w[None, :] / w[:, None])
    K[np.diag_indices(len(w))] += 1 - K.sum(axis=1)
    return K


def cold_risks(w, q, H, b, t, K=None):
    """R_Q(b,t)=E[(mean_{i=b}^{b+t-1} f(X_i))^2], f=h-Ph, for each column of H. X_0~Q."""
    w = np.asarray(w, dtype=float)
    q = np.asarray(q, dtype=float) / np.sum(q)
    H = np.atleast_2d(np.asarray(H, dtype=float).T).T
    if K is None:
        K = dense_kernel(w, q)
    P = q * w / np.sum(q * w)
    F = H - P @ H
    nu = q.copy()
    for _ in range(b):
        nu = nu @ K
    g1 = np.zeros_like(F)
    g2 = np.zeros_like(F)
    for _ in range(t):
        Kg1 = K @ g1
        g2 = F * F + 2 * F * Kg1 + K @ g2
        g1 = F + Kg1
    R = nu @ g2 / t**2
    if not np.all(np.isfinite(R)) or np.any(R < -1e-12):
        raise FloatingPointError("Non-finite or negative risk")
    return R, P @ H, P @ (H * H)


def envelopes(w, q, b, t):
    """Gamma_t, Xi_{b,t} (plan definition, no 1/t term), eta, P-median d."""
    w = np.asarray(w, dtype=float)
    q = np.asarray(q, dtype=float) / np.sum(q)
    z = q @ w
    w = w / z
    P = q * w
    L = b + t
    gamma = P @ np.minimum(w, t) / t
    xi = P @ np.minimum.reduce([w / t, np.ones_like(w), L / w]) + (P @ (w > L))**2
    eta = q @ np.minimum(1, w)
    return float(gamma), float(xi), float(eta)


def p_quantile_threshold(w, P, u):
    """Smallest v with P(W<=v) >= u."""
    order = np.argsort(w)
    cdf = np.cumsum(P[order])
    return float(w[order][min(np.searchsorted(cdf, u - 1e-15), len(w) - 1)])
