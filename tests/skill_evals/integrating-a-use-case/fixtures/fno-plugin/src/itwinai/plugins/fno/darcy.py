"""Darcy flow solver and dataset generation, moved unchanged from train.py."""

import numpy as np


def gaussian_random_field(n_samples, size, alpha=3.0, rng=None):
    """Sample smooth random fields by filtering white noise in Fourier space."""
    rng = rng or np.random.default_rng(0)
    noise = rng.normal(size=(n_samples, size, size))

    kx = np.fft.fftfreq(size) * size
    ky = np.fft.fftfreq(size) * size
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    k2[0, 0] = 1.0
    spectrum = k2 ** (-alpha / 2.0)
    spectrum[0, 0] = 0.0

    field = np.fft.ifft2(np.fft.fft2(noise) * spectrum).real
    std = field.std(axis=(1, 2), keepdims=True)
    return field / np.where(std == 0, 1.0, std)


def apply_operator(a_right, a_left, a_up, a_down, u, h2):
    """Matrix-free -div(a grad u) with zero Dirichlet boundaries."""
    out = (a_right + a_left + a_up + a_down) * u
    out[:, :, :-1] -= a_right[:, :, :-1] * u[:, :, 1:]
    out[:, :, 1:] -= a_left[:, :, 1:] * u[:, :, :-1]
    out[:, :-1, :] -= a_down[:, :-1, :] * u[:, 1:, :]
    out[:, 1:, :] -= a_up[:, 1:, :] * u[:, :-1, :]
    return out / h2


def solve_darcy(a, tol=1e-8, max_iter=2000):
    """Solve the Darcy system for a batch of permeability fields with batched CG."""
    n_samples, size, _ = a.shape
    h2 = (1.0 / (size + 1)) ** 2

    # Harmonic means at the cell faces. The domain is padded by replication so that
    # boundary faces keep a non-zero coefficient: u vanishes outside, which is what
    # imposes the Dirichlet condition, but the conductivity there must not vanish too.
    a_pad = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")

    def face(shifted):
        return 2.0 * a * shifted / np.maximum(a + shifted, 1e-12)

    a_right = face(a_pad[:, 1:-1, 2:])
    a_left = face(a_pad[:, 1:-1, :-2])
    a_down = face(a_pad[:, 2:, 1:-1])
    a_up = face(a_pad[:, :-2, 1:-1])

    b = np.ones((n_samples, size, size))
    u = np.zeros_like(b)
    r = b - apply_operator(a_right, a_left, a_up, a_down, u, h2)
    p = r.copy()
    rs_old = np.sum(r * r, axis=(1, 2))

    for _ in range(max_iter):
        ap = apply_operator(a_right, a_left, a_up, a_down, p, h2)
        denom = np.sum(p * ap, axis=(1, 2))
        alpha = rs_old / np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        u += alpha[:, None, None] * p
        r -= alpha[:, None, None] * ap
        rs_new = np.sum(r * r, axis=(1, 2))
        if np.all(np.sqrt(rs_new) < tol):
            break
        p = r + (rs_new / np.where(rs_old == 0, 1e-30, rs_old))[:, None, None] * p
        rs_old = rs_new

    return u


def make_dataset(n_samples, size, seed):
    """Thresholded random fields give the usual piecewise-constant Darcy benchmark."""
    rng = np.random.default_rng(seed)
    psi = gaussian_random_field(n_samples, size, rng=rng)
    a = np.where(psi >= 0.0, 12.0, 3.0)
    u = solve_darcy(a)
    return a.astype(np.float32), u.astype(np.float32)
