#!/usr/bin/env python3
"""
VPR RESEARCH Ltd
fft_explicit_formula_lib.py
============================
Shared utilities for the FFT-based numerical study of the explicit formula
that connects the prime-power sum psi(x, chi) (or psi(x) - x for zeta) to the
nontrivial zeros of an L-function.  Companion code of the paper
"Closed-Form Phase Prediction and Spectral-Leakage Modeling for the Explicit
Formula of L(s, chi_3), with Blind FFT Detection".

Used by:
    chi3_fft_verification.py     per-zero table, L(s, chi_3)
    chi3_model_comparison.py     models A-E, isolated/close splits, blind check
    generate_chi3_figures.py     Figures 1-2 and Appendix A
    zeta_fft_verification.py     per-zero table, zeta(s)
    zeta_model_comparison.py     models A-E for zeta, by height

Notation: `log_min` in the code is u_min = ln(x_min) in the paper;
`omega0 = gamma*du` and `omega_k = 2*pi*k/N` are digital angular frequencies;
`N` here is the length N_log of the log-grid (2**17), not the upper bound.

Provides
--------
run_fft(delta, limit, xmin, n_log_pow2, threshold_ratio)
    Resample delta(x) on a half-open log-grid (psi is constant between
    integers, so the sample at x = e^u is exactly psi(floor(x)); the
    normalisation uses sqrt(floor(x))), subtract the mean, apply the symmetric
    Kaiser(beta=14) window, FFT.  Returns the raw complex spectrum, the bin
    frequencies, and the local maxima above `threshold_ratio` times the largest
    amplitude.  No information about the zeros is used here.

predicted_phase(gamma_known, fft_result, k)                       [Eq. (8)]
    Closed-form prediction for arg(spectrum[k]): analytic phase of the
    x^rho/rho term, sampling-origin term, and the linear phase of the
    symmetric window.  It contains NO fitted parameter.  For L(s, chi_3) at
    N = 5e8, x_min = 150 the median absolute error is 4e-4 rad for isolated
    zeros (gap > 2), and at most ~0.01 rad for close pairs.

window_dtft(window, delta_omegas)                                 [Eq. (10)]
    Exact, numerically evaluated DTFT of the window.

calibrate_leakage_model(...), leakage_corrected_prediction(...)   [Eq. (11)]
    The multi-zero model  S[k] ~ C sum_i A0(gamma_i) e^{i gamma_i u_min}
    W(omega_k - omega_i).  C is one complex constant, fitted from one
    isolated zero; the fits give |C| = 1 to within 3.4e-4 (paper, Section
    3.5), and the model with C = 1 (no free parameter) performs identically.

isolation_gap(zeros, gamma)
    Distance from `gamma` to its nearest OTHER zero in the reference list.
    Used to split zeros into isolated (gap > 2) and close (gap < 2).  A close
    neighbour contributes through the window response at the detected bin
    (see the multi-zero model); the amplitude error of an isolated zero is
    mainly scalloping, i.e. the offset between the zero and the bin centre.

bijective_match(known_zeros, candidate_gammas, max_pct)
    One-to-one assignment of known zeros to detected peaks (Hungarian
    algorithm via scipy.optimize.linear_sum_assignment), minimising total
    |delta gamma|.  Every local maximum is a candidate; a zero is left
    unmatched if no free candidate lies within max_pct percent of its gamma.
    Unlike nearest-neighbour matching this never assigns one peak to two
    zeros, which would otherwise corrupt the amplitude and phase statistics.
    Note: a tolerance in percent is a poor criterion at low gamma (1% of
    gamma is below half a bin) and loose at high gamma (3% of 160 is about ten
    bins); the paper uses 3%.

Derivation of predicted_phase
------------------------------
For a real primitive character chi (or the principal character, i.e. zeta
itself after subtracting the main term x), the explicit formula gives

    delta(x) ~ -sum_{rho} x^rho / rho

For a real L-function the zeros come in conjugate pairs rho = 1/2 +- i*gamma,
and the pair sums to a real cosine term:

    delta(x)/sqrt(x) ~ -2 * sum_{gamma>0} (1/|rho|) * cos(gamma*u - theta),
        u = ln(x),  theta = arctan(2*gamma),  |rho| = sqrt(1/4 + gamma^2)

In the frequency domain this is a sinusoid e^{i*gamma*u} with complex
coefficient A0 = -(1/|rho|) * e^{-i*theta}, so angle(A0) = pi - theta.

Three more phase contributions must be tracked to match the FFT's exact
convention:
  1. Sampling origin: the signal starts at u = u_min, not u = 0, adding
     a phase of +gamma*u_min.
  2. Kaiser window centering: a symmetric real window w_n (n=0..N-1) has
     DTFT W(w) = exp(-i*w*(N-1)/2) * R(w) for a REAL, even R(w). The windowed
     FFT output at bin k is a convolution with W, which contributes a
     linear-phase term (omega0 - omega_k)*(N-1)/2, where omega0 is the
     true zero's digital frequency and omega_k is the actual FFT bin's
     digital frequency (they differ because the true gamma rarely lands
     exactly on a bin).
  3. R(w) is assumed positive near the main-lobe peak, which holds for a
     well-resolved, isolated zero; this assumption is what fails for close
     pairs, which therefore show larger phase errors.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import kaiser
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment


def run_fft(delta, limit, xmin, n_log_pow2=17, threshold_ratio=0.05):
    """
    Resample delta(x) in log-space, window with Kaiser(beta=14), and FFT.

    Returns a dict with the raw complex spectrum, the bin frequencies
    (cycles per unit u=ln x), the detected peak bin indices (k, into the
    full-length `spectrum`/`freqs` arrays -- needed by predicted_phase),
    and the corresponding detected gammas/amplitudes.
    """
    N = 1 << n_log_pow2
    log_min = np.log(max(2, xmin))
    log_max = np.log(limit)
    span = log_max - log_min
    du = span / N

    
    log_x = log_min + np.arange(N) * du
    x_log = np.exp(log_x).astype(np.int64)
    x_log = np.clip(x_log, 1, limit)

    delta_log = delta[x_log]
    delta_norm = delta_log / np.sqrt(x_log.astype(np.float64))
    delta_norm -= np.mean(delta_norm)

    window = kaiser(N, beta=14.0)
    windowed = delta_norm * window

    spectrum = fft(windowed)
    freqs = fftfreq(N, d=du)  # cycles per unit u = ln(x)

    pos_idx = np.where(freqs > 0)[0]
    amps_pos = np.abs(spectrum[pos_idx])
    peak_local, _ = find_peaks(amps_pos, height=threshold_ratio * amps_pos.max())
    peak_k = pos_idx[peak_local]  # true bin indices into `spectrum`/`freqs`

    gammas_peaks = 2 * np.pi * freqs[peak_k]
    amps_peaks = amps_pos[peak_local]

    return {
        "N": N, "du": du, "log_min": log_min, "span": span,
        "spectrum": spectrum, "freqs": freqs,
        "peak_k": peak_k, "gammas_peaks": gammas_peaks, "amps_peaks": amps_peaks,
    }


def predicted_phase(gamma_known, fft_result, k):
    """
    Predicted arg(spectrum[k]) for the x^rho/rho term with rho=1/2+i*gamma_known,
    sampled and windowed exactly as run_fft() does. See module docstring.
    """
    N, du, log_min = fft_result["N"], fft_result["du"], fft_result["log_min"]
    theta = np.arctan(2 * gamma_known)
    phase = (np.pi - theta) + gamma_known * log_min
    omega0 = gamma_known * du
    omega_k = 2 * np.pi * k / N
    phase += (omega0 - omega_k) * (N - 1) / 2.0
    return (phase + np.pi) % (2 * np.pi) - np.pi


def window_dtft(window, delta_omegas):
    """
    Exact DTFT of `window` (length-N real array, as used in run_fft) at one
    or more digital angular frequency offsets. Used to quantitatively
    predict spectral leakage between nearby zeros: no analytic
    approximation of the Kaiser window's sidelobes is used, only its
    literal, numerically evaluated frequency response.

    delta_omegas: scalar or 1D array. Returns a complex array of the same
    shape (or a 0-d array for a scalar input).
    """
    delta_omegas = np.atleast_1d(np.asarray(delta_omegas, dtype=float))
    n_idx = np.arange(len(window))
    # (n_eval, N) phase matrix times the window -> (n_eval,) complex result
    M = np.exp(-1j * np.outer(delta_omegas, n_idx))
    return M @ window


def _A0(gamma):
    """Unnormalized complex amplitude of the x^rho/rho term, rho=1/2+i*gamma."""
    theta = np.arctan(2 * gamma)
    return -(1.0 / np.sqrt(0.25 + gamma**2)) * np.exp(-1j * theta)


def calibrate_leakage_model(known_zeros, calib_idx, calib_k, fft_result, window):
    """
    Fit the single complex calibration constant C for the leakage-aware
    model, using one isolated known zero (so its own term dominates and
    neighbor leakage is negligible for calibration purposes).

    measured = C * A0(gamma_calib) * exp(i*gamma_calib*log_min) * W(omega_k - omega_calib)

    Returns C (complex). If the explicit-formula normalization (sqrt(x), the
    FFT/window conventions) is self-consistent, C = 1: the fit is a
    consistency check, and the model with C = 1 needs no calibration at all.
    """
    du, log_min, N = fft_result["du"], fft_result["log_min"], fft_result["N"]
    gamma_c = known_zeros[calib_idx]
    measured = fft_result["spectrum"][calib_k]
    omega_k = 2 * np.pi * calib_k / N
    omega_c = gamma_c * du
    W0 = window_dtft(window, omega_k - omega_c)[0]
    return measured / (_A0(gamma_c) * np.exp(1j * gamma_c * log_min) * W0)


def leakage_corrected_prediction(known_zeros, C, k, fft_result, window):
    """
    Predicted complex spectrum[k], summing the windowed contribution of
    EVERY known zero (not just the one nominally matched to bin k):

        spectrum[k] ~ C * sum_i A0(gamma_i) * exp(i*gamma_i*log_min) * W(omega_k - omega_i)

    This is the leakage-aware generalization of predicted_phase()/the
    single-zero amplitude model: for an isolated zero the sum is
    dominated by its own term and this reduces to the old prediction; for
    a zero with a close neighbor, the neighbor's windowed tail is included
    explicitly rather than showing up as an unexplained residual.
    """
    du, log_min, N = fft_result["du"], fft_result["log_min"], fft_result["N"]
    omega_k = 2 * np.pi * k / N
    omega_all = known_zeros * du
    B_all = C * _A0(known_zeros) * np.exp(1j * known_zeros * log_min)
    Wresp = window_dtft(window, omega_k - omega_all)
    return np.sum(B_all * Wresp)


def isolation_gap(zeros, gamma, eps=1e-3):
    """Distance from `gamma` to its nearest OTHER entry in `zeros`."""
    d = np.abs(np.asarray(zeros) - gamma)
    d = d[d > eps]
    return float(d.min()) if len(d) else np.inf


def bijective_match(known_zeros, candidate_gammas, max_pct=1.0):
    """
    One-to-one assignment of known zeros to detected peak gammas, minimizing
    total |delta gamma| (Hungarian algorithm). Guarantees no two known zeros
    are matched to the same detected peak, unlike nearest-neighbor matching.

    Plain linear_sum_assignment forces every known zero (column) to take
    SOME candidate row, even when no genuinely close candidate exists --
    with far more candidates than known zeros, this doesn't just leave that
    one zero with a bad match, it can cascade: a zero with no good
    candidate "steals" a mediocre one, which bumps a neighboring zero off
    ITS good candidate, and so on down the line. To avoid this, each known
    zero gets its own "give up" option: a dummy row whose cost equals
    exactly the max_pct threshold for that zero, and is prohibitively
    expensive for every other zero. The assignment then only uses a real
    candidate for a column when that's genuinely cheaper than giving up --
    never as a forced side effect of balancing the matrix.

    Returns a list, one entry per known zero, of:
        (known_idx, candidate_idx or None, pct_diff or None)
    candidate_idx/pct_diff are None if no assignment stays within max_pct.
    """
    known_zeros = np.asarray(known_zeros, dtype=float)
    candidate_gammas = np.asarray(candidate_gammas, dtype=float)
    n_known = len(known_zeros)
    n_cand = len(candidate_gammas)

    if n_cand == 0:
        return [(j, None, None) for j in range(n_known)]

    real_cost = np.abs(known_zeros[None, :] - candidate_gammas[:, None])  # (n_cand, n_known)

    # Dummy "give up" rows: one per known zero, cost = max_pct% of that
    # zero's gamma for its own column, effectively infinite for any other.
    give_up_cost = (max_pct / 100.0) * known_zeros
    BIG = 1e9
    dummy = np.full((n_known, n_known), BIG)
    np.fill_diagonal(dummy, give_up_cost)

    full_cost = np.vstack([real_cost, dummy])  # (n_cand + n_known, n_known)
    row_ind, col_ind = linear_sum_assignment(full_cost)
    assigned = {c: r for r, c in zip(row_ind, col_ind)}

    results = []
    for j, gk in enumerate(known_zeros):
        r = assigned[j]
        if r >= n_cand:  # dummy row used -> this zero gave up
            results.append((j, None, None))
            continue
        pct = abs(candidate_gammas[r] - gk) / gk * 100
        results.append((j, int(r), pct))
    return results