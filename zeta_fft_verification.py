#!/usr/bin/env python3
"""
VPR RESEARCH Ltd
zeta_fft_verification.py
=========================
Blind peak detection followed by reference-based validation of the explicit
formula for zeta(s), in three diagnostics (same pipeline as for L(s, chi_3)):

  1. POSITION  - FFT peaks vs known zeta zeros (one-to-one matching, so no
                 two known zeros can be matched to the same detected peak)
  2. AMPLITUDE - measured |spectrum| vs the naive prediction C/|rho|, with C
                 fitted from one isolated calibration peak, which is excluded
                 from the reported statistics
  3. PHASE     - measured arg(spectrum) vs the closed-form phase, Eq. (8) of
                 the paper (fft_explicit_formula_lib.predicted_phase); it has
                 no fitted parameter

Each matched zero is reported with its isolation gap (distance to the nearest
OTHER known zero).  The per-zero table here shows models A (single zero, no
window response) and C (multi-zero, fitted C).  Models A-E, the split into
isolated/close zeros and the dependence on height are produced by
zeta_model_comparison.py.

Reference zeros: the first 100 zeros of zeta(s), computed with mpmath, in
zeta_reference_zeros.py (keep it next to this script).

Usage:
    python3 zeta_fft_verification.py [--limit N] [--xmin X] [--top-n K] [--tol-pct P]
Defaults: --limit 500000000 --xmin 150 --top-n 60 --tol-pct 3
(the configuration of the paper; --limit 50000000 is much faster and
reproduces the qualitative results).  Memory at N = 5e8: about 5-6 GB.
"""
import argparse
import math
import os
import runpy
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal.windows import kaiser
from fft_explicit_formula_lib import (
    run_fft, predicted_phase, isolation_gap, bijective_match,
    calibrate_leakage_model, leakage_corrected_prediction,
)

ISOLATION_THRESHOLD = 2.0  # gamma units; above this, a zero is "isolated"


# =============================================================================
# psi(x) - x   (in place; ~8 bytes * limit of memory plus the prime sieve)
# =============================================================================

def primes_up_to(limit):
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0]


def chebyshev_psi(limit):
    """psi(x) for every integer x <= limit, computed in place.

    Every prime p adds log p at x = p (k = 1, vectorised); prime powers p^k,
    k >= 2, exist only for p <= sqrt(limit) and are added in a short loop.
    Each cell receives at most one addition.
    """
    psi = np.zeros(limit + 1, dtype=np.float64)
    primes = primes_up_to(limit)
    psi[primes] += np.log(primes.astype(np.float64))
    for p in primes[primes <= math.isqrt(limit)]:
        logp = float(np.log(p))
        power = int(p) * int(p)
        while power <= limit:
            psi[power] += logp
            power *= int(p)
    del primes
    np.cumsum(psi, out=psi)
    return psi


def psi_minus_x(limit):
    """psi(x) - x, in place and in chunks (no extra full-size temporaries)."""
    a = chebyshev_psi(limit)
    step = 1 << 24
    for i in range(0, limit + 1, step):
        j = min(i + step, limit + 1)
        a[i:j] -= np.arange(i, j, dtype=np.float64)
    return a


def load_reference_zeros():
    here = os.path.dirname(os.path.abspath(__file__))
    return runpy.run_path(os.path.join(here, "zeta_reference_zeros.py"))["REFERENCE_ZEROS"]


def main():
    ap = argparse.ArgumentParser(description="zeta(s) FFT verification")
    ap.add_argument("--limit", type=int, default=500_000_000)
    ap.add_argument("--xmin", type=int, default=150)
    ap.add_argument("--top-n", type=int, default=60,
                    help="number of known zeros to attempt to match (paper: 60)")
    ap.add_argument("--tol-pct", type=float, default=3.0,
                    help="matching tolerance in percent of gamma (paper: 3)")
    args = ap.parse_args()
    limit, xmin, top_n = args.limit, args.xmin, args.top_n

    print("=" * 90)
    print("ZETA FFT VERIFICATION — position, amplitude, phase (one-to-one matching)")
    print("=" * 90)
    print(f"limit={limit:,}  xmin={xmin}  known zeros considered={top_n}  tol={args.tol_pct}%")

    print("\nComputing psi(x) - x ...")
    delta = psi_minus_x(limit)

    print("Applying FFT...")
    fr = run_fft(delta, limit, xmin)
    print(f"  Total local maxima above threshold: {len(fr['gammas_peaks'])}")

    known = load_reference_zeros()[:top_n]

    # -------------------------------------------------------------------
    # 1. POSITION: one-to-one matching
    # -------------------------------------------------------------------
    assignment = bijective_match(known, fr["gammas_peaks"], max_pct=args.tol_pct)

    # -------------------------------------------------------------------
    # 2. AMPLITUDE: fit C from the first ISOLATED matched zero
    # -------------------------------------------------------------------
    calib_j = None
    for j, cand_idx, pct in assignment:
        if cand_idx is None:
            continue
        gk = known[j]
        if isolation_gap(known, gk) > ISOLATION_THRESHOLD:
            calib_j = j
            break
    if calib_j is None:
        print("ERROR: no isolated matched zero found for amplitude calibration.")
        sys.exit(1)

    _, calib_cand, _ = assignment[calib_j]
    gamma0 = known[calib_j]
    amp0 = fr["amps_peaks"][calib_cand]
    C = amp0 * np.sqrt(0.25 + gamma0 ** 2)

    # -------------------------------------------------------------------
    # 2b. MULTI-ZERO MODEL, Eq. (11):
    #   S[k] ~ C_leak * sum_i A0(gamma_i) exp(i gamma_i u_min) W(omega_k - omega_i)
    # summed over ALL reference zeros with the exact numerical DTFT of the
    # window.  C_leak is one complex constant fitted from the calibration zero;
    # if the normalisation is consistent it equals 1.
    # -------------------------------------------------------------------
    window = kaiser(fr["N"], beta=14.0)
    calib_k = fr["peak_k"][calib_cand]
    C_leak = calibrate_leakage_model(known, calib_j, calib_k, fr, window)
    print(f"\nCalibration zero gamma={gamma0:.4f}: |C_leak|={abs(C_leak):.5f}  "
          f"arg(C_leak)={np.angle(C_leak):+.5f} rad  (expected: |C_leak| = 1)")

    # -------------------------------------------------------------------
    # 3. Build the combined per-zero table
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'gamma_known':>12} {'gamma_det':>11} {'gap':>6} {'amp_ratio':>10} "
          f"{'phase_meas':>11} {'phase_pred':>11} {'phase_diff':>11} "
          f"{'resid_A':>11} {'resid_C':>10} {'note':<14}")
    print("-" * 100)

    amp_ratios, phase_diffs = [], []
    resid_single_list, resid_multi_list, resid_gaps = [], [], []
    for j, cand_idx, pct in assignment:
        gk = known[j]
        gap = isolation_gap(known, gk)
        isolated = gap > ISOLATION_THRESHOLD
        note = "calibration" if j == calib_j else ("" if isolated else "close neighbor")

        if cand_idx is None:
            print(f"{gk:>12.4f} {'--- no match ---':>11} {gap:>6.2f} {'':>10} "
                  f"{'':>11} {'':>11} {'':>11} {'NO MATCH':<14}")
            continue

        k = fr["peak_k"][cand_idx]
        g_det = fr["gammas_peaks"][cand_idx]
        amp = fr["amps_peaks"][cand_idx]
        predicted_amp = C / np.sqrt(0.25 + gk ** 2)
        ratio = amp / predicted_amp

        phase_meas = np.angle(fr["spectrum"][k])
        phase_pred = predicted_phase(gk, fr, k)
        phase_diff = (phase_meas - phase_pred + np.pi) % (2 * np.pi) - np.pi

        measured_c = fr["spectrum"][k]
        pred_single_c = predicted_amp * np.exp(1j * phase_pred)
        pred_multi_c = leakage_corrected_prediction(known, C_leak, k, fr, window)
        resid_single = abs(measured_c - pred_single_c) / abs(measured_c)
        resid_multi = abs(measured_c - pred_multi_c) / abs(measured_c)

        if j != calib_j:
            amp_ratios.append(ratio)
            resid_single_list.append(resid_single)
            resid_multi_list.append(resid_multi)
            resid_gaps.append(gap)
        phase_diffs.append((phase_diff, isolated))

        print(f"{gk:>12.4f} {g_det:>11.4f} {gap:>6.2f} {ratio:>10.3f} "
              f"{phase_meas:>11.4f} {phase_pred:>11.4f} {phase_diff:>11.4f} "
              f"{resid_single:>11.4f} {resid_multi:>10.4f} {note:<14}")

    amp_ratios = np.array(amp_ratios)
    phase_diffs_isolated = np.array([d for d, iso in phase_diffs if iso])
    phase_diffs_all = np.array([d for d, _ in phase_diffs])

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    n_matched = sum(1 for _, c, _ in assignment if c is not None)
    print(f"Position:  {n_matched}/{top_n} known zeros matched within {args.tol_pct}% (one-to-one)")
    print(f"Amplitude (naive, window response ignored): mean ratio={np.mean(amp_ratios):.4f}  "
          f"std={np.std(amp_ratios):.4f}  (n={len(amp_ratios)}, calibration point excluded)")
    print(f"Phase (isolated zeros only, gap>{ISOLATION_THRESHOLD}): "
          f"mean|diff|={np.mean(np.abs(phase_diffs_isolated)):.4f} rad  "
          f"median|diff|={np.median(np.abs(phase_diffs_isolated)):.4f} rad  (n={len(phase_diffs_isolated)})")
    print(f"Phase (all matched zeros): mean|diff|={np.mean(np.abs(phase_diffs_all)):.4f} rad  "
          f"median|diff|={np.median(np.abs(phase_diffs_all)):.4f} rad  (n={len(phase_diffs_all)})")

    resid_single_arr = np.array(resid_single_list)
    resid_multi_arr = np.array(resid_multi_list)
    resid_gaps_arr = np.array(resid_gaps)
    close_mask = resid_gaps_arr < ISOLATION_THRESHOLD
    print(f"\nComplex-spectrum residual, relative to |measured| (A: single zero, no window; "
          f"C: multi-zero):")
    print(f"  all matched (n={len(resid_single_arr)}):    "
          f"A mean={np.mean(resid_single_arr):.4f}  C mean={np.mean(resid_multi_arr):.4f}")
    if close_mask.sum():
        print(f"  close-neighbor only (n={close_mask.sum()}): "
              f"A mean={np.mean(resid_single_arr[close_mask]):.4f}  "
              f"C mean={np.mean(resid_multi_arr[close_mask]):.4f}")

    # -------------------------------------------------------------------
    # Diagnostic plot (not used in the paper): naive amplitude ratio and
    # phase error vs isolation gap.
    # -------------------------------------------------------------------
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    gaps_for_amp, ratios_for_plot = [], []
    for (j, cand_idx, pct), r in zip(
        [a for a in assignment if a[1] is not None and a[0] != calib_j], amp_ratios
    ):
        gaps_for_amp.append(isolation_gap(known, known[j]))
        ratios_for_plot.append(r)
    axes[0].scatter(gaps_for_amp, ratios_for_plot, color="darkorange")
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Isolation gap (gamma units)")
    axes[0].set_ylabel("Amplitude / predicted (naive)")
    axes[0].set_title("Amplitude ratio vs isolation gap")
    axes[0].grid(alpha=0.3)

    gaps_for_phase = [isolation_gap(known, known[j]) for j, c, _ in assignment if c is not None]
    axes[1].scatter(gaps_for_phase, np.abs(phase_diffs_all), color="seagreen")
    axes[1].set_xlabel("Isolation gap (gamma units)")
    axes[1].set_ylabel("|phase error| (rad)")
    axes[1].set_title("Phase error vs isolation gap")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("zeta_fft_verification.png", dpi=150)
    print("  Saved: zeta_fft_verification.png")


if __name__ == "__main__":
    main()