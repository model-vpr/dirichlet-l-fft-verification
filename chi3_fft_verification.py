#!/usr/bin/env python3
"""
VPR RESEARCH Ltd
chi3_fft_verification.py
==========================
Blind peak detection followed by reference-based validation of the explicit
formula for L(s, chi_3), the Dirichlet L-function of the nontrivial quadratic
character mod 3, in three diagnostics: position, amplitude, and phase.
See zeta_fft_verification.py for the zeta analogue and
fft_explicit_formula_lib.py for the shared derivations.

Only the peak detection is blind (no information about the zeros is used);
the matching to the reference zeros and the amplitude, phase, and
complex-spectrum tests use the reference list (paper, Section 1.2).

The per-zero table below shows model A (single zero, no window response) and
model C (multi-zero, fitted C).  Models A-E, the split into isolated and
close zeros and the blind check against more zeros are produced by
chi3_model_comparison.py; the figures and the table of Appendix A by
generate_chi3_figures.py.

Usage:
    python3 chi3_fft_verification.py [--limit N] [--xmin X] [--tol-pct P]
Defaults: --limit 500000000 --xmin 150 --tol-pct 3   (the configuration of the
paper; --limit 50000000 is much faster).  Memory at N = 5e8: about 5-6 GB.
"""
import sys
import gc
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal.windows import kaiser
from fft_explicit_formula_lib import (
    run_fft, predicted_phase, isolation_gap, bijective_match,
    calibrate_leakage_model, leakage_corrected_prediction,
)


# Zeros of L(s, chi_3), gamma > 0, computed with mpmath (30 digits);
# the first zero agrees with the LMFDB entry 1-3-3.2-r1-0-0.
KNOWN_ZEROS_CHI3 = np.array([
    8.039737156, 11.249206208, 15.704619177, 18.261997496, 20.455770808,
    24.059414856, 26.577868736, 28.218164506, 30.745040261, 33.897388927,
    35.608412654, 37.551796556, 39.485207261, 42.616379226, 44.120572912,
    46.274118024, 47.514104510,
])

ISOLATION_THRESHOLD = 2.0


def chi3(n: int) -> int:
    """Nontrivial quadratic character mod 3 (Legendre symbol (n/3))."""
    r = n % 3
    if r == 1:
        return 1
    elif r == 2:
        return -1
    return 0


def primes_up_to(limit: int) -> np.ndarray:
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.nonzero(sieve)[0]
    del sieve
    gc.collect()
    return primes


def psi_chi3(limit: int) -> np.ndarray:
    """
    psi(x, chi_3) = sum_{p^k <= x} chi_3(p^k) log p, cumulative for all x.

    chi_3 is completely multiplicative, so chi_3(p^k) = chi_3(p)**k: for
    p = 2 (mod 3) the sign alternates with k, and it would be wrong to use the
    constant chi_3(p) for every power.

    Every prime p adds chi_3(p) log p at x = p (k = 1, vectorised); the powers
    p^k, k >= 2, exist only for p <= sqrt(limit) and are added in a short loop.
    Each cell receives at most one addition.
    """
    psi = np.zeros(limit + 1, dtype=np.float64)
    primes = primes_up_to(limit)
    r = primes % 3
    chi = np.where(r == 1, 1.0, np.where(r == 2, -1.0, 0.0))
    psi[primes] += np.log(primes.astype(np.float64)) * chi          # k = 1
    for p in primes[primes <= math.isqrt(limit)]:                   # k >= 2
        c = chi3(int(p))
        if c == 0:  # p == 3
            continue
        logp = float(np.log(p))
        power = int(p) * int(p)
        k = 2
        while power <= limit:
            psi[power] += logp * (c ** k)
            power *= int(p)
            k += 1
    del primes, r, chi
    gc.collect()
    np.cumsum(psi, out=psi)
    return psi


def main():
    parser = argparse.ArgumentParser(description="L(s, chi_3) FFT verification")
    parser.add_argument("--limit", type=int, default=500_000_000)
    parser.add_argument("--xmin", type=int, default=150)
    parser.add_argument("--tol-pct", type=float, default=3.0,
                        help="matching tolerance in percent of gamma (paper: 3)")
    args = parser.parse_args()
    limit, xmin = args.limit, args.xmin
    known = KNOWN_ZEROS_CHI3

    print("=" * 90)
    print("L(s, chi_3) FFT VERIFICATION — position, amplitude, phase (one-to-one matching)")
    print("=" * 90)
    print(f"limit={limit:,}  xmin={xmin}  known zeros={len(known)}  tol={args.tol_pct}%")

    print("\nComputing psi(x, chi_3)...")
    delta = psi_chi3(limit)

    print("Applying FFT...")
    fr = run_fft(delta, limit, xmin)
    print(f"  Total local maxima above threshold: {len(fr['gammas_peaks'])}")

    assignment = bijective_match(known, fr["gammas_peaks"], max_pct=args.tol_pct)

    calib_j = None
    for j, cand_idx, pct in assignment:
        if cand_idx is None:
            continue
        if isolation_gap(known, known[j]) > ISOLATION_THRESHOLD:
            calib_j = j
            break
    if calib_j is None:
        print("ERROR: no isolated matched zero found for amplitude calibration.")
        sys.exit(1)

    _, calib_cand, _ = assignment[calib_j]
    gamma0 = known[calib_j]
    amp0 = fr["amps_peaks"][calib_cand]
    C = amp0 * np.sqrt(0.25 + gamma0**2)

    window = kaiser(fr["N"], beta=14.0)
    calib_k = fr["peak_k"][calib_cand]
    C_leak = calibrate_leakage_model(known, calib_j, calib_k, fr, window)
    print(f"\nCalibration zero gamma={gamma0:.4f}: |C_leak|={abs(C_leak):.5f}  "
          f"arg(C_leak)={np.angle(C_leak):+.5f} rad  (expected: |C_leak| = 1)")

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
        predicted_amp = C / np.sqrt(0.25 + gk**2)
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
    print(f"Position:  {n_matched}/{len(known)} known zeros matched within {args.tol_pct}% (one-to-one)")
    print(f"Amplitude (naive, window response ignored): mean ratio={np.mean(amp_ratios):.4f}  "
          f"std={np.std(amp_ratios):.4f}  (n={len(amp_ratios)}, calibration point excluded)")
    print(f"Phase (isolated zeros only, gap>{ISOLATION_THRESHOLD}): "
          f"mean|diff|={np.mean(np.abs(phase_diffs_isolated)):.4f} rad  "
          f"median|diff|={np.median(np.abs(phase_diffs_isolated)):.4f} rad  "
          f"(n={len(phase_diffs_isolated)}, includes the calibration zero)")
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

    # Diagnostic plot (not used in the paper): naive amplitude ratio and
    # phase error vs isolation gap.
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    gaps_amp = [isolation_gap(known, known[j]) for j, c, _ in assignment
                if c is not None and j != calib_j]
    axes[0].scatter(gaps_amp, amp_ratios, color="darkorange")
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Isolation gap (gamma units)")
    axes[0].set_ylabel("Amplitude / predicted (naive)")
    axes[0].set_title("chi_3: amplitude ratio vs isolation gap")
    axes[0].grid(alpha=0.3)

    gaps_phase = [isolation_gap(known, known[j]) for j, c, _ in assignment if c is not None]
    axes[1].scatter(gaps_phase, np.abs(phase_diffs_all), color="seagreen")
    axes[1].set_xlabel("Isolation gap (gamma units)")
    axes[1].set_ylabel("|phase error| (rad)")
    axes[1].set_title("chi_3: phase error vs isolation gap")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("chi3_fft_verification.png", dpi=150)
    print("  Saved: chi3_fft_verification.png")


if __name__ == "__main__":
    main()