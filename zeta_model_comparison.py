#!/usr/bin/env python3
"""
VPR RESEARCH Ltd
zeta_model_comparison.py
========================
Zeta counterpart of chi3_model_comparison.py: decomposes the residual of the
FFT model of  psi(x) - x  at the detected bin of each zero into

  A  single zero, no window response        (C_amp/|rho|) e^{i phase_Eq8}
  B  single zero, with window response      C A0 e^{i g u_min} W(w_k - w_j)
  C  multi-zero sum, fitted C
  E  multi-zero sum, C = 1 (no free parameter)

and reports them for isolated / close-neighbour zeros AND for gamma ranges
(the relative residual |S_meas - S_pred|/|S_meas| grows with gamma because
|A0| ~ 1/gamma, so comparisons with chi_3 must be made at equal gamma range).

Reference zeros: zeta_reference_zeros.py (first 100 zeros, mpmath) -- the
corrected list; keep it next to this script.

psi(x) - x is computed ONCE and reused for every x_min given, in place
(memory ~ 8 bytes * limit + the prime sieve; ~5 GB at limit = 5e8).

Usage:
    python3 zeta_model_comparison.py --limit 500000000 --xmin 50,100,200,300 \
            --top-n 60 --tol-pct 3
Options: --xmin (single value or comma-separated list)  --tol-pct (default 1.0)
         --gamma-bins "0,50,100,150,250"
"""
import argparse
import os
import runpy

import numpy as np
from scipy.signal.windows import kaiser

from fft_explicit_formula_lib import (
    run_fft, predicted_phase, isolation_gap, bijective_match,
    window_dtft, _A0, calibrate_leakage_model, leakage_corrected_prediction,
)

ISOLATION_THRESHOLD = 2.0


def load_reference_zeros():
    here = os.path.dirname(os.path.abspath(__file__))
    return runpy.run_path(os.path.join(here, "zeta_reference_zeros.py"))["REFERENCE_ZEROS"]


def primes_up_to(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def chebyshev_psi(limit):
    """psi(x) for every integer x <= limit, computed in place."""
    a = np.zeros(limit + 1)
    for p in primes_up_to(limit):
        lp = np.log(p)
        pw = int(p)
        while pw <= limit:
            a[pw] += lp
            if pw > limit // int(p):
                break
            pw *= int(p)
    np.cumsum(a, out=a)
    return a


def psi_minus_x(limit):
    """psi(x) - x, in place and in chunks (no extra full-size temporaries)."""
    a = chebyshev_psi(limit)
    step = 1 << 24
    for i in range(0, limit + 1, step):
        j = min(i + step, limit + 1)
        a[i:j] -= np.arange(i, j, dtype=float)
    return a


def analyze(delta, limit, xmin, known, args):
    fr = run_fft(delta, limit, xmin, threshold_ratio=args.threshold)
    N, du, lm = fr["N"], fr["du"], fr["log_min"]
    dg = 2 * np.pi / fr["span"]
    w = kaiser(N, beta=14.0)
    print(f"  maxima above threshold: {len(fr['gammas_peaks'])}   bin spacing = {dg:.4f}")

    asg = bijective_match(known, fr["gammas_peaks"], max_pct=args.tol_pct)
    unmatched = [(round(float(known[j]), 3), round(float(isolation_gap(known, known[j])), 2))
                 for j, c, _ in asg if c is None]
    print(f"  position: {len(known) - len(unmatched)}/{len(known)} matched within {args.tol_pct}%")
    if unmatched:
        print(f"  unmatched (gamma, isolation gap): {unmatched}")

    cj = next(j for j, c, _ in asg
              if c is not None and isolation_gap(known, known[j]) > ISOLATION_THRESHOLD)
    _, cc, _ = asg[cj]
    k0 = fr["peak_k"][cc]
    C_leak = calibrate_leakage_model(known, cj, k0, fr, w)
    C_amp = fr["amps_peaks"][cc] * np.sqrt(0.25 + known[cj] ** 2)
    print(f"  calibration zero gamma={known[cj]:.4f}   |C_leak|={abs(C_leak):.5f}  "
          f"arg={np.angle(C_leak):+.5f} rad")

    rows = []
    for j, c, _ in asg:
        if c is None or j == cj:
            continue
        k = fr["peak_k"][c]
        m = fr["spectrum"][k]
        g = known[j]
        amp_A = C_amp / np.sqrt(0.25 + g * g)
        ph = predicted_phase(g, fr, k)
        SA = amp_A * np.exp(1j * ph)
        Wj = window_dtft(w, 2 * np.pi * k / N - g * du)[0]
        SB = C_leak * _A0(g) * np.exp(1j * g * lm) * Wj
        SC = leakage_corrected_prediction(known, C_leak, k, fr, w)
        SE = leakage_corrected_prediction(known, 1.0, k, fr, w)
        phase_diff = (np.angle(m) - ph + np.pi) % (2 * np.pi) - np.pi
        r = lambda S: abs(m - S) / abs(m)
        gap = isolation_gap(known, g)
        rows.append((g, gap, r(SA), r(SB), r(SC), r(SE), abs(m) / amp_A, abs(m) / abs(SB), abs(phase_diff)))
    R = np.array(rows)
    gam, gap = R[:, 0], R[:, 1]
    iso = gap > ISOLATION_THRESHOLD

    def line(label, mask):
        if mask.sum() == 0:
            return
        print(f"  {label:<20} n={mask.sum():>3}  A={R[mask,2].mean():.4f}  B={R[mask,3].mean():.4f}  "
              f"C={R[mask,4].mean():.4f}  E={R[mask,5].mean():.4f}   "
              f"ratio_A={R[mask,6].mean():.4f}+-{R[mask,6].std():.4f}  "
              f"ratio_B={R[mask,7].mean():.4f}+-{R[mask,7].std():.4f}")

    print("\nMEAN RESIDUAL |S_meas - S_pred|/|S_meas| (calibration zero excluded)")
    line("all", np.ones(len(R), bool))
    line("isolated (gap>2)", iso)
    line("close (gap<=2)", ~iso)
    print("\n  by gamma range:")
    edges = [float(x) for x in args.gamma_bins.split(",")]
    for lo, hi in zip(edges[:-1], edges[1:]):
        line(f"gamma {lo:g}-{hi:g}", (gam >= lo) & (gam < hi))

    mA, mB, mC = R[:, 2].mean(), R[:, 3].mean(), R[:, 4].mean()
    print(f"\n  A->C {mA/mC:.1f}x   A->B {mA/mB:.1f}x   B->C {mB/mC:.1f}x   (all matched zeros)")
    print(f"  phase |diff|: isolated median={np.median(R[iso,8]):.4f} rad (n={iso.sum()})   "
          f"all median={np.median(R[:,8]):.4f} rad (n={len(R)})")
    print(f"  corr(resid_C, gamma) = {np.corrcoef(gam, R[:,4])[0,1]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50_000_000)
    ap.add_argument("--xmin", type=str, default="150",
                    help="x_min, or a comma-separated list (psi is computed once), e.g. 50,100,200")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--tol-pct", type=float, default=1.0)
    ap.add_argument("--gamma-bins", type=str, default="0,50,100,150,250")
    args = ap.parse_args()

    Z = load_reference_zeros()
    known = Z[: args.top_n]
    print("=" * 96)
    print("ZETA: MODEL COMPARISON  (A single | B single+W | C multi | E multi C=1)")
    print("=" * 96)
    print(f"limit={args.limit:,} threshold={args.threshold} reference zeros={len(known)} "
          f"(gamma {known[0]:.3f} .. {known[-1]:.3f}) tol={args.tol_pct}%")

    delta = psi_minus_x(args.limit)
    for xmin in [int(x) for x in args.xmin.split(",")]:
        print("\n" + "-" * 96)
        print(f"x_min = {xmin}")
        analyze(delta, args.limit, xmin, known, args)


if __name__ == "__main__":
    main()