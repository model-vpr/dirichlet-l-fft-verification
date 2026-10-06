#!/usr/bin/env python3
"""
VPR RESEARCH Ltd
chi3_model_comparison.py
========================
Decomposes the "leakage-model improvement" of the L(s, chi_3) FFT analysis
into its separate ingredients, and adds two stricter tests.

Models of the complex FFT coefficient S[k] at the detected bin k of zero j
(all residuals are |S_meas - S_pred| / |S_meas|):

  A  single, no window   : (C_amp/|rho_j|) * exp(i*phase_eq8)        <- current "single-zero"
  B  single + window     : C * A0_j e^{i g_j log_min} W(w_k - w_j)   (one zero, window response
                                                                       at the actual bin offset)
  C  multi-zero          : C * sum_i A0_i e^{i g_i log_min} W(w_k - w_i)   <- current "leakage model"
  D  single + window,  C = 1  (no free parameter)
  E  multi-zero,       C = 1  (no free parameter)

If A -> B carries almost all of the improvement, the gain comes from the
window response at the bin offset (scalloping), not from neighbouring zeros.
B -> C isolates the genuine multi-zero (neighbour) contribution.
D, E test the model with NO calibration at all.

Also reported:
  * amplitude ratio without and with the window factor (scalloping check);
  * relation between C_amp and C_leak (C_amp = |C_leak| * |W(offset_cal)|);
  * optional "blind" check against a zero list (--zeros-file): all detected
    peaks up to the last listed zero are compared with the list, so that
    unmatched peaks (false positives) and missed zeros are counted too.

Usage (put next to chi3_fft_verification.py and fft_explicit_formula_lib.py):
    python3 chi3_model_comparison.py --limit 50000000 --xmin 150 --threshold 0.05 \
            --zeros-file chi3_zeros_below_160.txt --n-zeros 17
"""
import argparse
import sys
import numpy as np
from scipy.signal.windows import kaiser

from fft_explicit_formula_lib import (
    run_fft, predicted_phase, isolation_gap, bijective_match,
    window_dtft, _A0, leakage_corrected_prediction,
)
from chi3_fft_verification import psi_chi3, KNOWN_ZEROS_CHI3

ISOLATION_THRESHOLD = 2.0


def load_zeros(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                vals.append(float(line))
    return np.array(vals)


def single_with_window(gj, k, fr, window, C):
    """C * A0(g_j) e^{i g_j log_min} W(w_k - w_j): one zero, exact window response."""
    N, du, log_min = fr["N"], fr["du"], fr["log_min"]
    omega_k = 2 * np.pi * k / N
    W = window_dtft(window, omega_k - gj * du)[0]
    return C * _A0(gj) * np.exp(1j * gj * log_min) * W, W


def stats(label, arr, mask_all, mask_iso, mask_close):
    def f(m):
        a = arr[m]
        return (f"{np.mean(a):.4f}/{np.median(a):.4f}" if len(a) else "   n/a   ")
    return (f"  {label:<34} all(n={mask_all.sum():>2}) {f(mask_all):>15}   "
            f"isolated(n={mask_iso.sum():>2}) {f(mask_iso):>15}   "
            f"close(n={mask_close.sum():>2}) {f(mask_close):>15}")


def blind_check(fr, zeros_file_zeros, du_gamma):
    """Compare ALL detected peaks (gamma <= last listed zero) with a zero list."""
    g_pk = fr["gammas_peaks"]
    zmax = zeros_file_zeros.max()
    in_range = g_pk <= zmax
    gp = g_pk[in_range]
    # distance (in bins) from each peak to its nearest listed zero
    d_bins = np.array([np.min(np.abs(zeros_file_zeros - g)) / du_gamma for g in gp])
    hit = d_bins <= 1.0
    print(f"\nBLIND CHECK vs zero list (peaks with gamma <= {zmax:.2f}; bin = {du_gamma:.4f} in gamma)")
    print(f"  detected peaks in range: {len(gp)}   within 1 bin of a listed zero: {hit.sum()}   "
          f"NOT matched: {(~hit).sum()}")
    if (~hit).sum():
        print("  unmatched peaks (gamma, distance in bins):",
              [(round(float(g), 3), round(float(d), 2)) for g, d in zip(gp[~hit], d_bins[~hit])])
    # zeros without a peak (ignore the last few near the upper edge)
    edge = zmax - 3.0
    z_in = zeros_file_zeros[zeros_file_zeros <= edge]
    miss = [float(z) for z in z_in if np.min(np.abs(g_pk - z)) / du_gamma > 1.0]
    print(f"  listed zeros up to {edge:.1f}: {len(z_in)}   without a peak within 1 bin: {len(miss)}"
          + (f"  -> {[round(m, 3) for m in miss]}" if miss else ""))
    print("  (a merged close pair can show up as a missing zero; threshold affects the high end)")


def main():
    ap = argparse.ArgumentParser(description="Single vs single+window vs multi-zero comparison")
    ap.add_argument("--limit", type=int, default=50_000_000)
    ap.add_argument("--xmin", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.05, help="peak threshold ratio (default 0.05)")
    ap.add_argument("--zeros-file", type=str, default=None,
                    help="text file with one zero per line (e.g. chi3_zeros_below_160.txt)")
    ap.add_argument("--n-zeros", type=int, default=17,
                    help="how many of the first zeros form the reference set (default 17)")
    args = ap.parse_args()

    zeros_all = load_zeros(args.zeros_file) if args.zeros_file else KNOWN_ZEROS_CHI3.copy()
    known = zeros_all[: args.n_zeros]
    print("=" * 96)
    print("L(s, chi_3): MODEL COMPARISON  (A single | B single+W | C multi | D single+W C=1 | E multi C=1)")
    print("=" * 96)
    print(f"limit={args.limit:,} xmin={args.xmin} threshold={args.threshold} reference zeros={len(known)} "
          f"(gamma {known[0]:.4f} .. {known[-1]:.4f})")

    print("\nComputing psi(x, chi_3) ...")
    delta = psi_chi3(args.limit)
    fr = run_fft(delta, args.limit, args.xmin, threshold_ratio=args.threshold)
    N, du = fr["N"], fr["du"]
    dgamma = 2 * np.pi / fr["span"]  # bin spacing in gamma
    window = kaiser(N, beta=14.0)
    print(f"  local maxima above threshold: {len(fr['gammas_peaks'])}   bin spacing = {dgamma:.4f}")

    assignment = bijective_match(known, fr["gammas_peaks"], max_pct=3.0)
    n_matched = sum(1 for _, c, _ in assignment if c is not None)
    print(f"  position: {n_matched}/{len(known)} reference zeros matched within 3% (bijective)")

    calib_j = None
    for j, c, _ in assignment:
        if c is not None and isolation_gap(known, known[j]) > ISOLATION_THRESHOLD:
            calib_j = j
            break
    if calib_j is None:
        print("ERROR: no isolated matched zero for calibration.")
        sys.exit(1)
    _, calib_cand, _ = assignment[calib_j]
    k_cal = fr["peak_k"][calib_cand]
    g_cal = known[calib_j]

    # calibrations (identical to the original scripts)
    C_amp = fr["amps_peaks"][calib_cand] * np.sqrt(0.25 + g_cal ** 2)
    omega_k = 2 * np.pi * k_cal / N
    W_cal = window_dtft(window, omega_k - g_cal * du)[0]
    C_leak = fr["spectrum"][k_cal] / (_A0(g_cal) * np.exp(1j * g_cal * fr["log_min"]) * W_cal)
    off_cal = (2 * np.pi * fr["freqs"][k_cal] - g_cal) / dgamma
    print(f"\nCalibration zero gamma={g_cal:.4f}, offset from bin centre = {off_cal:+.3f} bins")
    print(f"  C_amp = {C_amp:.6g}     |C_leak| = {abs(C_leak):.6f}   arg(C_leak) = {np.angle(C_leak):+.5f} rad")
    print(f"  window DC gain W(0) = {window.sum():.6g};  |W(offset_cal)| = {abs(W_cal):.6g}")
    print(f"  consistency: |C_leak|*|W(offset_cal)| = {abs(C_leak)*abs(W_cal):.6g}  vs  C_amp = {C_amp:.6g}")

    rows = []
    hdr = (f"\n{'gamma':>9} {'gap':>5} {'off(bins)':>9} {'amp/A':>7} {'amp/B':>7} "
           f"{'A':>8} {'B':>8} {'C':>8} {'D(C=1)':>8} {'E(C=1)':>8}  note")
    print(hdr)
    print("-" * len(hdr))
    for j, cand, _ in assignment:
        if cand is None:
            print(f"{known[j]:>9.4f}  -- no match --")
            continue
        gj = known[j]
        k = fr["peak_k"][cand]
        meas = fr["spectrum"][k]
        gap = isolation_gap(known, gj)
        off = (2 * np.pi * fr["freqs"][k] - gj) / dgamma

        # model A (existing "single-zero")
        pred_amp_A = C_amp / np.sqrt(0.25 + gj ** 2)
        S_A = pred_amp_A * np.exp(1j * predicted_phase(gj, fr, k))
        # model B (single zero, window response at actual offset)
        S_B, Wj = single_with_window(gj, k, fr, window, C_leak)
        # model C (multi-zero, fitted C)
        S_C = leakage_corrected_prediction(known, C_leak, k, fr, window)
        # models D, E (C = 1, no calibration)
        S_D, _ = single_with_window(gj, k, fr, window, 1.0)
        S_E = leakage_corrected_prediction(known, 1.0, k, fr, window)

        r = lambda S: abs(meas - S) / abs(meas)
        amp = abs(meas)
        ratio_A = amp / pred_amp_A
        ratio_B = amp / abs(S_B)
        note = "calibration" if j == calib_j else ("" if gap > ISOLATION_THRESHOLD else "close neighbor")
        rows.append(dict(j=j, g=gj, gap=gap, off=off, ratioA=ratio_A, ratioB=ratio_B,
                         A=r(S_A), B=r(S_B), C=r(S_C), D=r(S_D), E=r(S_E),
                         calib=(j == calib_j), iso=gap > ISOLATION_THRESHOLD))
        print(f"{gj:>9.4f} {gap:>5.2f} {off:>+9.3f} {ratio_A:>7.3f} {ratio_B:>7.3f} "
              f"{r(S_A):>8.4f} {r(S_B):>8.4f} {r(S_C):>8.4f} {r(S_D):>8.4f} {r(S_E):>8.4f}  {note}")

    use = [x for x in rows if not x["calib"]]
    g = lambda key: np.array([x[key] for x in use])
    m_all = np.ones(len(use), dtype=bool)
    m_iso = g("iso").astype(bool)
    m_close = ~m_iso

    print("\n" + "=" * 96)
    print("RESIDUAL |S_meas - S_pred|/|S_meas|   (mean / median; calibration zero excluded)")
    print("=" * 96)
    for key, label in [("A", "A  single, no window"),
                       ("B", "B  single + window (fitted C)"),
                       ("C", "C  multi-zero (fitted C)"),
                       ("D", "D  single + window, C=1"),
                       ("E", "E  multi-zero, C=1")]:
        print(stats(label, g(key), m_all, m_iso, m_close))

    mA, mB, mC = g("A").mean(), g("B").mean(), g("C").mean()
    print("\nDECOMPOSITION of the mean-residual improvement (all matched zeros):")
    print(f"  A -> C  (current claim, 'leakage model'): {mA/mC:6.1f}x")
    print(f"  A -> B  (window response of the SAME zero only): {mA/mB:6.1f}x")
    print(f"  B -> C  (genuine multi-zero / neighbour part):   {mB/mC:6.1f}x")
    if m_close.sum():
        cA, cB, cC = g("A")[m_close].mean(), g("B")[m_close].mean(), g("C")[m_close].mean()
        print(f"  close neighbours only:  A->C {cA/cC:5.1f}x   A->B {cA/cB:5.1f}x   B->C {cB/cC:5.1f}x")
    print(f"  no calibration at all (C=1): D mean={g('D').mean():.4f}   E mean={g('E').mean():.4f}")

    ra, rb = g("ratioA"), g("ratioB")
    print("\nAMPLITUDE RATIO measured/predicted (calibration excluded):")
    print(f"  without window factor (Eq. 5 as written): mean={ra.mean():.4f} std={ra.std():.4f} "
          f"min={ra.min():.4f} max={ra.max():.4f}")
    print(f"  with window factor  |C|/|rho|*|W(offset)|: mean={rb.mean():.4f} std={rb.std():.4f} "
          f"min={rb.min():.4f} max={rb.max():.4f}")
    off_abs = np.abs(g("off"))
    print(f"  corr(|1 - ratio_A|, |offset|) = {np.corrcoef(np.abs(1 - ra), off_abs)[0, 1]:.3f}   "
          f"corr(|1 - ratio_B|, |offset|) = {np.corrcoef(np.abs(1 - rb), off_abs)[0, 1]:.3f}")

    if args.zeros_file:
        blind_check(fr, zeros_all, dgamma)


if __name__ == "__main__":
    main()