#!/usr/bin/env python3
"""
VPR RESEARCH Ltd
Generate figures (and the per-zero table) for the L(s, chi_3) FFT preprint.

  figures/chi3_blind_test.png          Fig. 1: FFT spectrum, detected peaks, known zeros
  figures/chi3_leakage_comparison.png  Fig. 2: residuals of models A, B, C
                                       (a) vs gamma, (b) vs isolation gap,
                                       (c) vs distance (in bins) from the detected bin
                                           to the nearest NEIGHBOURING zero

Models of the complex coefficient S[k] at the detected bin k of zero j
(residual = |S_meas - S_pred| / |S_meas|):
  A  single zero, no window response        (C_amp/|rho|) e^{i phase_Eq8}
  B  single zero, with window response      C A0_j e^{i g_j log_min} W(w_k - w_j)
  C  multi-zero, with window response       C sum_i A0_i e^{i g_i log_min} W(w_k - w_i)

Usage:
    python3 generate_chi3_figures.py [--limit N] [--xmin X] [--threshold T]
                                     [--zeros-file chi3_zeros_below_160.txt]
                                     [--gamma-max G] [--dpi D]
Defaults: --limit 500000000 --xmin 150 --threshold 0.05 --gamma-max 55 --dpi 200
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal.windows import kaiser
from fft_explicit_formula_lib import (
    run_fft, predicted_phase, isolation_gap, bijective_match,
    calibrate_leakage_model, leakage_corrected_prediction,
    window_dtft, _A0,
)
from chi3_fft_verification import psi_chi3, KNOWN_ZEROS_CHI3

ISOLATION_THRESHOLD = 2.0

C_SPEC, C_PEAK = "#4C72B0", "black"
C_REF, C_OTHER = "#DD8452", "#999999"
C_A, C_B, C_C = "#DD8452", "#55A868", "#4C72B0"   # orange / green / blue


def load_zeros(path):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(float(s))
    return np.array(out)


def run_full_analysis(limit, xmin, threshold):
    """Run the pipeline once and collect per-zero data for the figures/table."""
    delta = psi_chi3(limit)
    fr = run_fft(delta, limit, xmin, threshold_ratio=threshold)
    known = KNOWN_ZEROS_CHI3
    N, du, log_min = fr["N"], fr["du"], fr["log_min"]
    dgamma = 2 * np.pi / fr["span"]
    window = kaiser(N, beta=14.0)

    assignment = bijective_match(known, fr["gammas_peaks"], max_pct=3.0)

    calib_j = None
    for j, cand_idx, _ in assignment:
        if cand_idx is not None and isolation_gap(known, known[j]) > ISOLATION_THRESHOLD:
            calib_j = j
            break
    if calib_j is None:
        raise RuntimeError("no isolated matched zero for calibration")
    _, calib_cand, _ = assignment[calib_j]
    calib_k = fr["peak_k"][calib_cand]

    C_leak = calibrate_leakage_model(known, calib_j, calib_k, fr, window)
    g0 = known[calib_j]
    C_amp = fr["amps_peaks"][calib_cand] * np.sqrt(0.25 + g0 ** 2)

    rows = []
    for j, cand_idx, _ in assignment:
        gk = known[j]
        gap = isolation_gap(known, gk)
        isolated = gap > ISOLATION_THRESHOLD
        if cand_idx is None:
            rows.append(dict(gk=gk, g_det=None, gap=gap, isolated=isolated,
                             is_calibration=False, note="NO MATCH"))
            continue

        k = fr["peak_k"][cand_idx]
        g_bin = 2 * np.pi * fr["freqs"][k]
        meas = fr["spectrum"][k]
        off_bins = (g_bin - gk) / dgamma
        others = np.delete(known, j)
        nb_dist_bins = float(np.min(np.abs(g_bin - others)) / dgamma)

        amp_A = C_amp / np.sqrt(0.25 + gk ** 2)
        ph_pred = predicted_phase(gk, fr, k)
        S_A = amp_A * np.exp(1j * ph_pred)
        omega_k = 2 * np.pi * k / N
        Wj = window_dtft(window, omega_k - gk * du)[0]
        S_B = C_leak * _A0(gk) * np.exp(1j * gk * log_min) * Wj
        S_C = leakage_corrected_prediction(known, C_leak, k, fr, window)

        ph_meas = np.angle(meas)
        ph_diff = (ph_meas - ph_pred + np.pi) % (2 * np.pi) - np.pi
        is_cal = (j == calib_j)
        rows.append(dict(
            gk=gk, g_det=g_bin, gap=gap, isolated=isolated,
            off_bins=off_bins, nb_dist_bins=nb_dist_bins,
            ratio_A=abs(meas) / amp_A, ratio_B=abs(meas) / abs(S_B),
            phase_diff=ph_diff,
            rA=abs(meas - S_A) / abs(meas),
            rB=abs(meas - S_B) / abs(meas),
            rC=abs(meas - S_C) / abs(meas),
            is_calibration=is_cal,
            note="calibration" if is_cal else ("" if isolated else "close neighbor"),
        ))
    return fr, rows, C_leak, C_amp, window, dgamma


def plot_blind_test(fr, limit, xmin, threshold, gamma_max, extra_zeros, filename, dpi):
    """Figure 1: spectrum, detected peaks, reference zeros (+ other known zeros)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    freqs, spectrum = fr["freqs"], fr["spectrum"]
    pos = np.where(freqs > 0)[0]
    g = 2 * np.pi * freqs[pos]
    a = np.abs(spectrum[pos])
    m = g <= gamma_max
    ax.plot(g[m], a[m], lw=0.7, color=C_SPEC, label="FFT amplitude spectrum")

    gp, ap = fr["gammas_peaks"], fr["amps_peaks"]
    pm = gp <= gamma_max
    ax.scatter(gp[pm], ap[pm], color=C_PEAK, s=16, zorder=5,
               label=f"Detected peaks ({pm.sum()} with $\\gamma \\leq {gamma_max:g}$; "
                     f"{len(gp)} in total, threshold {threshold:.0%} of max)")

    first = True
    for z in KNOWN_ZEROS_CHI3:
        if z <= gamma_max:
            ax.axvline(z, color=C_REF, ls="--", lw=1.0, alpha=0.6,
                       label=r"Reference set (17 zeros)" if first else None)
            first = False
    if extra_zeros is not None:
        extra = [z for z in extra_zeros
                 if z <= gamma_max and np.min(np.abs(KNOWN_ZEROS_CHI3 - z)) > 1e-3]
        first = True
        for z in extra:
            ax.axvline(z, color=C_OTHER, ls=":", lw=1.0, alpha=0.8,
                       label="Other known zeros (not in reference set)" if first else None)
            first = False

    ax.set_xlim(0, gamma_max)
    ax.set_xlabel(r"$\gamma$", fontsize=12)
    ax.set_ylabel("FFT amplitude", fontsize=12)
    ax.set_title(f"BLIND FFT peak detection for $L(s, \\chi_3)$ "
                 f"(N={limit:,}, $x_{{\\min}}$={xmin})", fontsize=12)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"  Saved: {filename}")


def plot_leakage_comparison(rows, window, filename, dpi):
    """Figure 2: residuals of models A, B, C."""
    data = [r for r in rows if r.get("rA") is not None and not r["is_calibration"]]
    gam = np.array([r["gk"] for r in data])
    gap = np.array([r["gap"] for r in data])
    nb = np.array([r["nb_dist_bins"] for r in data])
    rA = np.array([r["rA"] for r in data])
    rB = np.array([r["rB"] for r in data])
    rC = np.array([r["rC"] for r in data])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    kw = dict(s=34, alpha=0.8)

    for ax, x, xl, ttl in [
        (axes[0], gam, r"$\gamma$ (known zero)", r"(a) residual vs $\gamma$"),
        (axes[1], gap, r"isolation gap (units of $\gamma$)", "(b) residual vs isolation gap"),
    ]:
        ax.scatter(x, rA, color=C_A, marker="o", label="A: single zero, no window", **kw)
        ax.scatter(x, rB, color=C_B, marker="^", label="B: single zero + window", **kw)
        ax.scatter(x, rC, color=C_C, marker="s", label="C: multi-zero", **kw)
        ax.set_yscale("log")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel("relative residual", fontsize=11)
        ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=0.3, which="both")
    axes[1].axvline(ISOLATION_THRESHOLD, color="gray", ls=":", alpha=0.6)
    for a_ in axes[:2]:
        a_.set_ylim(2e-5, 0.8)   # място за легендата, да не закрива точки
    axes[0].legend(fontsize=8.5, loc="upper left")

    # (c) mechanism: B and C vs distance (bins) to nearest neighbouring zero
    ax = axes[2]
    d_curve = np.linspace(0.0, 6.0, 61)
    om = d_curve * 2 * np.pi / len(window)
    Wc = np.abs(window_dtft(window, om)) / window.sum()
    ax.plot(d_curve, Wc, color="gray", ls="--", lw=1.2,
            label=r"Kaiser response $|W(d)|/|W(0)|$")
    ax.scatter(nb, rB, color=C_B, marker="^", label="B: single zero + window", **kw)
    ax.scatter(nb, rC, color=C_C, marker="s", label="C: multi-zero", **kw)
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 1.0)
    ax.set_xlabel("distance of detected bin to nearest neighbouring zero (bins)", fontsize=11)
    ax.set_ylabel("relative residual", fontsize=11)
    ax.set_title("(c) neighbour contribution follows the window response", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right") 
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"  Saved: {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500_000_000)
    ap.add_argument("--xmin", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--zeros-file", type=str, default=None)
    ap.add_argument("--gamma-max", type=float, default=55.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--outdir", type=str, default="figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    extra = load_zeros(args.zeros_file) if args.zeros_file else None
    if extra is None and args.gamma_max > KNOWN_ZEROS_CHI3.max() + 1.0:
        print("WARNING: gamma-max exceeds the reference list but no --zeros-file was given; "
              "peaks beyond the 17th zero will appear without a marker for the zero.")

    print("=" * 90)
    print("GENERATING FIGURES FOR L(s, chi_3)")
    print("=" * 90)
    print(f"limit={args.limit:,}  xmin={args.xmin}  threshold={args.threshold}")

    fr, rows, C_leak, C_amp, window, dgamma = run_full_analysis(
        args.limit, args.xmin, args.threshold)
    print(f"  maxima above threshold: {len(fr['gammas_peaks'])}   bin spacing = {dgamma:.4f}")
    print(f"  |C_leak| = {abs(C_leak):.5f}  arg(C_leak) = {np.angle(C_leak):+.5f} rad")

    plot_blind_test(fr, args.limit, args.xmin, args.threshold, args.gamma_max, extra,
                    os.path.join(args.outdir, "chi3_blind_test.png"), args.dpi)
    plot_leakage_comparison(rows, window,
                            os.path.join(args.outdir, "chi3_leakage_comparison.png"), args.dpi)

    print("\n" + "=" * 118)
    print("PER-ZERO TABLE (Appendix A)   A: single, no window | B: single + window | C: multi-zero")
    print("=" * 118)
    print(f"{'gamma_known':>11} {'gamma_det':>10} {'gap':>5} {'off(bins)':>9} {'nb(bins)':>8} "
          f"{'amp_A':>7} {'amp_B':>7} {'phase_diff':>10} {'res_A':>7} {'res_B':>7} {'res_C':>7}  note")
    print("-" * 118)
    for r in rows:
        if r["g_det"] is None:
            print(f"{r['gk']:>11.4f}  --- no match ---")
            continue
        print(f"{r['gk']:>11.4f} {r['g_det']:>10.4f} {r['gap']:>5.2f} {r['off_bins']:>+9.3f} "
              f"{r['nb_dist_bins']:>8.2f} {r['ratio_A']:>7.4f} {r['ratio_B']:>7.4f} "
              f"{r['phase_diff']:>+10.4f} {r['rA']:>7.4f} {r['rB']:>7.4f} {r['rC']:>7.4f}  {r['note']}")


if __name__ == "__main__":
    main()