#!/usr/bin/env python3
"""
VPR RESEARCH Ltd
Figures for Appendix C (zeta(s)) of the preprint.

  figures/zeta_spectrum.png         FFT amplitude spectrum with detected peaks and the 60
                                    reference zeros (top), and two zooms on close pairs
                                    comparing the measured |S[k]| with the multi-zero
                                    model, Eq. (11)
  figures/zeta_residual_height.png  relative residual of models A, B, C versus gamma (a)
                                    and versus the isolation gap (b)

Usage:
    python3 generate_zeta_figures.py [--limit N] [--xmin X] [--top-n K] [--tol-pct P]
                                     [--gamma-max G] [--zoom1 LO,HI] [--zoom2 LO,HI] [--dpi D]
Defaults: --limit 500000000 --xmin 150 --top-n 60 --tol-pct 3 --gamma-max 170
          --zoom1 108,119 --zoom2 146,156 --dpi 200
Needs fft_explicit_formula_lib.py, zeta_fft_verification.py and zeta_reference_zeros.py
in the same directory.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal.windows import kaiser
from fft_explicit_formula_lib import (
    run_fft, predicted_phase, isolation_gap, bijective_match, window_dtft, _A0,
    calibrate_leakage_model, leakage_corrected_prediction,
)
from zeta_fft_verification import psi_minus_x, load_reference_zeros

ISOLATION_THRESHOLD = 2.0
C_SPEC, C_PEAK = "#4C72B0", "black"
C_REF, C_OTHER = "#DD8452", "#999999"
C_A, C_B, C_C = "#DD8452", "#55A868", "#4C72B0"


def analyse(limit, xmin, top_n, tol_pct):
    delta = psi_minus_x(limit)
    fr = run_fft(delta, limit, xmin)
    Z = load_reference_zeros()
    known = Z[:top_n]
    N, du, lm = fr["N"], fr["du"], fr["log_min"]
    window = kaiser(N, beta=14.0)
    dgamma = 2 * np.pi / fr["span"]

    asg = bijective_match(known, fr["gammas_peaks"], max_pct=tol_pct)
    cj = next(j for j, c, _ in asg
              if c is not None and isolation_gap(known, known[j]) > ISOLATION_THRESHOLD)
    _, cc, _ = asg[cj]
    C_leak = calibrate_leakage_model(known, cj, fr["peak_k"][cc], fr, window)
    C_amp = fr["amps_peaks"][cc] * np.sqrt(0.25 + known[cj] ** 2)

    rows = []
    for j, c, _ in asg:
        if c is None or j == cj:
            continue
        k = fr["peak_k"][c]
        m = fr["spectrum"][k]
        g = known[j]
        SA = C_amp / np.sqrt(0.25 + g * g) * np.exp(1j * predicted_phase(g, fr, k))
        Wj = window_dtft(window, 2 * np.pi * k / N - g * du)[0]
        SB = C_leak * _A0(g) * np.exp(1j * g * lm) * Wj
        SC = leakage_corrected_prediction(known, C_leak, k, fr, window)
        r = lambda S: abs(m - S) / abs(m)
        rows.append(dict(g=g, gap=isolation_gap(known, g), rA=r(SA), rB=r(SB), rC=r(SC)))
    unmatched = [float(known[j]) for j, c, _ in asg if c is None]
    return dict(fr=fr, window=window, known=known, Z=Z, C_leak=C_leak, rows=rows,
                dgamma=dgamma, unmatched=unmatched)


def plot_spectrum(D, limit, xmin, gamma_max, zoom1, zoom2, filename, dpi):
    fr, known, Z, window, C_leak = D["fr"], D["known"], D["Z"], D["window"], D["C_leak"]
    freqs, spectrum = fr["freqs"], fr["spectrum"]
    pos = np.where(freqs > 0)[0]
    g_bin = 2 * np.pi * freqs[pos]
    a_bin = np.abs(spectrum[pos])

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # ---- full spectrum ---------------------------------------------------
    m = g_bin <= gamma_max
    ax0.plot(g_bin[m], a_bin[m], lw=0.7, color=C_SPEC, label="FFT amplitude spectrum")
    gp, ap = fr["gammas_peaks"], fr["amps_peaks"]
    pm = gp <= gamma_max
    ax0.scatter(gp[pm], ap[pm], color=C_PEAK, s=12, zorder=5,
                label=f"Detected peaks ({pm.sum()} with $\\gamma \\leq {gamma_max:g}$; "
                      f"{len(gp)} in total)")
    for i, z in enumerate(known):
        ax0.axvline(z, color=C_REF, ls="--", lw=0.7, alpha=0.5,
                    label=f"Reference zeros ({len(known)})" if i == 0 else None)
    extra = [z for z in Z[len(known):] if z <= gamma_max]
    for i, z in enumerate(extra):
        ax0.axvline(z, color=C_OTHER, ls=":", lw=0.8, alpha=0.8,
                    label="Other known zeros" if i == 0 else None)
    for (lo, hi), col in [(zoom1, "tab:red"), (zoom2, "tab:red")]:
        ax0.axvspan(lo, hi, color=col, alpha=0.07)
    ax0.set_xlim(0, gamma_max)
    ax0.set_xlabel(r"$\gamma$")
    ax0.set_ylabel("FFT amplitude")
    ax0.set_title(f"BLIND FFT peak detection for $\\zeta(s)$ (N={limit:,}, $x_{{\\min}}$={xmin})")
    ax0.legend(loc="upper right", fontsize=8.5)
    ax0.grid(alpha=0.3)

    # ---- zooms: measured vs multi-zero model -------------------------------
    for ax, (lo, hi), tag in [(ax1, zoom1, "(b)"), (ax2, zoom2, "(c)")]:
        ks = pos[(g_bin >= lo) & (g_bin <= hi)]
        gk_ = 2 * np.pi * freqs[ks]
        meas = np.abs(spectrum[ks])
        pred = np.array([abs(leakage_corrected_prediction(known, C_leak, k, fr, window)) for k in ks])
        dev = np.max(np.abs(meas - pred)) / np.max(meas)
        print(f"  zoom {tag} [{lo:g},{hi:g}]: max |measured - model| / max measured = {100 * dev:.2f}% "
              f"({len(ks)} bins)")
        ax.plot(gk_, meas, "o-", color=C_SPEC, ms=4, lw=1.0, label="measured $|S[k]|$")
        ax.plot(gk_, pred, "s--", color=C_REF, ms=4, lw=1.0, mfc="none",
                label="multi-zero model, Eq. (11), $C$ fitted")
        ax.set_ylim(-0.03 * meas.max(), 1.32 * meas.max())
        trans = ax.get_xaxis_transform()
        for z in known[(known >= lo) & (known <= hi)]:
            ax.axvline(z, color="gray", ls=":", lw=0.9)
            ax.text(z, 0.03, f"{z:.2f}", transform=trans, rotation=90, va="bottom", ha="right",
                    fontsize=7.5, color="dimgray",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8))
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("FFT amplitude")
        ax.set_title(f"{tag} zoom, $\\gamma\\in[{lo:g},{hi:g}]$ (bin spacing {D['dgamma']:.3f})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"  Saved: {filename}")


def plot_residual(D, filename, dpi):
    rows = D["rows"]
    g = np.array([r["g"] for r in rows]); gap = np.array([r["gap"] for r in rows])
    rA = np.array([r["rA"] for r in rows]); rB = np.array([r["rB"] for r in rows])
    rC = np.array([r["rC"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    kw = dict(s=26, alpha=0.8)
    for ax, x, xl, ttl in [(axes[0], g, r"$\gamma$ (known zero)", r"(a) residual vs $\gamma$"),
                           (axes[1], gap, r"isolation gap (units of $\gamma$)", "(b) residual vs isolation gap")]:
        ax.scatter(x, rA, color=C_A, marker="o", label="A: single zero, no window", **kw)
        ax.scatter(x, rB, color=C_B, marker="^", label="B: single zero + window", **kw)
        ax.scatter(x, rC, color=C_C, marker="s", label="C: multi-zero", **kw)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlabel(xl)
        ax.set_ylabel("relative residual")
        ax.set_title(ttl)
        ax.grid(alpha=0.3, which="both")
    xs = np.linspace(50, g.max(), 50)
    axes[0].plot(xs, 3.7e-5 * xs, color="gray", ls="--", lw=1.2,
                 label=r"guide $3.7\times10^{-5}\,\gamma$")
    axes[0].legend(fontsize=8.5, loc="upper left")
    axes[1].axvline(ISOLATION_THRESHOLD, color="gray", ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"  Saved: {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500_000_000)
    ap.add_argument("--xmin", type=int, default=150)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--tol-pct", type=float, default=3.0)
    ap.add_argument("--gamma-max", type=float, default=170.0)
    ap.add_argument("--zoom1", type=str, default="108,119")
    ap.add_argument("--zoom2", type=str, default="146,156")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--outdir", type=str, default="figures")
    args = ap.parse_args()
    zoom1 = tuple(float(x) for x in args.zoom1.split(","))
    zoom2 = tuple(float(x) for x in args.zoom2.split(","))
    os.makedirs(args.outdir, exist_ok=True)

    print(f"limit={args.limit:,} xmin={args.xmin} top_n={args.top_n} tol={args.tol_pct}%")
    D = analyse(args.limit, args.xmin, args.top_n, args.tol_pct)
    print(f"  maxima above threshold: {len(D['fr']['gammas_peaks'])}  bin spacing = {D['dgamma']:.4f}  "
          f"|C_leak| = {abs(D['C_leak']):.5f}")
    if D["unmatched"]:
        print(f"  unmatched zeros: {[round(z, 3) for z in D['unmatched']]}")
    plot_spectrum(D, args.limit, args.xmin, args.gamma_max, zoom1, zoom2,
                  os.path.join(args.outdir, "zeta_spectrum.png"), args.dpi)
    plot_residual(D, os.path.join(args.outdir, "zeta_residual_height.png"), args.dpi)


if __name__ == "__main__":
    main()