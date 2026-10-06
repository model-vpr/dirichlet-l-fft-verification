#!/usr/bin/env python3
"""
make_zero_lists.py
==================
Regenerate the reference zero lists used by the scripts, with mpmath:

  chi3_zeros_below_160.txt   nontrivial zeros gamma > 0 of L(s, chi_3), gamma < 160
                             (chi_3 = (n/3)); 85 zeros
  zeta_reference_zeros.py    first 100 nontrivial zeros gamma > 0 of zeta(s)

L(s, chi_3): chi_3 is real, odd and primitive of conductor 3, so
    Lambda(s) = (3/pi)^{(s+1)/2} Gamma((s+1)/2) L(s, chi_3)
is real on the critical line (root number 1).  Zeros are located as sign
changes of Lambda(1/2 + it) on a grid of step 0.04 and refined with a secant
iteration at 30 digits.  The step is far below the smallest gap between zeros
in this range (about 0.6), so no pair is missed; the count is also compared
with the asymptotic formula N(T) ~ (T/2pi) log(qT/(2 pi e)) as a sanity check.

zeta(s): mpmath.zetazero.

Usage:
    python3 make_zero_lists.py [--tmax 160] [--step 0.04] [--chi3-out FILE] [--zeta-out FILE]
Requires: numpy, mpmath.  The first zero of L(s, chi_3), 8.039737..., agrees with
the LMFDB entry 1-3-3.2-r1-0-0.
"""
import argparse
import math

import numpy as np
from mpmath import mp, mpf, mpc, gamma, pi, dirichlet, findroot, zetazero


def chi3_zeros(tmax=160.0, step=0.04, dps=30):
    mp.dps = dps
    chi = [0, 1, -1]            # chi_3(0), chi_3(1), chi_3(2)

    def lam(t):
        s = mpc(0.5, t)
        return ((mpf(3) / pi) ** ((s + 1) / 2) * gamma((s + 1) / 2) * dirichlet(s, chi)).real

    ts = np.arange(1.0, tmax, step)
    vals = [lam(mpf(t)) for t in ts]
    zeros = []
    for i in range(len(ts) - 1):
        if vals[i] * vals[i + 1] < 0:
            z0 = findroot(lam, (mpf(ts[i]), mpf(ts[i + 1])), solver="illinois", tol=1e-20)
            z = findroot(lam, z0, solver="secant", tol=1e-25, maxsteps=50)
            zeros.append(float(z))
    zeros = np.array(zeros)
    assert np.all(np.diff(zeros) > 0), "zeros not strictly increasing"
    return zeros


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=160.0)
    ap.add_argument("--step", type=float, default=0.04)
    ap.add_argument("--chi3-out", type=str, default="chi3_zeros_below_160.txt")
    ap.add_argument("--zeta-out", type=str, default="zeta_reference_zeros.py")
    args = ap.parse_args()

    print(f"L(s, chi_3): zeros with 1 < gamma < {args.tmax} ...")
    z = chi3_zeros(args.tmax, args.step)
    q, T = 3.0, args.tmax
    expected = (T / (2 * math.pi)) * math.log(q * T / (2 * math.pi * math.e))
    print(f"  found {len(z)} zeros; asymptotic count (T/2pi) log(qT/2pi e) = {expected:.1f} "
          f"(agreement within a few units expected)")
    print(f"  first three: {z[:3]};  smallest gap: {np.min(np.diff(z)):.3f}")
    with open(args.chi3_out, "w") as f:
        f.write("# Nontrivial zeros gamma>0 of L(s, chi_3), chi_3=(n/3); computed with mpmath "
                f"(30 digits), gamma<{args.tmax:g}. Count={len(z)}\n")
        for g in z:
            f.write(f"{g:.9f}\n")
    print(f"  wrote {args.chi3_out}")

    print("zeta(s): first 100 zeros ...")
    mp.dps = 30
    zs = [zetazero(n).imag for n in range(1, 101)]
    lines = ["# REFERENCE_ZEROS for the zeta scripts: first 100 nontrivial zeros of zeta(s), gamma>0.",
             "# Computed with mpmath.zetazero (30 digits), printed with 15 decimals.",
             "import numpy as np", "", "REFERENCE_ZEROS = np.array(["]
    for i in range(0, 100, 3):
        lines.append("    " + ", ".join(f"{float(x):.15f}" for x in zs[i:i + 3]) + ",")
    lines += ["])", "",
              "assert len(REFERENCE_ZEROS) == 100 and np.all(np.diff(REFERENCE_ZEROS) > 0)", ""]
    with open(args.zeta_out, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {args.zeta_out}")


if __name__ == "__main__":
    main()