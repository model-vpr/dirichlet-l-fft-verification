# dirichlet-l-fft-verification

Companion code of the preprint

> S. Georgieva, *Closed-Form Phase Prediction and Spectral-Leakage Modeling for the Explicit Formula of L(s, χ₃), with Blind FFT Detection*, VPR Research Ltd, 2026.
> DOI: `10.5281/zenodo.XXXXXXX` *(Zenodo record of the preprint; replace after upload)*

The nontrivial zeros of an L-function appear as peaks in the logarithmic Fourier transform of the prime-power sum ψ(x, χ)/√x. This code builds a **parameter-free forward model of the complex FFT spectrum** of that signal from the zero list alone — a closed-form phase for each FFT bin (Eq. 8 of the paper), a window-corrected single-zero amplitude (Eq. 5), and a multi-zero leakage model using the exact DTFT of the Kaiser window (Eq. 11) — and checks it against the signal computed from the primes, for the quadratic character χ₃ (17 reference zeros, γ ≤ 47.5) and, as a replication, for ζ(s) (60 zeros, γ ≤ 163).

**What this is not.** It does not locate new zeros and does not test the Riemann hypothesis. Only the peak detection is blind; the matching to the reference zeros and the amplitude, phase, and leakage tests use the reference list (paper, Section 1.2). Real characters only.

## Contents

| File | Purpose |
|---|---|
| `fft_explicit_formula_lib.py` | Shared module: log-space resampling and windowed FFT, phase formula (Eq. 8), exact window DTFT (Eq. 10), multi-zero model (Eq. 11), isolation gap, one-to-one matching |
| `chi3_fft_verification.py` | L(s, χ₃): ψ(x, χ₃), peak detection, per-zero table (position, naive amplitude, phase, models A and C) |
| `chi3_model_comparison.py` | L(s, χ₃): models A–E, isolated/close split, amplitude with window factor, blind check against more zeros |
| `generate_chi3_figures.py` | Figures 1–2 and the per-zero table of Appendix A |
| `zeta_fft_verification.py` | ζ(s): per-zero table (position, naive amplitude, phase, models A and C) |
| `zeta_model_comparison.py` | ζ(s): models A–E, isolated/close split, split by height; accepts a list of `--xmin` values |
| `generate_zeta_figures.py` | Figures C.1–C.2 of Appendix C |
| `make_zero_lists.py` | Regenerates the zero lists below with mpmath |
| `chi3_zeros_below_160.txt` | The 85 zeros of L(s, χ₃) with γ < 160 (comparison list for blind checks) |
| `zeta_reference_zeros.py` | The first 100 zeros of ζ(s) (`REFERENCE_ZEROS`) |
| `requirements.txt` | Python dependencies |

All scripts must stay in the **same directory**: they import each other (`chi3_model_comparison.py`, `generate_chi3_figures.py` import from `chi3_fft_verification.py`; `generate_zeta_figures.py` imports from `zeta_fft_verification.py`; all import the library).

## Requirements

Python ≥ 3.8 with `numpy`, `scipy`, `matplotlib` (`pip install -r requirements.txt`). `mpmath` is only needed by `make_zero_lists.py`.

**Memory and time.** The prime sieve and the ψ array dominate: at N = 5·10⁸ a run needs about 5–6 GB of memory. Use `--limit 50000000` (N = 5·10⁷) first: it is much faster and reproduces the qualitative results, and is the right way to check your setup. The paper's main configuration is N = 5·10⁸, x_min = 150.

## Quick start

```bash
pip install -r requirements.txt

# L(s, chi_3): per-zero table and summary (fast check)
python3 chi3_fft_verification.py --limit 50000000 --xmin 150

# Models A-E and blind check against the 85 zeros below gamma = 160
python3 chi3_model_comparison.py --limit 50000000 --xmin 150 \
        --zeros-file chi3_zeros_below_160.txt --n-zeros 17

# zeta(s)
python3 zeta_model_comparison.py --limit 50000000 --xmin 150 --top-n 60 --tol-pct 3
```

## Reproducing the paper

All commands use the paper's main configuration (N = 5·10⁸, x_min = 150, 5% peak threshold, 3% matching tolerance) unless stated.

| Paper | Command |
|---|---|
| Fig. 1, Fig. 2, Table A.1 | `python3 generate_chi3_figures.py --zeros-file chi3_zeros_below_160.txt` (writes `figures/chi3_blind_test.png`, `figures/chi3_leakage_comparison.png`, prints Table A.1) |
| Sections 3.1–3.3 (position, naive amplitude, phase) | `python3 chi3_fft_verification.py` |
| Sections 3.2, 3.4 (amplitude with window factor; models A–E) | `python3 chi3_model_comparison.py --limit 500000000 --xmin 150 --zeros-file chi3_zeros_below_160.txt --n-zeros 17` |
| Section 3.1 (blind check against 85 zeros, 5·10⁷) | `python3 chi3_model_comparison.py --limit 50000000 --xmin 150 --zeros-file chi3_zeros_below_160.txt --n-zeros 17` (the *BLIND CHECK* block) |
| Table B.1 (ten configurations) | run `chi3_model_comparison.py` for N ∈ {5·10⁷, 5·10⁸} and x_min ∈ {50, 100, 150, 200, 300}; the phase column comes from `chi3_fft_verification.py` with the same `--limit`, `--xmin` |
| Section 4, 25-zero test | `python3 chi3_model_comparison.py --limit 50000000 --xmin 150 --zeros-file chi3_zeros_below_160.txt --n-zeros 25` |
| Tables C.1, C.2, Section 3.6 | `python3 zeta_model_comparison.py --limit 500000000 --xmin 150 --top-n 60 --tol-pct 3` |
| Table C.3 | `python3 zeta_model_comparison.py --limit <N> --xmin 50,100,150,200,300 --top-n 60 --tol-pct 3` for N = 50000000 and 500000000 (ψ is computed once per call) |
| Table C.4, Figs. C.1–C.2 | `python3 generate_zeta_figures.py` (the pair residuals of Table C.4 are in the per-zero table of `python3 zeta_fft_verification.py`) |

The sweep of Table B.1, for example:

```bash
for N in 50000000 500000000; do
  for X in 50 100 150 200 300; do
    python3 chi3_model_comparison.py --limit $N --xmin $X \
            --zeros-file chi3_zeros_below_160.txt --n-zeros 17 > chi3_N${N}_x${X}.txt
  done
done
```

Results are deterministic on a given platform; differences in the last printed digit between platforms or NumPy/SciPy versions are possible.

## Conventions

- **Reference zeros.** χ₃: the first 17 zeros (γ ≤ 47.514), hard-coded in `chi3_fft_verification.py`. ζ: the first 60 of `zeta_reference_zeros.py`. All values are computed with mpmath (`make_zero_lists.py`); the first zero of L(s, χ₃), 8.03974, agrees with the LMFDB entry 1-3-3.2-r1-0-0.
- **Peak detection.** Local maxima of |S[k]| above 5% of the largest amplitude; no information about the zeros is used.
- **Matching.** One-to-one assignment (Hungarian algorithm) of the reference zeros to all detected maxima, minimising the total |Δγ|; a zero is left unmatched if no free peak lies within `--tol-pct` percent of its γ (3% in the paper). Percent tolerances are loose at high γ and tight at low γ; deviations are also reported in FFT bins.
- **Isolation gap.** Distance to the nearest other reference zero. *Isolated*: gap > 2; *close*: gap < 2.
- **Calibration zero.** The first matched isolated zero (γ = 8.04 for χ₃, 14.13 for ζ) fixes the constants C and C_amp; it is excluded from the residual and amplitude statistics. Phase statistics for χ₃ include it (Eq. 8 has no fitted parameter).
- **Models.** A: single zero, window response ignored; B: single zero with window response; C: multi-zero, Eq. (11), C fitted; D, E: B and C with C = 1 (no fitted parameter). Residual = |S_meas − S_pred|/|S_meas| at the detected bin.
- **Statistics.** Standard deviations are population standard deviations (`ddof = 0`). Improvement factors in the paper are ratios of unrounded means.

## Zero lists

```bash
python3 make_zero_lists.py     # regenerates chi3_zeros_below_160.txt and zeta_reference_zeros.py
```

The regenerated files are identical to the ones in the repository.

## Known limitations

See Section 4 of the paper. In short: the reference set for χ₃ is the first 17 zeros; pairs of zeros closer than about two FFT bins are not separated by the peak finder (they are still described by the multi-zero model once the zeros are given); the residual floor of about 10⁻³ is not explained and grows roughly in proportion to γ; only real characters (χ₃) and ζ(s) are treated.

## Citation

```bibtex
@misc{georgieva2026chi3fft,
  author       = {Georgieva, Stefka},
  title        = {Closed-Form Phase Prediction and Spectral-Leakage Modeling for the
                  Explicit Formula of $L(s,\chi_3)$, with Blind FFT Detection},
  year         = {2026},
  publisher    = {VPR Research Ltd},
  doi          = {10.5281/zenodo.XXXXXXX}
}
```

## License

MIT

## Contact

Stefka Georgieva · VPR Research Ltd · georgieva@vpr-research.eu
