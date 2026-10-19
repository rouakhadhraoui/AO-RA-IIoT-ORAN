"""
K_max_dev sensitivity sweep -- run on the ACTUAL ao_ra.py codebase.

This reproduces the numbers needed for the review response to reviewer
comment #2 (justification of K_max_dev = 3), using YOUR real simulation
code rather than a reconstruction, so the numbers are ground truth.

USAGE
-----
1. Save this file in the SAME FOLDER as ao_ra.py (project root, next to
   generate_figures.py and verify_constraints.py).
2. Run:  python kmax_sensitivity_sweep.py
3. Copy the FULL console output and paste it back.

Expected runtime: a few minutes (D=18, 30 valid Monte Carlo runs per
K_max_dev value, 3 values tested = 90 total AO-RA runs).

If your main model file is not literally named ao_ra.py, change the
import line below to match its actual filename.
"""

import time
import numpy as np
import ao_ra as m   # <-- adjust this import if your main file has a different name

D_FIXED  = 18        # same device count used in the paper's power sweep (Fig. 4)
N_RUNS   = 30         # valid Monte Carlo realizations per K_max_dev value
K_VALUES = [1, 2, 3, 4, 5, 8, 25]  # the three values needed for the review response


def run_one(kmax):
    """Run one AO-RA realization at the given PRB cap. Returns None if the
    realization is infeasible or a device ends up starved (rate = 0),
    mirroring the discard logic in run_monte_carlo()."""
    m.MAX_PRB_PER_DEVICE = kmax

    devices, orus = m.generate_positions(D_FIXED)
    H = m.generate_channels(devices, orus, D_FIXED)
    z = m.assign_service_classes(D_FIXED)
    lam, Tmax, Pmax = m.get_device_params(z, D_FIXED)

    # feasibility pre-check (same as run_monte_carlo)
    for d in range(D_FIXED):
        b_fa = int(np.argmin([np.linalg.norm(devices[d] - orus[b])
                              for b in range(m.B)]))
        s_fa = Pmax[d] * H[b_fa, d, d % m.K] / m.N0
        if m._rate(s_fa, z[d], d) <= lam[d]:
            return None

    result = m.AO_RA(D_FIXED, H, z, lam, Tmax, Pmax)
    R_d = result[1]
    if R_d is None or np.any(R_d == 0):
        return None

    U = sum(m.W[z[d]] * R_d[d] for d in range(D_FIXED)) / 1e6
    devs_mmtc = [d for d in range(D_FIXED) if z[d] == 'mMTC']
    mmtc_rate = float(np.mean([R_d[dd] for dd in devs_mmtc])) / 1e6
    return U, mmtc_rate


results = {}
t_start = time.time()

for kmax in K_VALUES:
    U_vals, mmtc_vals = [], []
    valid, attempt = 0, 0
    while valid < N_RUNS:
        attempt += 1
        np.random.seed(attempt * 1000 + D_FIXED)   # paired seeding across K values
        out = run_one(kmax)
        if out is None:
            continue
        U_vals.append(out[0])
        mmtc_vals.append(out[1])
        valid += 1

    U_arr, R_arr = np.array(U_vals), np.array(mmtc_vals)
    n = len(U_arr)
    results[kmax] = dict(
        U_mean=U_arr.mean(), U_sem=U_arr.std(ddof=1) / np.sqrt(n),
        mmtc_mean=R_arr.mean(), mmtc_sem=R_arr.std(ddof=1) / np.sqrt(n),
        n=n,
    )
    r = results[kmax]
    print(f"K_max_dev={kmax:2d}  "
          f"U = {r['U_mean']:.2f} +/- {r['U_sem']:.2f} Mbps (SEM, n={n})   "
          f"mMTC_rate = {r['mmtc_mean']:.2f} +/- {r['mmtc_sem']:.2f} Mbps")

print(f"\nTotal time: {time.time() - t_start:.1f}s\n")

base = results[3]
for k in K_VALUES:
    if k == 3:
        continue
    o = results[k]
    diffU = o['U_mean'] - base['U_mean']
    seU   = np.sqrt(base['U_sem']**2 + o['U_sem']**2)
    diffM = base['mmtc_mean'] - o['mmtc_mean']
    seM   = np.sqrt(base['mmtc_sem']**2 + o['mmtc_sem']**2)
    pctM  = diffM / base['mmtc_mean'] * 100
    print(f"K=3 vs K={k:2d}:  "
          f"U diff = {diffU:+.2f} Mbps (z = {diffU/seU:.2f})   |   "
          f"mMTC drop = {diffM:+.2f} Mbps ({pctM:+.1f}%, z = {diffM/seM:.2f})")

k_ref = max(K_VALUES)
for k in (8, k_ref):
    pct = base['U_mean'] / results[k]['U_mean'] * 100
    print(f"K=3 captures {pct:.1f}% of the weighted sum rate reached at K={k}")

