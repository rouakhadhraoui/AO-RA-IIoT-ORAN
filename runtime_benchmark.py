"""
runtime_benchmark.py -- wall-clock runtime of AO-RA (Reviewer 2, Comment 5)

Measures the complete AO-RA algorithm (Algorithm 2) for D in {9,18,27,36}
over the same 100 valid realizations as run_monte_carlo (same seeds, same
feasibility pre-check, same "no starved device" rule), and breaks the time
down by sub-problem / O-RAN node:

    SP1 pre-processing : compute_dynamic_rmin + eta_d + initial association
                         (Near-RT RIC, once per run)
    SP3                : lagrangian_kkt           (O-DU, per outer iteration,
                                                   includes T_INNER=20 inner steps)
    SP4                : compute_vnf              (O-CU-UP/UPF, per iteration)
    SP2                : greedy_association (GAA) (Near-RT RIC, per iteration)

Excluded from timing: scenario/channel generation, benchmark schemes,
plotting, logging. The loop below mirrors ao_ra.AO_RA line by line; the
script asserts that it returns the same weighted sum rate as ao_ra.AO_RA.

Usage:  python runtime_benchmark.py            (100 runs per D)
        python runtime_benchmark.py 20         (quick test, 20 runs per D)
"""
import os
# Single-threaded BLAS, set BEFORE numpy is imported (fair, reproducible timing)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import sys, time, platform
import numpy as np
import pandas as pd
import ao_ra as A

D_LIST = (9, 18, 27, 36)
N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED_BASE = 0
clock = time.perf_counter


def timed_aora(D, H, z, lam, Tmax, Pmax):
    """Mirror of ao_ra.AO_RA with a timer around every sub-problem."""
    t = dict(sp1=0.0, sp3=0.0, sp4=0.0, sp2=0.0)
    t_all0 = clock()

    # --- SP1 pre-processing (Near-RT RIC) ---
    t0 = clock()
    R_min_dyn = A.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
    eta_d = np.array([max(R_min_dyn[z[d]], lam[d] + 1.0 / A.T_MAX[z[d]])
                      for d in range(D)])
    x = np.zeros((A.B, D), dtype=int)
    for d in range(D):
        x[int(np.argmax([np.mean(H[b, d, :]) for b in range(A.B)])), d] = 1
    h_dual, m_dual = np.zeros(D), np.zeros(A.B)
    t["sp1"] += clock() - t0

    history, best_U, best_R = [], -np.inf, None
    n_iter = 0
    for i in range(A.I_MAX):
        n_iter += 1
        t0 = clock()                                    # SP3 (O-DU)
        p, a, R_d, h_dual, m_dual = A.lagrangian_kkt(
            D, A.B, A.K, H, x, z, lam, Tmax, Pmax, eta_d, h_dual, m_dual)
        t["sp3"] += clock() - t0

        t0 = clock()                                    # SP4 (O-CU-UP/UPF)
        M_c, alpha_c = A.compute_vnf(z, R_d, lam, D)
        t["sp4"] += clock() - t0

        U = sum(A.W[z[d]] * R_d[d] for d in range(D))
        history.append(U)
        if U > best_U:
            best_U, best_R = U, R_d.copy()
        if len(history) > 1:
            if abs(history[-1] - history[-2]) / max(abs(history[-1]), 1.0) < A.EPS_CONV:
                break

        t0 = clock()                                    # SP2 (Near-RT RIC)
        x = A.greedy_association(D, H, z, Pmax)
        t["sp2"] += clock() - t0

    t["total"] = clock() - t_all0
    t["iters"] = n_iter
    return best_U, best_R, t


def valid_scenario(D, attempt):
    """Same seeding / pre-check as run_monte_carlo. Returns None if rejected."""
    np.random.seed(SEED_BASE + attempt * 1000 + D)
    devices, orus = A.generate_positions(D)
    H = A.generate_channels(devices, orus, D)
    z = A.assign_service_classes(D)
    lam, Tmax, Pmax = A.get_device_params(z, D)
    for d in range(D):
        b_fa = int(np.argmin([np.linalg.norm(devices[d] - orus[b]) for b in range(A.B)]))
        snr_fa = Pmax[d] * H[b_fa, d, d % A.K] / A.N0
        if A._rate(snr_fa, z[d], d) <= lam[d]:
            return None
    return H, z, lam, Tmax, Pmax


def main():
    print(f"Machine : {platform.processor() or platform.machine()} | {platform.platform()}")
    print(f"Python  : {platform.python_version()} | numpy {np.__version__} | "
          f"cores visible: {os.cpu_count()} | BLAS threads: 1")
    print(f"Runs/D  : {N_RUNS}\n")

    # Warm-up (imports, scipy norm.ppf first call, caches); discarded
    sc = None; k = 0
    while sc is None:
        k += 1; sc = valid_scenario(9, k)
    timed_aora(9, *sc)

    rows = []
    for D in D_LIST:
        valid, attempt, checked = 0, 0, False
        while valid < N_RUNS:
            attempt += 1
            sc = valid_scenario(D, attempt)
            if sc is None:
                continue
            H, z, lam, Tmax, Pmax = sc
            U, R, t = timed_aora(D, H, z, lam, Tmax, Pmax)
            if R is None or np.any(R == 0):         # starved device -> discard
                continue
            if not checked:                          # correctness check, untimed
                _, R_ref, _, _ = A.AO_RA(D, H, z, lam, Tmax, Pmax)
                U_ref = sum(A.W[z[d]] * R_ref[d] for d in range(D))
                assert abs(U - U_ref) < 1e-6 * max(U_ref, 1.0), "timed loop != AO_RA"
                checked = True
            rows.append(dict(D=D, **t)); valid += 1
        print(f"D={D}: {valid} runs ({attempt} attempts)")

    df = pd.DataFrame(rows)
    df.to_csv("runtime_raw.csv", index=False)
    ms = lambda s: s * 1e3
    out = []
    for D, g in df.groupby("D"):
        out.append(dict(
            D=D,
            iters=g.iters.mean(),
            total_mean_ms=ms(g.total.mean()),
            total_p95_ms=ms(np.percentile(g.total, 95)),
            total_max_ms=ms(g.total.max()),
            SP1_ms=ms(g.sp1.mean()),
            SP2_ms=ms(g.sp2.mean()),
            SP3_ms=ms(g.sp3.mean()),
            SP4_ms=ms(g.sp4.mean()),
            SP3_per_call_ms=ms((g.sp3 / g.iters).mean()),
            SP2_per_call_ms=ms((g.sp2 / (g.iters - 1).clip(lower=1)).mean()),
            ric_ms=ms((g.sp1 + g.sp2).mean()),      # Near-RT RIC share: SP1+SP2
        ))
    s = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print("\n=== Runtime of AO-RA (ms) ===")
    print(s.round(2).to_string(index=False))
    s.to_csv("runtime_summary.csv", index=False)

    print("\n=== LaTeX: total runtime ===")
    for _, r in s.iterrows():
        print(f"{int(r.D)} & {r.iters:.2f} & {r.total_mean_ms:.0f} & "
              f"{r.total_p95_ms:.0f} & {r.total_max_ms:.0f} \\\\")
    print("\n=== LaTeX: breakdown (mean, ms) ===")
    for _, r in s.iterrows():
        print(f"{int(r.D)} & {r.SP1_ms:.0f} & {r.SP2_ms:.1f} & "
              f"{r.SP3_ms:.0f} & {r.SP4_ms:.1f} & {r.SP3_per_call_ms:.0f} \\\\")
    print("\nSaved runtime_raw.csv and runtime_summary.csv")


if __name__ == "__main__":
    main()