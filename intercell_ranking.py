"""
Ranking of AO-RA, DR and Baseline under actual inter-cell interference
(response to Reviewer 2, Comment 1). Allocations are NOT modified; each scheme's
final (x, a, p) is re-evaluated with and without inter-cell interference.

Usage:  python intercell_ranking.py
"""
import os
import numpy as np
import pandas as pd

import ao_ra as m
import delay_components as dc
from intercell_interference import ao_ra_with_allocation, evaluate

D_LIST = [9, 18, 27, 36]


def dr_with_allocation(D, H, z, lam, Tmax, Pmax):
    """Same loop as m.dr_scheme, but returns the best iterate's (x, a, p)."""
    x = np.zeros((m.B, D), dtype=int)
    for d in range(D):
        x[int(np.argmax([np.mean(H[b, d, :]) for b in range(m.B)])), d] = 1
    R_min = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
    eta = np.array([max(R_min[z[d]], lam[d] + 1.0 / m.T_MAX[z[d]]) for d in range(D)])
    h_dual, m_dual = np.zeros(D), np.zeros(m.B)
    best, best_U, prev_U = None, -np.inf, None
    for _ in range(m.I_MAX):
        p, a, R, h_dual, m_dual = m.lagrangian_kkt(
            D, m.B, m.K, H, x, z, lam, Tmax, Pmax, eta, h_dual, m_dual)
        U = sum(m.W[z[d]] * R[d] for d in range(D))
        if U > best_U:
            best_U, best = U, dict(x=x.copy(), a=a.copy(), p=p.copy(), R=R.copy())
        if prev_U is not None and abs(U - prev_U) / max(abs(U), 1.0) < m.EPS_CONV:
            break
        prev_U = U
    return best


def baseline_with_allocation(D, H, z, Pmax, devices, orus):
    """Same construction as m.baseline_scheme (consumes RNG identically)."""
    x = np.zeros((m.B, D), dtype=int)
    for d in range(D):
        x[int(np.argmin([np.linalg.norm(devices[d] - orus[b]) for b in range(m.B)])), d] = 1
    a = np.zeros((m.B, D, m.K))
    for b in range(m.B):
        served = [d for d in range(D) if x[b, d] == 1]
        pool = list(range(m.K))
        np.random.shuffle(pool)
        for idx, d in enumerate(served):
            if idx < m.K:
                a[b, d, pool[idx]] = 1
    p = np.zeros((D, m.K))
    for d in range(D):
        b_d = int(np.argmax(x[:, d]))
        for k in range(m.K):
            if a[b_d, d, k] > 0.5:
                p[d, k] = Pmax[d]
    R = m.compute_rates(D, m.B, m.K, H, x, z, p, a, Pmax)
    return dict(x=x, a=a, p=p, R=R)


def run(n_runs=m.MC_RUNS):
    rows = []
    for D in D_LIST:
        valid, attempt, checked = 0, 0, False
        while valid < n_runs:
            attempt += 1
            sc = dc.valid_realization(D, attempt)
            if sc is None:
                continue
            devices, orus, H, z, lam, Tmax, Pmax = sc
            ao, _ = ao_ra_with_allocation(D, H, z, lam, Tmax, Pmax)
            if ao is None or np.any(ao['R'] == 0):
                continue
            dr = dr_with_allocation(D, H, z, lam, Tmax, Pmax)
            st = np.random.get_state()
            ba = baseline_with_allocation(D, H, z, Pmax, devices, orus)
            if not checked:                          # self-check against the original schemes
                np.random.set_state(st)
                _, R_ba, _, _ = m.baseline_scheme(D, H, z, lam, Tmax, Pmax, devices, orus)
                _, R_dr, _, _ = m.dr_scheme(D, H, z, lam, Tmax, Pmax)
                print(f"  [check D={D}] baseline diff = {np.max(np.abs(R_ba - ba['R'])):.2e}, "
                      f"DR diff = {np.max(np.abs(R_dr - dr['R'])):.2e}")
                checked = True
            valid += 1
            for name, al in (('AO-RA', ao), ('DR', dr), ('Baseline', ba)):
                for mode in ('none', 'actual'):
                    R, _ = evaluate(D, H, z, al['x'], al['a'], al['p'], Pmax, mode)
                    U = sum(m.W[z[d]] * R[d] for d in range(D))
                    rows.append(dict(D=D, run=valid, scheme=name, mode=mode, U_Mbps=U / 1e6))
        print(f"D={D} done ({attempt - valid} discarded of {attempt})")

    df = pd.DataFrame(rows)
    os.makedirs(m.RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(m.RESULTS_DIR, 'intercell_ranking_runs.csv'), index=False)
    piv = df.pivot_table(index=['D', 'run'], columns=['scheme', 'mode'], values='U_Mbps')

    print("\n=== MEAN WEIGHTED SUM RATE (Mbps) and AO-RA gains ===")
    tex = []
    for D in D_LIST:
        s = piv.loc[D]
        mu = s.mean()
        out = {}
        for mode in ('none', 'actual'):
            g_dr = 100 * (mu[('AO-RA', mode)] - mu[('DR', mode)]) / mu[('DR', mode)]
            g_ba = 100 * (mu[('AO-RA', mode)] - mu[('Baseline', mode)]) / mu[('Baseline', mode)]
            wins = int((s[('AO-RA', mode)] > s[('DR', mode)] + 1e-9).sum())
            ties = int((abs(s[('AO-RA', mode)] - s[('DR', mode)]) <= 1e-9).sum())
            loss = int((s[('AO-RA', mode)] < s[('DR', mode)] - 1e-9).sum())
            out[mode] = (g_dr, g_ba, wins, ties, loss)
            print(f"D={D:2d} {mode:6s} AO-RA={mu[('AO-RA', mode)]:6.2f} DR={mu[('DR', mode)]:6.2f} "
                  f"Base={mu[('Baseline', mode)]:6.2f} | gain vs DR {g_dr:5.2f}% "
                  f"vs Base {g_ba:6.1f}% | W/T/L vs DR {wins}/{ties}/{loss}")
        tex.append(f"{D} & {mu[('AO-RA','actual')]:.1f} & {mu[('DR','actual')]:.1f} & "
                   f"{mu[('Baseline','actual')]:.1f} & {out['actual'][0]:.2f} & "
                   f"{out['actual'][1]:.1f} & {out['actual'][2]}/{out['actual'][3]}/{out['actual'][4]} \\\\")
    with open(os.path.join(m.RESULTS_DIR, 'intercell_ranking_table.tex'), 'w') as f:
        f.write("\n".join(tex))
    print("\n=== LaTeX rows ===")
    print("\n".join(tex))


if __name__ == '__main__':
    run()