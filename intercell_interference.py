"""
Impact of inter-cell interference on the AO-RA allocations
(response to Reviewer 2, Comment 1).

The optimization model is NOT modified. For the final AO-RA allocation
(x, a, p) of each realization, the rates are re-evaluated under three models:
  none      : noise-limited (what the paper reports)
  actual    : inter-cell interference from devices that really use the same
              PRB at the other O-RUs, with their AO-RA powers
  worstcase : the old bound: every device served by another O-RU transmits
              at P_max on every PRB (all-interferers-at-max-power)

Usage:
    python intercell_interference.py --runs 5     # quick test
    python intercell_interference.py              # 100 realizations, D in {9,18,27,36}
"""
import argparse
import os
import numpy as np
import pandas as pd

import ao_ra as m
import delay_components as dc

D_LIST = [9, 18, 27, 36]
MODES = ['none', 'actual', 'worstcase']


def ao_ra_with_allocation(D, H, z, lam, Tmax, Pmax):
    """Same loop as m.AO_RA (Algorithm 2), but also returns the best iterate's (x, a, p)."""
    R_min_dyn = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
    eta = np.array([max(R_min_dyn[z[d]], lam[d] + 1.0 / m.T_MAX[z[d]])
                    for d in range(D)])
    x = np.zeros((m.B, D), dtype=int)
    for d in range(D):
        x[int(np.argmax([np.mean(H[b, d, :]) for b in range(m.B)])), d] = 1
    h_dual, m_dual = np.zeros(D), np.zeros(m.B)
    history, best, best_U = [], None, -np.inf
    for _ in range(m.I_MAX):
        p, a, R, h_dual, m_dual = m.lagrangian_kkt(
            D, m.B, m.K, H, x, z, lam, Tmax, Pmax, eta, h_dual, m_dual)
        M_c, alpha_c = m.compute_vnf(z, R, lam, D)
        U = sum(m.W[z[d]] * R[d] for d in range(D))
        history.append(U)
        if U > best_U:
            best_U = U
            best = dict(x=x.copy(), a=a.copy(), p=p.copy(), R=R.copy(),
                        M_c=dict(M_c), alpha_c=dict(alpha_c))
        if len(history) > 1:
            if abs(history[-1] - history[-2]) / max(abs(history[-1]), 1.0) < m.EPS_CONV:
                break
        x = m.greedy_association(D, H, z, Pmax)
    return best, R_min_dyn


def evaluate(D, H, z, x, a, p, Pmax, mode):
    """Per-device total rate (Eq. 10) with SINR under the chosen interference model."""
    R = np.zeros(D)
    occ = -np.ones((m.B, m.K), dtype=int)           # occupant device of PRB k at O-RU b
    for b in range(m.B):
        for k in range(m.K):
            ds = np.where(a[b, :, k] > 0.5)[0]
            if len(ds):
                occ[b, k] = ds[0]
    sinr_loss_db = []
    for b in range(m.B):
        for k in range(m.K):
            d = occ[b, k]
            if d < 0:
                continue
            I = 0.0
            if mode == 'actual':
                for bp in range(m.B):
                    if bp != b and occ[bp, k] >= 0:
                        dp = occ[bp, k]
                        I += p[dp, k] * H[b, dp, k]
            elif mode == 'worstcase':
                for dp in range(D):
                    if x[b, dp] == 0:
                        I += Pmax[dp] * H[b, dp, k]
            snr = p[d, k] * H[b, d, k] / max(m.N0 + I, 1e-30)
            R[d] += m._rate(snr, z[d], d)
            if I > 0:
                sinr_loss_db.append(10 * np.log10((m.N0 + I) / m.N0))
    return R, sinr_loss_db


def run(n_runs):
    rows = []
    for D in D_LIST:
        valid, attempt, checked = 0, 0, False
        while valid < n_runs:
            attempt += 1
            sc = dc.valid_realization(D, attempt)      # same seeding / pre-check as run_monte_carlo
            if sc is None:
                continue
            devices, orus, H, z, lam, Tmax, Pmax = sc
            best, R_min = ao_ra_with_allocation(D, H, z, lam, Tmax, Pmax)
            if best is None or np.any(best['R'] == 0):
                continue
            if not checked:                            # self-check vs the original solver
                _, R_ref, _, _ = m.AO_RA(D, H, z, lam, Tmax, Pmax)
                diff = np.max(np.abs(R_ref - best['R']))
                print(f"  [check D={D}] max |R - R_AO_RA| = {diff:.3e}"
                      + ("" if diff < 1e-3 else "   <-- WARNING: mismatch"))
                checked = True
            valid += 1

            U_none = None
            for mode in MODES:
                R, loss_db = evaluate(D, H, z, best['x'], best['a'], best['p'], Pmax, mode)
                U = sum(m.W[z[d]] * R[d] for d in range(D))
                if mode == 'none':
                    U_none = U
                c4 = np.mean([R[d] < R_min[z[d]] for d in range(D)])
                c5 = np.mean([m.compute_delay(d, z[d], R, lam, best['M_c'],
                                              best['alpha_c']) > m.T_MAX[z[d]]
                              for d in range(D)])
                rows.append(dict(D=D, run=valid, mode=mode, U_Mbps=U / 1e6,
                                 loss_pct=100 * (U_none - U) / U_none,
                                 c4_viol_pct=100 * c4, c5_viol_pct=100 * c5,
                                 sinr_loss_dB=np.mean(loss_db) if loss_db else 0.0))
            if valid % 10 == 0:
                print(f"  D={D}: {valid}/{n_runs}")
        print(f"D={D} done ({attempt - valid} discarded of {attempt})")

    df = pd.DataFrame(rows)
    os.makedirs(m.RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(m.RESULTS_DIR, 'intercell_interference_runs.csv'), index=False)

    g = df.groupby(['D', 'mode']).agg(
        U_Mbps=('U_Mbps', 'mean'), loss_pct=('loss_pct', 'mean'),
        loss_std=('loss_pct', 'std'), c4=('c4_viol_pct', 'mean'),
        c5=('c5_viol_pct', 'mean'), sinr_dB=('sinr_loss_dB', 'mean')).reset_index()
    g.to_csv(os.path.join(m.RESULTS_DIR, 'intercell_interference_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print("\n=== SUMMARY ===")
    print(g.round(2).to_string(index=False))

    tex = []
    for D in D_LIST:
        r = {md: g[(g.D == D) & (g['mode'] == md)].iloc[0] for md in MODES}
        tex.append(f"{D} & {r['none'].U_Mbps:.1f} & {r['actual'].U_Mbps:.1f} "
                   f"({r['actual'].loss_pct:.1f}\\%) & {r['worstcase'].U_Mbps:.1f} "
                   f"({r['worstcase'].loss_pct:.1f}\\%) & {r['actual'].sinr_dB:.1f} & "
                   f"{r['actual'].c4:.1f} & {r['actual'].c5:.1f} \\\\")
    with open(os.path.join(m.RESULTS_DIR, 'intercell_table.tex'), 'w') as f:
        f.write("\n".join(tex))
    print("\n=== LaTeX rows ===")
    print("\n".join(tex))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=m.MC_RUNS)
    a = ap.parse_args()
    run(a.runs)