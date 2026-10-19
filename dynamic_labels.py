"""
dynamic_labels.py -- Preliminary evaluation of a service-label change (R2 Comment 3).

Usage (same style as ablation.py):
    python dynamic_labels.py [N] [D] [mode]
        N    : number of valid realizations (default 100)
        D    : number of devices (default 18)
        mode : 'random'   -> n random devices move to another class (default)
               'escalate' -> n mMTC devices become URLLC

Protocol (per realization, same seeds/drops as ablation.py):
  1. AO-RA is run with the registration-time labels z0 -> stored solution
     (x, a, p, M)  [Stale].
  2. n devices change class (z0 -> z1). They take the lambda_d, T_max, R_min,
     w_c, N_d, eps_d of the NEW class. P_max_d is a hardware limit and is kept.
  3. Stale      : (x, a, p, M) from step 1 is kept; rates, delays and
                  constraints are re-evaluated under z1.
     Re-optimized: AO-RA is re-run from scratch with z1 (reference).
  Both are evaluated with the SAME function (normalized unit-packet delay model
  of Sec. IV, as in Sections VI-B..VI-F).
Metrics: weighted sum rate U, % devices violating C4 and C5 (and C5 on
unchanged devices only), % realizations violating C6 (stale M_c unstable)
and C8 (O-RU fronthaul over capacity).
"""
import sys
import numpy as np
import ao_ra as m
from scipy.stats import wilcoxon

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
D = int(sys.argv[2]) if len(sys.argv) > 2 else 18
MODE = sys.argv[3] if len(sys.argv) > 3 else 'random'
N_CHANGES = [2, 4, 6]
MAX_TRIES = 20


def aora_full(D, H, z, lam, Tmax, Pmax):
    """Algorithm 2, identical to m.AO_RA, but also returns (x, p, a, M)
    of the best iterate (m.AO_RA only returns R_d, M_c, Lambda_c)."""
    R_min = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
    eta = np.array([max(R_min[z[d]], lam[d] + 1.0 / m.T_MAX[z[d]])
                    for d in range(D)])
    x = np.zeros((m.B, D), dtype=int)
    for d in range(D):
        x[int(np.argmax([np.mean(H[b, d, :]) for b in range(m.B)])), d] = 1
    h_dual, m_dual = np.zeros(D), np.zeros(m.B)
    hist, best = [], None
    for _ in range(m.I_MAX):
        p, a, R, h_dual, m_dual = m.lagrangian_kkt(
            D, m.B, m.K, H, x, z, lam, Tmax, Pmax, eta, h_dual, m_dual)
        M, alpha = m.compute_vnf(z, R, lam, D)
        U = sum(m.W[z[d]] * R[d] for d in range(D))
        hist.append(U)
        if best is None or U > best['U']:
            best = dict(U=U, x=x.copy(), p=p.copy(), a=a.copy(),
                        R=R.copy(), M=dict(M))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) / max(abs(hist[-1]), 1.0) < m.EPS_CONV:
            break
        x = m.greedy_association(D, H, z, Pmax)
    best['iters'] = len(hist)
    return best


def precheck(D, H, z, lam, Pmax, devices, orus):
    """Same necessary-condition pre-check as the main Monte Carlo."""
    for d in range(D):
        b = int(np.argmin([np.linalg.norm(devices[d] - orus[k]) for k in range(m.B)]))
        if m._rate(Pmax[d] * H[b, d, d % m.K] / m.N0, z[d], d) <= lam[d]:
            return False
    return True


def draw_change(z0, n, rng):
    z1 = dict(z0)
    if MODE == 'escalate':
        cand = [d for d in range(D) if z0[d] == 'mMTC']
        if n > len(cand):
            return None, None
        idx = rng.choice(cand, n, replace=False)
        for d in idx:
            z1[d] = 'URLLC'
    else:
        idx = rng.choice(D, n, replace=False)
        for d in idx:
            others = [c for c in m.SERVICE_CLASSES if c != z0[d]]
            z1[d] = others[rng.randint(2)]
    return z1, set(int(i) for i in idx)


def evaluate(x, p, a, M, H, z1, lam1, Tmax1, Pmax, Rmin1, changed):
    R = m.compute_rates(D, m.B, m.K, H, x, z1, p, a, Pmax)
    alpha1 = {sc: sum(lam1[d] for d in range(D) if z1[d] == sc)
              for sc in m.SERVICE_CLASSES}
    U = sum(m.W[z1[d]] * R[d] for d in range(D)) / 1e6
    c4 = np.array([R[d] < Rmin1[z1[d]] for d in range(D)])
    delay = [m.compute_delay(d, z1[d], R, lam1, M, alpha1) for d in range(D)]
    c5 = np.array([(not np.isfinite(t)) or t > Tmax1[d] for d, t in enumerate(delay)])
    unch = np.array([d not in changed for d in range(D)])
    c6 = any(alpha1[sc] > 0 and alpha1[sc] >= M[sc] * m.MU_C_DEFAULT
             for sc in m.SERVICE_CLASSES)
    load = [sum(R[d] for d in range(D) if x[b, d] == 1) for b in range(m.B)]
    c8 = any(l > m.C_MAX_B * (1 + 1e-9) for l in load)
    return dict(U=U, c4=c4.mean() * 100, c5=c5.mean() * 100,
                c5_unch=(c5[unch].mean() * 100 if unch.any() else 0.0),
                c6=float(c6) * 100, c8=float(c8) * 100,
                unserved=float(np.any(R == 0)) * 100)


KEYS = ['U', 'c4', 'c5', 'c5_unch', 'c6', 'c8', 'unserved']
res = {n: {'stale': {k: [] for k in KEYS}, 'reopt': {k: [] for k in KEYS},
           'iters': []} for n in N_CHANGES}
valid, seed, checked = 0, 0, False
while valid < N:
    seed += 1
    np.random.seed(seed * 1000 + D)          # same drops as ablation.py
    devices, orus = m.generate_positions(D)
    H = m.generate_channels(devices, orus, D)
    z0 = m.assign_service_classes(D)
    lam0, Tmax0, Pmax = m.get_device_params(z0, D)
    if not precheck(D, H, z0, lam0, Pmax, devices, orus):
        continue
    sol0 = aora_full(D, H, z0, lam0, Tmax0, Pmax)
    if np.any(sol0['R'] == 0):
        continue
    if not checked:                           # sanity: same result as m.AO_RA
        _, R_chk, _, _ = m.AO_RA(D, H, z0, lam0, Tmax0, Pmax)
        U_chk = sum(m.W[z0[d]] * R_chk[d] for d in range(D))
        print(f"[check] |U_aora_full - U_AO_RA| = {abs(sol0['U'] - U_chk):.3e}")
        checked = True

    rng = np.random.RandomState(seed)
    changes, ok = {}, True
    for n in N_CHANGES:
        for _ in range(MAX_TRIES):
            z1, idx = draw_change(z0, n, rng)
            if z1 is None:
                break
            lam1, Tmax1, _ = m.get_device_params(z1, D)
            if precheck(D, H, z1, lam1, Pmax, devices, orus):
                changes[n] = (z1, idx, lam1, Tmax1)
                break
        if n not in changes:
            ok = False
            break
    if not ok:
        continue

    for n in N_CHANGES:
        z1, idx, lam1, Tmax1 = changes[n]
        Rmin1 = m.compute_dynamic_rmin(D, H, z1, Pmax, silent=True)
        st = evaluate(sol0['x'], sol0['p'], sol0['a'], sol0['M'],
                      H, z1, lam1, Tmax1, Pmax, Rmin1, idx)
        sol1 = aora_full(D, H, z1, lam1, Tmax1, Pmax)
        re = evaluate(sol1['x'], sol1['p'], sol1['a'], sol1['M'],
                      H, z1, lam1, Tmax1, Pmax, Rmin1, idx)
        for k in KEYS:
            res[n]['stale'][k].append(st[k])
            res[n]['reopt'][k].append(re[k])
        res[n]['iters'].append(sol1['iters'])
    valid += 1
    if valid % 10 == 0:
        print(f"  {valid}/{N} realizations (attempts: {seed})")

print(f"\nD={D}, mode={MODE}, N={N}  (discarded {seed - N} of {seed} drawn)")
print(f"{'n':>2} | {'U_stale':>8} {'U_reopt':>8} {'loss%':>6} {'p(W)':>8} | "
      f"{'C4 st/re':>11} {'C5 st/re':>11} {'C5unch st':>9} | "
      f"{'C6 st/re':>9} {'C8 st/re':>9} | iters_reopt")
rows = []
for n in N_CHANGES:
    S, R = res[n]['stale'], res[n]['reopt']
    us, ur = np.array(S['U']), np.array(R['U'])
    diff = ur - us; nz = diff[diff != 0]
    pv = wilcoxon(nz, alternative='greater').pvalue if len(nz) else float('nan')
    loss = (ur.mean() - us.mean()) / ur.mean() * 100
    f = lambda k, d: np.mean(d[k])
    print(f"{n:>2} | {us.mean():8.2f} {ur.mean():8.2f} {loss:6.1f} {pv:8.1e} | "
          f"{f('c4',S):5.1f}/{f('c4',R):5.1f} {f('c5',S):5.1f}/{f('c5',R):5.1f} "
          f"{f('c5_unch',S):9.1f} | {f('c6',S):4.0f}/{f('c6',R):4.0f} "
          f"{f('c8',S):4.0f}/{f('c8',R):4.0f} | {np.mean(res[n]['iters']):.2f}")
    rows.append(f"{n} & {us.mean():.1f} & {ur.mean():.1f} & {loss:.1f} & "
                f"{f('c4',S):.1f} / {f('c5',S):.1f} & {f('c6',S):.0f} \\\\")
print("\n=== LaTeX rows (stale violations) ===")
print("\n".join(rows))
np.savez(f'dynamic_labels_D{D}_{MODE}.npz',
         **{f'{k}_{s}_n{n}': np.array(res[n][s][k])
            for n in N_CHANGES for s in ['stale', 'reopt'] for k in KEYS})