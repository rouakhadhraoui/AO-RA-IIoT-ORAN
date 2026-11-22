"""
verify_mc.py -- re-run the EXACT Monte Carlo drops of run_monte_carlo()
(same seeds, same pre-check, same starved-device filter) and count violations
of C1, C4, C5, C6, C8 and Lemma-1 validity for AO-RA and DR.

Put it next to ao_ra.py and run:  python verify_mc.py
It does not modify ao_ra.py.
"""
import numpy as np
import ao_ra as m

TOL = 1e-9
mu = m.MU_C_DEFAULT
saved = np.load('results/mc_results.npy', allow_pickle=True).item()


def run_loop(D, H, z, lam, Tmax, Pmax, eta, refresh):
    """Same loop as AO_RA (refresh=True) / dr_scheme (refresh=False).
    Returns the best iterate WITH its allocation (x, p, a, R)."""
    x = np.zeros((m.B, D), dtype=int)
    for d in range(D):
        x[int(np.argmax([np.mean(H[b, d, :]) for b in range(m.B)])), d] = 1
    h, mm = np.zeros(D), np.zeros(m.B)
    hist, best = [], None
    for _ in range(m.I_MAX):
        p, a, R, h, mm = m.lagrangian_kkt(D, m.B, m.K, H, x, z, lam, Tmax,
                                          Pmax, eta, h, mm)
        U = sum(m.W[z[d]] * R[d] for d in range(D))
        hist.append(U)
        if best is None or U > best[0]:
            best = (U, x.copy(), p.copy(), a.copy(), R.copy())
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) / max(abs(hist[-1]), 1.0) < m.EPS_CONV:
            break
        if refresh:
            x = m.greedy_association(D, H, z, Pmax)
    return best


def check(alloc, D, z, lam, Pmax, Rmin, cnt):
    U, x, p, a, R = alloc
    M, alp = m.compute_vnf(z, R, lam, D)
    for d in range(D):
        sc, T = z[d], m.T_MAX[z[d]]
        cnt['dev'] += 1
        if p[d].sum() > Pmax[d] * (1 + TOL):                 cnt['C1'] += 1
        if R[d] < Rmin[sc] * (1 - TOL):                      cnt['C4'] += 1
        if not m.compute_delay(d, sc, R, lam, M, alp) <= T:  cnt['C5'] += 1
        # Lemma 1 validity / budget (the places where compute_vnf falls back to M_MAX)
        num = alp[sc] * (T * R[d] - T * lam[d] - 1)
        den = (T * mu - 3) * (R[d] - lam[d]) - mu
        if R[d] <= lam[d] + 1e-10 or num <= 0 or den <= 0:   cnt['lemma_inapplicable'] += 1
        elif np.ceil(num / den) > m.M_MAX:                   cnt['M_over_Mmax'] += 1
    for sc in m.SERVICE_CLASSES:
        cnt['slices'] += 1
        if not alp[sc] < M[sc] * mu:                         cnt['C6'] += 1   # strict
        cnt['min_margin'] = min(cnt['min_margin'], (mu - alp[sc] / M[sc]) / mu)
    for b in range(m.B):
        load = sum(R[d] for d in range(D) if x[b, d] == 1)
        cnt['C8_max'] = max(cnt['C8_max'], load / m.C_MAX_B)
        if load > m.C_MAX_B * (1 + TOL):                     cnt['C8'] += 1


def new_cnt():
    return dict(dev=0, slices=0, C1=0, C4=0, C5=0, C6=0, C8=0,
                lemma_inapplicable=0, M_over_Mmax=0, min_margin=1.0, C8_max=0.0, drops_C8=0, drops_C4=0)


for D in (9, 18, 27, 36):
    cA, cR = new_cnt(), new_cnt()
    valid, att, u_ao, u_dr, dr_bad = 0, 0, [], [], []
    while valid < 100:
        att += 1
        np.random.seed(0 + att * 1000 + D)                 # same as run_monte_carlo
        dev, oru = m.generate_positions(D)
        H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D)
        lam, Tmax, Pmax = m.get_device_params(z, D)
        if any(m._rate(Pmax[d] * H[int(np.argmin([np.linalg.norm(dev[d] - oru[b])
               for b in range(m.B)])), d, d % m.K] / m.N0, z[d], d) <= lam[d]
               for d in range(D)):
            continue
        Rmin = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
        eta = np.array([max(Rmin[z[d]], lam[d] + 1.0 / m.T_MAX[z[d]]) for d in range(D)])
        ao = run_loop(D, H, z, lam, Tmax, Pmax, eta, refresh=True)
        if np.any(ao[4] == 0):
            continue
        dr = run_loop(D, H, z, lam, Tmax, Pmax, eta, refresh=False)
        bA, bR = dict(cA), dict(cR)
        check(ao, D, z, lam, Pmax, Rmin, cA)
        check(dr, D, z, lam, Pmax, Rmin, cR)
        if cA['C8'] > bA['C8']: cA['drops_C8'] += 1
        if cR['C8'] > bR['C8']: cR['drops_C8'] += 1
        bad = cR['C4'] > bR['C4']                  # DR leaves a device below C4
        if bad: cR['drops_C4'] += 1
        dr_bad.append(bad)
        u_ao.append(ao[0]); u_dr.append(dr[0])
        valid += 1
    same = (np.allclose(u_ao, saved['aora'][D]['utility_runs'], rtol=0, atol=1e-6) and
            np.allclose(u_dr, saved['dr'][D]['utility_runs'], rtol=0, atol=1e-6))
    print(f"\nD={D}: attempts={att}  drops identical to saved MC: {same}")
    for name, c in (('AO-RA', cA), ('DR', cR)):
        print(f"  {name}: devices={c['dev']} slices={c['slices']} | "
              f"C1={c['C1']} C4={c['C4']} C5={c['C5']} C6={c['C6']} C8(O-RU of {100*m.B})={c['C8']} (max load = {100*c['C8_max']:.1f}% of 46 Mbps) | "
              f"Lemma1 inapplicable={c['lemma_inapplicable']} M*>Mmax={c['M_over_Mmax']} | "
              f"min margin={c['min_margin']:.3f}")
        print(f"      drops with >=1 O-RU over C8: {c['drops_C8']}/100 | "
              f"max O-RU load / C8max = {c['C8_max']:.3f}"
              + (f" | drops where DR breaks C4: {c['drops_C4']}/100" if name == 'DR' else ""))
    ua, ud, ok = np.array(u_ao) / 1e6, np.array(u_dr) / 1e6, ~np.array(dr_bad)
    g_all = 100 * (ua.mean() - ud.mean()) / ud.mean()
    g_ok = 100 * (ua[ok].mean() - ud[ok].mean()) / ud[ok].mean() if ok.any() else float('nan')
    print(f"  gain AO-RA vs DR: all 100 drops = {g_all:.2f}% | only the {ok.sum()} drops "
          f"where DR satisfies C4 = {g_ok:.2f}%")