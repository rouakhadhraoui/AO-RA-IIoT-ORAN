import numpy as np, ao_ra as m

def waterfill(P, g, N0):
    lo, hi = 0.0, P + N0 / g.min()
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        lo, hi = (mu, hi) if np.maximum(0, mu - N0 / g).sum() < P else (lo, mu)
    p = np.maximum(0, 0.5 * (lo + hi) - N0 / g)
    return p * (P / p.sum())

for D in (9, 18, 27, 36):
    gaps, mh, mm_ = [], 0.0, 0.0
    for s in range(1, 101):
        np.random.seed(s * 1000 + D)
        dev, oru = m.generate_positions(D); H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D); lam, Tmax, Pmax = m.get_device_params(z, D)
        Rm = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
        eta = np.array([max(Rm[z[d]], lam[d] + 1 / m.T_MAX[z[d]]) for d in range(D)])
        x = np.zeros((m.B, D), int)
        for d in range(D):
            x[int(np.argmax([H[b, d].mean() for b in range(m.B)])), d] = 1
        h, mm = np.zeros(D), np.zeros(m.B)
        for it in range(2):
            p, a, R, h, mm = m.lagrangian_kkt(D, m.B, m.K, H, x, z, lam, Tmax, Pmax, eta, h, mm)
            mh, mm_ = max(mh, h.max()), max(mm_, mm.max())
            for d in range(D):
                b = int(np.argmax(x[:, d])); ks = np.where(a[b, d] > 0.5)[0]
                if len(ks) < 2: continue
                g = H[b, d, ks]
                rate = lambda pp: sum(m._rate(pk * gk / m.N0, z[d], d) for pk, gk in zip(pp, g))
                r_eq = rate(np.full(len(ks), Pmax[d] / len(ks)))
                gaps.append((rate(waterfill(Pmax[d], g, m.N0)) - r_eq) / r_eq)
            x = m.greedy_association(D, H, z, Pmax)
    print(D, "max WF gain over equal split: %.2e" % max(gaps), "| max h: %.2f max m: %.3f" % (mh, mm_))