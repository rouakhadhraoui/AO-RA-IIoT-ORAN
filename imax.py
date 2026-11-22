import numpy as np, ao_ra as m
r = np.load('results/mc_results.npy', allow_pickle=True).item()
for D in (9, 18, 27, 36):
    rows, valid, att = [], 0, 0
    while valid < 100:
        att += 1
        np.random.seed(att * 1000 + D)
        dev, oru = m.generate_positions(D); H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D); lam, Tmax, Pmax = m.get_device_params(z, D)
        if any(m._rate(Pmax[d] * H[int(np.argmin([np.linalg.norm(dev[d] - oru[b]) for b in range(m.B)])), d, d % m.K] / m.N0, z[d], d) <= lam[d] for d in range(D)):
            continue
        hist, R, M, alp = m.AO_RA(D, H, z, lam, Tmax, Pmax)
        if R is None or np.any(R == 0): continue
        rows.append(hist); valid += 1
    f = lambda v: np.mean(v) / 1e6
    print(D, "Imax=1: %.2f | Imax=2: %.2f | Imax>=3: %.2f" % (
        f([h[0] for h in rows]), f([max(h[:2]) for h in rows]), f([max(h) for h in rows])),
        "| mean iter %.2f | continue after it.2: %.0f%%" % (
        np.mean([len(h) for h in rows]), 100 * np.mean([len(h) > 2 for h in rows])),
        "| gain it1->2: %.2f | best==last: %.0f%% | GAA lower than initial: %.0f%%" % (
        f([h[1] - h[0] for h in rows]), 100 * np.mean([abs(max(h) - h[-1]) < 1e-6 for h in rows]),
        100 * np.mean([h[1] < h[0] - 1e-6 for h in rows])),
        "| std AO-RA %.2f DR %.2f" % (r['aora'][D]['utility_std'] / 1e6, r['dr'][D]['utility_std'] / 1e6))