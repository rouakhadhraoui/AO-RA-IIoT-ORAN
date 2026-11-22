import numpy as np, ao_ra as m
for D in (9, 18, 27, 36):
    lo, hi = np.inf, -np.inf
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
            for d in range(D):
                b = int(np.argmax(x[:, d]))
                for k in np.where(a[b, d] > 0.5)[0]:
                    if p[d, k] > 0:
                        v = 10 * np.log10(p[d, k] * H[b, d, k] / m.N0)
                        lo, hi = min(lo, v), max(hi, v)
            x = m.greedy_association(D, H, z, Pmax)
    print(D, "SNR range dB: %.1f to %.1f" % (lo, hi))