import numpy as np, ao_ra as m
for D in (9, 18, 27, 36):
    n_bad = n_over = n_c5 = n_dev = 0
    valid = att = 0
    while valid < 100:
        att += 1
        np.random.seed(att * 1000 + D)
        dev, oru = m.generate_positions(D); H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D); lam, Tmax, Pmax = m.get_device_params(z, D)
        if any(m._rate(Pmax[d] * H[int(np.argmin([np.linalg.norm(dev[d] - oru[b]) for b in range(m.B)])), d, d % m.K] / m.N0, z[d], d) <= lam[d] for d in range(D)):
            continue
        _, R, M, alp = m.AO_RA(D, H, z, lam, Tmax, Pmax)
        if R is None or np.any(R == 0): continue
        valid += 1
        for d in range(D):
            sc, T, mu = z[d], m.T_MAX[z[d]], m.MU_C_DEFAULT
            den = (T * mu - 3) * (R[d] - lam[d]) - mu
            if R[d] <= lam[d] + 1 / T or den <= 0: n_bad += 1
            elif np.ceil(alp[sc] * (T * (R[d] - lam[d]) - 1) / den) > m.M_MAX: n_over += 1
            n_dev += 1
            if not m.compute_delay(d, sc, R, lam, M, alp) <= T: n_c5 += 1
    print(D, "devices:", n_dev, "| Lemma1 inapplicable:", n_bad, "| M*>Mmax:", n_over, "| C5 violated:", n_c5)