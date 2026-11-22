import numpy as np, ao_ra as m
D = 18
for p_dbm in (20, 26, 33, 35):
    pw = 10 ** (p_dbm / 10) * 1e-3
    m.P_MAX['eMBB'] = pw; m.P_MAX['URLLC'] = pw
    mh = mm_ = 0.0
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
            x = m.greedy_association(D, H, z, Pmax)
    print(p_dbm, "dBm: max h %.2f, max m %.3f" % (mh, mm_))