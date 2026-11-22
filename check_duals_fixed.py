import numpy as np, ao_ra as m
m.RMIN_MODE = 'fixed'
m.R_MIN_FIXED = {'eMBB': 5e6, 'URLLC': 5e6, 'mMTC': 5e6}
for D in (9, 18, 27, 36):
    mh, mm_ = 0.0, 0.0
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
    print(f"D={D}: max h_d = {mh:.2f}, max m_b = {mm_:.3f}")