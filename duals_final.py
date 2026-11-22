import numpy as np, ao_ra as m

def duals_of_run(D, H, z, lam, Tmax, Pmax):
    Rm = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
    eta = np.array([max(Rm[z[d]], lam[d] + 1.0 / m.T_MAX[z[d]]) for d in range(D)])
    x = np.zeros((m.B, D), int)
    for d in range(D):
        x[int(np.argmax([H[b, d].mean() for b in range(m.B)])), d] = 1
    h, mm = np.zeros(D), np.zeros(m.B)
    hist, mh, mb = [], 0.0, 0.0
    for _ in range(m.I_MAX):
        p, a, R, h, mm = m.lagrangian_kkt(D, m.B, m.K, H, x, z, lam, Tmax, Pmax, eta, h, mm)
        mh, mb = max(mh, h.max()), max(mb, mm.max())
        hist.append(sum(m.W[z[d]] * R[d] for d in range(D)))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) / max(abs(hist[-1]), 1.0) < m.EPS_CONV:
            break
        x = m.greedy_association(D, H, z, Pmax)
    return mh, mb

for D in (9, 18, 27, 36):
    valid = att = 0
    MH = MB = 0.0
    while valid < 100:
        att += 1
        np.random.seed(att * 1000 + D)
        dev, oru = m.generate_positions(D); H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D); lam, Tmax, Pmax = m.get_device_params(z, D)
        if any(m._rate(Pmax[d] * H[int(np.argmin([np.linalg.norm(dev[d] - oru[b]) for b in range(m.B)])), d, d % m.K] / m.N0, z[d], d) <= lam[d] for d in range(D)):
            continue
        _, R, _, _ = m.AO_RA(D, H, z, lam, Tmax, Pmax)
        if R is None or np.any(R == 0):
            continue
        mh, mb = duals_of_run(D, H, z, lam, Tmax, Pmax)
        MH, MB = max(MH, mh), max(MB, mb)
        valid += 1
    print("D=%d: max h_d = %.2f, max m_b = %.3f" % (D, MH, MB))