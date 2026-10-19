"""Are the AO-RA dual variables ever non-zero? (supports the claim in Remark 1 / Sec. VI-I)"""
import sys
import numpy as np
import ao_ra as m

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

def run(D, p_dbm=None):
    if p_dbm is not None:
        pw = 10 ** (p_dbm / 10) * 1e-3
        m.P_MAX['eMBB'] = pw; m.P_MAX['URLLC'] = pw
    hmax = mmax = 0.0; n = 0; seed = 0
    while n < N:
        seed += 1
        np.random.seed(seed * 1000 + D)
        dev, orus = m.generate_positions(D); H = m.generate_channels(dev, orus, D)
        z = m.assign_service_classes(D); lam, Tmax, Pmax = m.get_device_params(z, D)
        ok = True
        for d in range(D):
            b = int(np.argmin([np.linalg.norm(dev[d] - orus[k]) for k in range(m.B)]))
            if m._rate(Pmax[d] * H[b, d, d % m.K] / m.N0, z[d], d) <= lam[d]:
                ok = False; break
        if not ok: continue
        n += 1
        rm = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
        eta = np.array([max(rm[z[d]], lam[d] + 1 / m.T_MAX[z[d]]) for d in range(D)])
        x = np.zeros((m.B, D), dtype=int)
        for d in range(D):
            x[int(np.argmax([np.mean(H[b, d, :]) for b in range(m.B)])), d] = 1
        h = np.zeros(D); mm = np.zeros(m.B)
        for it in range(3):                      # AO-RA usually stops in <= 3 outer iterations
            p, a, R, h, mm = m.lagrangian_kkt(D, m.B, m.K, H, x, z, lam, Tmax, Pmax, eta, h, mm)
            hmax = max(hmax, h.max()); mmax = max(mmax, mm.max())
            x = m.greedy_association(D, H, z, Pmax)
    return hmax, mmax

orig = dict(m.P_MAX)
for D in (9, 18, 27, 36):
    print(f"D={D:2d}, Pmax=33 dBm : max h_d = {run(D)[0]:.4f}, max m_b = {run(D)[1]:.4f}")
for pdbm in (20, 26, 35):
    h_, m_ = run(18, pdbm)
    print(f"D=18, Pmax={pdbm} dBm : max h_d = {h_:.4f}, max m_b = {m_:.4f}")
m.P_MAX.update(orig)