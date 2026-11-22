import numpy as np, ao_ra as m
orig = m.lagrangian_kkt
rec = {'h': 0.0, 'm': 0.0}
def wrap(*a, **k):
    out = orig(*a, **k)
    rec['h'] = max(rec['h'], float(out[3].max()))
    rec['m'] = max(rec['m'], float(out[4].max()))
    return out
m.lagrangian_kkt = wrap

for D in (9, 18, 27, 36):
    valid = att = 0
    pre_all = st_all = pre100 = st100 = 0
    hmax = mmax = 0.0
    while valid < 100:
        att += 1
        np.random.seed(att * 1000 + D)
        dev, oru = m.generate_positions(D); H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D); lam, Tmax, Pmax = m.get_device_params(z, D)
        pre_fail = any(m._rate(Pmax[d] * H[int(np.argmin([np.linalg.norm(dev[d] - oru[b]) for b in range(m.B)])), d, d % m.K] / m.N0, z[d], d) <= lam[d] for d in range(D))
        if pre_fail:
            pre_all += 1; pre100 += (att <= 100); continue
        rec['h'] = rec['m'] = 0.0
        hist, R, M, alp = m.AO_RA(D, H, z, lam, Tmax, Pmax)
        if R is None or np.any(R == 0):
            st_all += 1; st100 += (att <= 100); continue
        valid += 1
        hmax = max(hmax, rec['h']); mmax = max(mmax, rec['m'])
    print(f"D={D}: attempts={att} | precheck-fail={pre_all} unserved={st_all} | "
          f"within first 100 attempts: precheck-fail={pre100} unserved={st100} | "
          f"max h={hmax:.2f} max m={mmax:.3f}")