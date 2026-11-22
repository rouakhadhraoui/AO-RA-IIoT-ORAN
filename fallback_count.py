import numpy as np, ao_ra as m

for D in (9, 18, 27, 36):
    fb_dev = fb_drops = valid = att = 0
    while valid < 100:
        att += 1
        np.random.seed(att * 1000 + D)
        dev, oru = m.generate_positions(D)
        H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D)
        lam, Tmax, Pmax = m.get_device_params(z, D)
        if any(m._rate(Pmax[d] * H[int(np.argmin([np.linalg.norm(dev[d] - oru[b]) for b in range(m.B)])), d, d % m.K] / m.N0, z[d], d) <= lam[d] for d in range(D)):
            continue
        _, R, _, _ = m.AO_RA(D, H, z, lam, Tmax, Pmax)
        if R is None or np.any(R == 0):
            continue
        valid += 1
        s = {'fb': 0}
        m.greedy_association(D, H, z, Pmax, stats=s)
        fb_dev += s['fb']
        fb_drops += int(s['fb'] > 0)
    print(D, "| devices using fallback:", fb_dev, "| drops with fallback:", fb_drops, "/ 100")