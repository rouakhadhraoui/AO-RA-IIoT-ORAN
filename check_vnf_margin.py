import numpy as np
import ao_ra as m

for D in (9, 18, 27, 36):
    margins, n_slices = [], 0
    for s in range(100):
        np.random.seed(s * 1000 + D)
        dev, oru = m.generate_positions(D)
        H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D)
        lam, Tmax, Pmax = m.get_device_params(z, D)
        hist, R, M, alp = m.AO_RA(D, H, z, lam, Tmax, Pmax)
        if R is None:
            continue
        for sc in m.SERVICE_CLASSES:
            slack = m.MU_C_DEFAULT - alp[sc] / M[sc]       # mu_c - Lambda_c/M_c
            margins.append(slack / m.MU_C_DEFAULT)
            n_slices += 1
    print(f"D={D}: slices={n_slices}  min relative margin={min(margins):.3f}  "
          f"(all > 0: {all(v > 0 for v in margins)})")