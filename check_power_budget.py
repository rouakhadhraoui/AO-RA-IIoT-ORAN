import numpy as np
import ao_ra as m

np.random.seed(42)
D = 18
devices, orus = m.generate_positions(D)
H = m.generate_channels(devices, orus, D)
z = m.assign_service_classes(D)
lam, Tmax, Pmax = m.get_device_params(z, D)
R_min = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
eta = np.array([max(R_min[z[d]], lam[d] + 1.0 / m.T_MAX[z[d]]) for d in range(D)])
x = np.zeros((m.B, D), dtype=int)
for d in range(D):
    x[int(np.argmax([np.mean(H[b, d, :]) for b in range(m.B)])), d] = 1
p, a, R, h, mm = m.lagrangian_kkt(D, m.B, m.K, H, x, z, lam, Tmax, Pmax,
                                  eta, np.zeros(D), np.zeros(m.B))
tot = p.sum(axis=1)
print("max total power / Pmax :", round(float((tot / Pmax).max()), 3))
print("devices over budget    :", int((tot > Pmax * (1 + 1e-9)).sum()), "of", D)
print("weighted sum rate (Mbps):", round(sum(m.W[z[d]] * R[d] for d in range(D)) / 1e6, 2))