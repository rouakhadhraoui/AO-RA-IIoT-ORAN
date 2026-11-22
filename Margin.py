import numpy as np
mu = 5e6; worst = 1e99
for D in (9, 18, 27):
    for lam in np.linspace(0.1e6, 1.62e6, 10):
        L = (D // 3) * lam
        worst = min(worst, mu - L / (np.floor(L / mu) + 1))
print("min margin over the sweep (Mbps):", worst / 1e6)
