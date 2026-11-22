import numpy as np
r = np.load('results/mc_results.npy', allow_pickle=True).item()
for D in (9, 18, 27, 36):
    a = np.asarray(r['aora'][D]['utility_runs']) / 1e6
    b = np.asarray(r['dr'][D]['utility_runs']) / 1e6
    nt = np.abs(a - b) > 1e-6
    print(D, round(100 * (a[nt].mean() - b[nt].mean()) / b[nt].mean(), 1))