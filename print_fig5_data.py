"""Print the data behind Fig. 5 (D = 27, seed = 42): devices and fronthaul
load per O-RU for AO-RA, DR and Baseline. Run: python print_fig5_data.py"""
import numpy as np
import ao_ra as a

D = 27
res, devices, orus, z = a.analyze_schemes(D, 42)
print(f"ao_ra loaded from: {a.__file__}\n")
for key, name in [('aora', 'AO-RA'), ('dr', 'DR'), ('baseline', 'Baseline')]:
    x, R = res[key]['x'], res[key]['R']
    counts = [int(x[b].sum()) for b in range(a.B)]
    loads = [sum(R[d] for d in range(D) if x[b, d] == 1) / 1e6
             for b in range(a.B)]
    print(f"{name:9s} devices per O-RU: {counts}")
    print(f"{'':9s} load per O-RU (Mbps): {[round(v, 1) for v in loads]}"
          f"  min={min(loads):.1f} max={max(loads):.1f}")
    print(f"{'':9s} std of device counts: {np.std(counts):.2f}\n")