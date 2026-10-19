# rerun_meta.py
import numpy as np
import ao_ra
from ao_ra import run_monte_carlo

D_LIST = [9, 18, 27, 36]
results = run_monte_carlo(D_LIST, n_runs=100)   # same seed_base=0 default

print("\nOverall discard rate across all D:")
tot_attempt = sum(results['meta'][D]['n_attempts'] for D in D_LIST)
tot_valid   = sum(results['meta'][D]['n_valid']    for D in D_LIST)
print(f"{tot_attempt - tot_valid}/{tot_attempt} "
      f"({(tot_attempt - tot_valid)/tot_attempt*100:.1f}%)")

for D in D_LIST:
    m = results['meta'][D]
    print(f"D={D:2d}: {m['n_discarded']}/{m['n_attempts']} discarded "
          f"({m['discard_pct']:.1f}%)")

np.save('results/mc_results_with_meta.npy', results, allow_pickle=True)