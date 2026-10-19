import os
import numpy as np
from scipy import stats
from ao_ra import run_monte_carlo

D_LIST = [9, 18, 27, 36]
N_RUNS = 100            
FILE = f'results/mc_runs_{N_RUNS}.npy'

if os.path.exists(FILE):
    results = np.load(FILE, allow_pickle=True).item()
else:
    results = run_monte_carlo(D_LIST, n_runs=N_RUNS)
    np.save(FILE, results, allow_pickle=True)

def paired_report(u_a, u_b, n_boot=10000, seed=0):
    a, b = np.asarray(u_a)/1e6, np.asarray(u_b)/1e6     # Mbps
    d = a - b; n = len(d)
    se = d.std(ddof=1)/np.sqrt(n)
    ci = stats.t.interval(0.95, n-1, d.mean(), se)
    p_t = stats.ttest_rel(a, b, alternative='greater').pvalue
    p_w = stats.wilcoxon(d, alternative='greater').pvalue
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    gain = 100*(a[idx].mean(1)-b[idx].mean(1))/b[idx].mean(1)
    return dict(n=n, mean_AORA=a.mean(), mean_DR=b.mean(),
                gain_pct=100*d.mean()/b.mean(),
                gain_CI95=(round(np.percentile(gain,2.5),2), round(np.percentile(gain,97.5),2)),
                diff_CI95_Mbps=(round(ci[0],2), round(ci[1],2)),
                p_ttest=p_t, p_wilcoxon=p_w,
                pct_runs_AORA_wins=100*np.mean(d>0))

for D in D_LIST:
    u_a = results['aora'][D]['utility_runs']
    u_b = results['dr'][D]['utility_runs']
    r = paired_report(u_a, u_b)
    print(f"\nD = {D}")
    for k, v in r.items():
        print(f"  {k}: {v}")

    # Wins / ties / losses (differences in Mbps, tolerance 1e-6)
    d = (np.asarray(u_a) - np.asarray(u_b)) / 1e6
    wins   = int(np.sum(d >  1e-6))
    ties   = int(np.sum(np.abs(d) <= 1e-6))
    losses = int(np.sum(d < -1e-6))
    print(f"  wins/ties/losses: {wins}/{ties}/{losses}")
    if losses > 0:
        print(f"  mean loss size (Mbps): {d[d < -1e-6].mean():.3f}")
        print(f"  worst loss (Mbps): {d.min():.3f}")