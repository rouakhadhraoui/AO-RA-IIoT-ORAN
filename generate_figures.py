"""
Reproduce all simulation figures of the paper (Section VI).

Usage
-----
    python generate_figures.py               # use cached Monte Carlo results if present
    python generate_figures.py --force       # re-run the full Monte Carlo experiment
    python generate_figures.py --quick       # fast smoke test (fewer runs; NOT paper numbers)

Outputs (in ./results/):
    fig1_convergence.{pdf,png}       Weighted sum rate vs. outer iterations
    fig2_utility_vs_D.{pdf,png}      Weighted sum rate vs. number of devices
    fig3_delay_vs_lambda.{pdf,png}   URLLC end-to-end delay vs. arrival rate
    fig4_utility_vs_power.{pdf,png}  Weighted sum rate vs. max transmit power
    fig5_oru_load.{pdf,png}          O-RU load distribution (D = 27)
    mc_results.npy                   Cached Monte Carlo statistics

Full reproduction (--force) takes on the order of a few hours on a laptop
(the dominant cost is Fig. 3's delay sweep: 3 device counts x 10 arrival
rates x 100 realizations).
"""

import argparse
import os
import numpy as np

import ao_ra
from ao_ra import (
    RESULTS_DIR, MC_RUNS, W,
    run_monte_carlo, apply_ieee_style,
    plot_fig1_convergence, plot_fig2_utility_vs_D, plot_fig3_delay,
    plot_fig4_power, plot_fig5_oru_load,
    compute_delay_vs_lambda, compute_utility_vs_power,
)

D_LIST = [9, 18, 27, 36]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                        help='Re-run the full Monte Carlo experiment even if '
                             'cached results exist.')
    parser.add_argument('--quick', action='store_true',
                        help='Fast smoke test with fewer runs (results will '
                             'NOT match the paper).')
    args = parser.parse_args()

    n_runs        = 3 if args.quick else MC_RUNS
    n_runs_delay  = 2 if args.quick else 100
    n_runs_power  = 2 if args.quick else 20
    d_list_delay  = [9] if args.quick else [9, 18, 27]

    mc_path = os.path.join(RESULTS_DIR, 'mc_results.npy')

    if not args.force and not args.quick and os.path.exists(mc_path):
        results_mc = np.load(mc_path, allow_pickle=True).item()
        print(f"Loaded cached results from {mc_path} (use --force to re-run).")
    else:
        print(f"Running Monte Carlo experiment ({n_runs} runs per D)...")
        results_mc = run_monte_carlo(D_LIST, n_runs=n_runs)
        if not args.quick:
            np.save(mc_path, results_mc, allow_pickle=True)
            print(f"Saved -> {mc_path}")

    apply_ieee_style()

    # Fig. 1 -- Convergence of the outer AO loop
    plot_fig1_convergence(results_mc, D_LIST, n_show=15)

    # Fig. 2 -- Weighted sum data rate vs. number of devices
    plot_fig2_utility_vs_D(results_mc, D_LIST)

    # Fig. 3 -- URLLC end-to-end delay vs. arrival rate
    lam_delay, delays_per_D = compute_delay_vs_lambda(
        D_list_delay=d_list_delay, n_runs=n_runs_delay)
    plot_fig3_delay(lam_delay, delays_per_D, d_list_delay)

    # Fig. 4 -- Weighted sum data rate vs. maximum transmit power
    p_dbm_vals, u_ao_p, u_dr_p, u_ba_p = compute_utility_vs_power(
        D_fixed=18, n_runs=n_runs_power)
    plot_fig4_power(p_dbm_vals, u_ao_p, u_dr_p, u_ba_p)

    # Fig. 5 -- O-RU load distribution (D = 27, single realization)
    plot_fig5_oru_load(D_fixed=27, seed=42)

    # Summary table: mean weighted sum data rate per scheme
    print("\n" + "=" * 65)
    print(f"{'D':>4} | {'AO-RA':>10} | {'DR':>10} | {'Baseline':>10} | "
          f"{'Gain vs Base':>12}")
    print("-" * 65)
    for D in D_LIST:
        u_ao = results_mc['aora'][D]['utility_mean'] / 1e6
        u_dr = results_mc['dr'][D]['utility_mean'] / 1e6
        u_ba = results_mc['baseline'][D]['utility_mean'] / 1e6
        gain = (u_ao - u_ba) / u_ba * 100
        print(f"{D:>4} | {u_ao:>10.3f} | {u_dr:>10.3f} | {u_ba:>10.3f} | "
              f"{gain:>11.1f}%")
    print("=" * 65)


if __name__ == '__main__':
    main()
