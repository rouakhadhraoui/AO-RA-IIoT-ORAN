"""
Per-device constraint verification tables (reviewer-facing evidence).

Runs AO-RA on a single deterministic realization (seed = 42) for a given
device count D and prints/exports a table checking, per device:
    C1 (transmit power cap), C4 (minimum rate), C5 (end-to-end delay)
and, per O-RU:
    C8 (fronthaul capacity)
plus starvation (devices with 0 PRBs) and the per-class PRB distribution.

These are the verification tables produced for D = 9 and D = 27
(zero violations, zero starvation). Not part of the paper figure pipeline.

Usage
-----
    python verify_constraints.py            # D = 9 and D = 27 (default)
    python verify_constraints.py -D 18 36   # custom device counts
"""

import argparse
import numpy as np
import pandas as pd

from ao_ra import (
    B, K, P_MAX, T_MAX, C_MAX_B, MAX_PRB_PER_DEVICE,
    MU_C_DEFAULT, SERVICE_CLASSES,
    generate_positions, generate_channels, assign_service_classes,
    get_device_params, compute_dynamic_rmin,
    lagrangian_kkt, compute_vnf, greedy_association, sinr_lower_bound,
)


def debug_table(D, H, z, x, a, p, R_d, M_c, alpha_c,
                lambda_d, R_min_dyn, mu_c=MU_C_DEFAULT):
    P_max_arr = np.array([P_MAX[z[i]] for i in range(D)])
    rows = []

    for d in range(D):
        b_d  = int(np.argmax(x[:, d]))
        prbs = [k for k in range(K) if a[b_d, d, k] > 0.5]
        sc   = z[d]

        p_max_prb = float(np.max(p[d, :])) if np.any(p[d, :] > 0) else 0.0
        sinr = (sinr_lower_bound(d, b_d, prbs[0], p, H, x, D, P_max_arr)
                if prbs else 0.0)

        if R_d[d] > lambda_d[d] + 1e-6 and M_c[sc] > 0:
            T_tx     = 1.0 / (R_d[d] - lambda_d[d])
            denom_q  = mu_c - alpha_c[sc] / M_c[sc]
            T_proc   = 3.0 / denom_q if denom_q > 0 else np.inf
            delay_ms = (T_tx + T_proc) * 1e3
        else:
            delay_ms = np.inf

        rows.append({
            'device'        : d,
            'class'         : sc,
            'O-RU'          : b_d,
            'n_PRBs'        : len(prbs),
            'PRBs'          : str(prbs),
            'power_W'       : round(p_max_prb, 4),
            'P_max_W'       : round(P_MAX[sc], 4),
            'meets_C1'      : bool(p_max_prb <= P_MAX[sc] * 1.001),
            'SINR_dB'       : round(10 * np.log10(sinr), 2) if sinr > 0 else None,
            'rate_Mbps'     : round(R_d[d] / 1e6, 3),
            'R_min_dyn_Mbps': round(R_min_dyn[sc] / 1e6, 3),
            'meets_C4'      : bool(R_d[d] >= R_min_dyn[sc]),
            'delay_ms'      : round(delay_ms, 4) if np.isfinite(delay_ms) else 'INF',
            'T_max_ms'      : T_MAX[sc] * 1e3,
            'meets_C5'      : (bool(delay_ms <= T_MAX[sc] * 1e3)
                               if np.isfinite(delay_ms) else False),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print(f"\nVNF allocation M_c : {M_c}")
    print(f"Devices with 0 PRBs: {(df['n_PRBs'] == 0).sum()}")
    print(f"C1 violations      : {(~df['meets_C1']).sum()}")
    print(f"C4 violations      : {(~df['meets_C4']).sum()}")
    print(f"C5 violations      : {(df['meets_C5'] == False).sum()}")

    print("\nC8 check:")
    c8_viol = 0
    for b in range(B):
        load = sum(R_d[d] for d in range(D) if x[b, d] == 1)
        flag = "OK" if load <= C_MAX_B * 1.001 else "VIOLATION"
        if load > C_MAX_B * 1.001:
            c8_viol += 1
        print(f"  O-RU {b}: {load/1e6:6.3f} Mbps / {C_MAX_B/1e6:.1f} Mbps  {flag}")
    if c8_viol == 0:
        print("  All O-RUs satisfy C8")

    print("\nPRB distribution per service class:")
    for sc in SERVICE_CLASSES:
        sc_df = df[df['class'] == sc]
        if len(sc_df) == 0:
            continue
        print(f"  {sc:5s}: min={sc_df['n_PRBs'].min()}  "
              f"max={sc_df['n_PRBs'].max()}  "
              f"mean={sc_df['n_PRBs'].mean():.2f}  "
              f"(cap={MAX_PRB_PER_DEVICE})")

    return df


def run_and_debug(D_dbg, seed=42, n_iter=10):
    np.random.seed(seed)

    dev, oru        = generate_positions(D_dbg)
    H_d             = generate_channels(dev, oru, D_dbg)
    z_d             = assign_service_classes(D_dbg)
    lam, Tmax, Pmax = get_device_params(z_d, D_dbg)

    R_min_dyn = compute_dynamic_rmin(D_dbg, H_d, z_d, Pmax)

    eta = np.array([
        max(R_min_dyn[z_d[d]],
            lam[d] + 1.0 / T_MAX[z_d[d]])
        for d in range(D_dbg)
    ])

    x_d = np.zeros((B, D_dbg), dtype=int)
    for d in range(D_dbg):
        best_b     = int(np.argmax([np.mean(H_d[b, d, :]) for b in range(B)]))
        x_d[best_b, d] = 1

    h0 = np.zeros(D_dbg)
    m0 = np.zeros(B)

    for _ in range(n_iter):
        p_d, a_d, R_d, h0, m0 = lagrangian_kkt(
            D_dbg, B, K, H_d, x_d, z_d, lam, Tmax, Pmax, eta, h0, m0)
        M_d, alp_d = compute_vnf(z_d, R_d, lam, D_dbg)
        x_d = greedy_association(D_dbg, H_d, z_d, Pmax)

    p_d, a_d, R_true, h0, m0 = lagrangian_kkt(
        D_dbg, B, K, H_d, x_d, z_d, lam, Tmax, Pmax, eta, h0, m0)
    M_d, alp_d = compute_vnf(z_d, R_true, lam, D_dbg)

    df = debug_table(D_dbg, H_d, z_d, x_d, a_d, p_d,
                     R_true, M_d, alp_d, lam, R_min_dyn)

    fname = f'AO_RA_debug_D{D_dbg}.xlsx'
    df.to_excel(fname, index=False)
    print(f"\nSaved -> {fname}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-D', '--devices', type=int, nargs='+',
                        default=[9, 27],
                        help='Device counts to verify (default: 9 27)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    for D_dbg in args.devices:
        print("\n" + "#" * 70)
        print(f"#  Constraint verification: D = {D_dbg}, seed = {args.seed}")
        print("#" * 70)
        run_and_debug(D_dbg, seed=args.seed)
