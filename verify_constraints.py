"""
Per-device constraint verification tables (reviewer-facing evidence).

Runs AO-RA on a single deterministic realization (seed = 42) for a given
device count D and prints/exports a table checking, per device:
    C1 (transmit power cap), C4 (minimum rate), C5 (end-to-end delay)
and, per O-RU:
    C8 (fronthaul capacity)
plus starvation (devices with 0 PRBs) and the per-class PRB distribution.

The SNR column uses the interference-free (noise-limited) SNR of Eq. (5).

The alternating loop below mirrors AO_RA (Algorithm 2) in ao_ra.py:
SP3 -> SP4 -> record iterate -> convergence test -> SP2. The BEST iterate
(highest weighted sum rate) is the one verified, exactly as AO_RA reports it.

Before running, the script checks that the imported ao_ra.py uses the
paper's configuration (Table III) and aborts otherwise, so that the
verification tables can never silently come from a different setup.

Usage
-----
    python verify_constraints.py            # D = 9 and D = 27 (default)
    python verify_constraints.py -D 18 36   # custom device counts
    python verify_constraints.py --allow-mismatch   # skip the config abort
"""

import argparse
import numpy as np
import pandas as pd

import ao_ra
from ao_ra import (
    B, K, P_MAX, T_MAX, W, C_MAX_B, MAX_PRB_PER_DEVICE, N0,
    MU_C_DEFAULT, SERVICE_CLASSES, I_MAX, EPS_CONV,
    generate_positions, generate_channels, assign_service_classes,
    get_device_params, compute_dynamic_rmin,
    lagrangian_kkt, compute_vnf, greedy_association,
)

# Paper configuration (Table III) that the verification must run under
PAPER_CFG = dict(B=6, K=25, C_MAX_B=46e6, MU_C=5e6, MAX_PRB=3,
                 LAMBDA_EMBB=3e6)


def check_config(allow_mismatch=False):
    """Abort if the loaded ao_ra constants differ from the paper's Table III."""
    print(f"ao_ra loaded from: {ao_ra.__file__}")
    got = dict(B=B, K=K, C_MAX_B=C_MAX_B, MU_C=MU_C_DEFAULT,
               MAX_PRB=MAX_PRB_PER_DEVICE,
               LAMBDA_EMBB=ao_ra.LAMBDA['eMBB'])
    bad = {k: (got[k], v) for k, v in PAPER_CFG.items()
           if abs(got[k] - v) > 1e-9 * max(1.0, abs(v))}
    print("Config check vs paper (got, expected):", bad if bad else "OK")
    if bad and not allow_mismatch:
        raise SystemExit(
            "ao_ra.py does not match the paper's Table III. Fix ao_ra.py "
            "(or delete __pycache__ / check for a second copy of ao_ra.py), "
            "or rerun with --allow-mismatch.")


def debug_table(D, H, z, x, a, p, R_d, M_c, alpha_c,
                lambda_d, R_min_dyn, mu_c=MU_C_DEFAULT):
    rows = []

    for d in range(D):
        b_d  = int(np.argmax(x[:, d]))
        prbs = [k for k in range(K) if a[b_d, d, k] > 0.5]
        sc   = z[d]

        p_max_prb = float(np.max(p[d, :])) if np.any(p[d, :] > 0) else 0.0
        # Interference-free SNR (Eq. 5) on the first assigned PRB
        snr = (p[d, prbs[0]] * H[b_d, d, prbs[0]] / N0) if prbs else 0.0

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
            'SNR_dB'        : round(10 * np.log10(snr), 2) if snr > 0 else None,
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
    print(f"C5 violations      : {(~df['meets_C5']).sum()}")

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
    else:
        print(f"  C8 violations: {c8_viol}")

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


def run_and_debug(D_dbg, seed=42):
    np.random.seed(seed)

    dev, oru        = generate_positions(D_dbg)
    H_d             = generate_channels(dev, oru, D_dbg)
    z_d             = assign_service_classes(D_dbg)
    lam, Tmax, Pmax = get_device_params(z_d, D_dbg)

    # Sanity print: per-class arrival rates and the C6 stability floor
    for sc in SERVICE_CLASSES:
        idx = [d for d in range(D_dbg) if z_d[d] == sc]
        if idx:
            Lam = float(sum(lam[d] for d in idx))
            print(f"  lambda[{sc}] = {lam[idx[0]]/1e6:.2f} Mbps, "
                  f"Lambda_c = {Lam/1e6:.2f} Mbps, "
                  f"ceil(Lambda_c/mu) = {int(np.ceil(Lam / MU_C_DEFAULT))}")

    R_min_dyn = compute_dynamic_rmin(D_dbg, H_d, z_d, Pmax)

    # Merged rate bound eta_d (C4 + transmission part of C5), as in AO_RA
    eta = np.array([
        max(R_min_dyn[z_d[d]], lam[d] + 1.0 / T_MAX[z_d[d]])
        for d in range(D_dbg)
    ])

    # Initial association: best mean channel gain (as in AO_RA)
    x_d = np.zeros((B, D_dbg), dtype=int)
    for d in range(D_dbg):
        best_b = int(np.argmax([np.mean(H_d[b, d, :]) for b in range(B)]))
        x_d[best_b, d] = 1

    h0 = np.zeros(D_dbg)
    m0 = np.zeros(B)

    history = []
    best = None   # (U, x, p, a, R, M_c, alpha_c) of the best iterate

    for _ in range(I_MAX):
        # SP3: PRB + power given x
        p_d, a_d, R_d, h0, m0 = lagrangian_kkt(
            D_dbg, B, K, H_d, x_d, z_d, lam, Tmax, Pmax, eta, h0, m0)
        # SP4: VNF sizing given rates
        M_d, alp_d = compute_vnf(z_d, R_d, lam, D_dbg)

        U = sum(W[z_d[d]] * R_d[d] for d in range(D_dbg))
        history.append(U)

        # Keep the best iterate together with the association it used
        if best is None or U > best[0]:
            best = (U, x_d.copy(), p_d.copy(), a_d.copy(), R_d.copy(),
                    dict(M_d), dict(alp_d))

        # Relative convergence test (same as AO_RA)
        if len(history) > 1:
            diff = abs(history[-1] - history[-2]) / max(abs(history[-1]), 1.0)
            if diff < EPS_CONV:
                break

        # SP2: refresh association for next iteration
        x_d = greedy_association(D_dbg, H_d, z_d, Pmax)

    U_best, x_b, p_b, a_b, R_b, M_b, alp_b = best
    print(f"\nAO-RA iterations: {len(history)} | "
          f"best weighted sum rate = {U_best/1e6:.3f} Mbps")

    df = debug_table(D_dbg, H_d, z_d, x_b, a_b, p_b,
                     R_b, M_b, alp_b, lam, R_min_dyn)

    fname = f'AO_RA_debug_D{D_dbg}.xlsx'
    try:
        df.to_excel(fname, index=False)
    except ImportError:
        # openpyxl missing -> fall back to CSV
        fname = f'AO_RA_debug_D{D_dbg}.csv'
        df.to_csv(fname, index=False)
    print(f"\nSaved -> {fname}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-D', '--devices', type=int, nargs='+',
                        default=[9, 27],
                        help='Device counts to verify (default: 9 27)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-mismatch', action='store_true',
                        help='Do not abort if ao_ra.py differs from the paper config')
    args = parser.parse_args()

    check_config(args.allow_mismatch)

    for D_dbg in args.devices:
        print("\n" + "#" * 70)
        print(f"#  Constraint verification: D = {D_dbg}, seed = {args.seed}")
        print("#" * 70)
        run_and_debug(D_dbg, seed=args.seed)