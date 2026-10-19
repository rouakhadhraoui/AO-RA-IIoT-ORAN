"""
Impact of neglected delay components on URLLC end-to-end latency
(response to Reviewer 1, Comment 5).

The optimization model of the paper (Eqs. 13-16) is NOT modified. For the
allocations (R_d, M_c) that AO-RA already produces, this script re-evaluates
the URLLC latency with:
  1. explicit packet size L (M/M/1 queues in packets/s: delay = L x per-bit delay),
     which splits the device-side delay into transmission time L/R_d and queueing wait;
  2. serialization of one packet on the fronthaul / midhaul / backhaul links;
  3. radio propagation (distance / c) and fiber propagation (5 us per km).

Usage:
    python delay_components.py                  # 100 realizations, D in {9, 18, 27}
    python delay_components.py --runs 10        # quick check
    python delay_components.py --lambda-sweep   # also sweep the URLLC arrival rate
"""
import argparse
import os
import numpy as np
import pandas as pd

import ao_ra as m

L_BITS      = m.PKT_SIZE['URLLC'] * 8     # URLLC packet size [bits]
C_RADIO     = 3e8                         # radio propagation speed [m/s]
US_PER_KM   = 5.0                         # fiber propagation delay [us/km]
N_LINKS     = 3                           # fronthaul, midhaul (F1), backhaul
FIBER_KM    = [0, 10, 20, 40, 80]         # one-way O-RU -> UPF fiber length [km]
T_MAX       = m.T_MAX['URLLC']            # 1 ms budget
D_LIST      = [9, 18, 27]


def valid_realization(D, attempt):
    """Same scenario generation and filtering as run_monte_carlo (seed_base=0)."""
    np.random.seed(attempt * 1000 + D)
    devices, orus = m.generate_positions(D)
    H = m.generate_channels(devices, orus, D)
    z = m.assign_service_classes(D)
    lam, Tmax, Pmax = m.get_device_params(z, D)
    for d in range(D):
        b = int(np.argmin([np.linalg.norm(devices[d] - orus[k]) for k in range(m.B)]))
        if m._rate(Pmax[d] * H[b, d, d % m.K] / m.N0, z[d], d) <= lam[d]:
            return None
    return devices, orus, H, z, lam, Tmax, Pmax


def resize_vnf_packet(R, lam, z, D, alpha_urllc):
    """
    Lemma 1 re-applied with packet-level queues: the per-bit delay budget becomes
    T_max / L (delay_pkt = L x delay_bit <= T_max). Returns M_urllc.
    """
    mu, Tp = m.MU_C_DEFAULT, T_MAX / L_BITS
    M_min  = max(int(np.ceil(alpha_urllc / mu)), 1)
    cand = []
    for d in [i for i in range(D) if z[i] == 'URLLC']:
        num = alpha_urllc * (Tp * R[d] - Tp * lam[d] - 1)
        den = max((Tp * mu - 3) * (R[d] - lam[d]) - mu, 1e-6)
        cand.append(m.M_MAX if num <= 0 else int(np.ceil(num / den)))
    return int(np.clip(max(cand), M_min, m.M_MAX))


def urllc_components(D, devices, orus, z, lam, R, M_c, alpha_c, resize=False):
    """Per-URLLC-device delay components in seconds. resize=True re-sizes M_urllc (packet-level Lemma 1)."""
    mu = m.MU_C_DEFAULT
    M_u = resize_vnf_packet(R, lam, z, D, alpha_c['URLLC']) if resize else M_c['URLLC']
    rows = []
    for d in [i for i in range(D) if z[i] == 'URLLC']:
        t_tx_bit   = 1.0 / (R[d] - lam[d])                         # model, per-bit units
        t_proc_bit = 3.0 / (mu - alpha_c['URLLC'] / M_u)
        t_tx_pkt   = L_BITS * t_tx_bit                             # packet-level sojourn
        t_trans    = L_BITS / R[d]                                 # transmission (service) time
        t_proc_pkt = L_BITS * t_proc_bit
        t_ser      = N_LINKS * L_BITS / m.C_MAX_B                  # conservative: all links at C_max
        d_max      = max(np.linalg.norm(devices[d] - orus[b]) for b in range(m.B))
        rows.append(dict(
            model_total   = t_tx_bit + t_proc_bit,                 # what the paper reports
            tx_trans      = t_trans,
            tx_queue      = t_tx_pkt - t_trans,
            proc          = t_proc_pkt,
            serialization = t_ser,
            M_urllc       = M_u,
            prop_radio    = d_max / C_RADIO,                       # upper bound
        ))
    return rows


def run(n_runs):
    results = []
    for D in D_LIST:
        comps, valid, attempt = [], 0, 0
        while valid < n_runs:
            attempt += 1
            sc = valid_realization(D, attempt)
            if sc is None:
                continue
            devices, orus, H, z, lam, Tmax, Pmax = sc
            _, R, M_c, alpha_c = m.AO_RA(D, H, z, lam, Tmax, Pmax)
            if R is None or np.any(R == 0):
                continue
            comps += urllc_components(D, devices, orus, z, lam, R, M_c, alpha_c)
            valid += 1
        df = pd.DataFrame(comps)
        base = df.tx_trans + df.tx_queue + df.proc + df.serialization + df.prop_radio
        row = {'D': D,
               'model_total_us': df.model_total.mean() * 1e6,
               'tx_trans_us':    df.tx_trans.mean() * 1e6,
               'tx_queue_us':    df.tx_queue.mean() * 1e6,
               'proc_us':        df.proc.mean() * 1e6,
               'serial_us':      df.serialization.mean() * 1e6,
               'prop_radio_us':  df.prop_radio.mean() * 1e6,
               'refined_noFiber_mean_us': base.mean() * 1e6,
               'refined_noFiber_max_us':  base.max() * 1e6}
        for km in FIBER_KM:
            tot = base + km * US_PER_KM * 1e-6
            row[f'meet_1ms_pct_{km}km'] = 100 * np.mean(tot <= T_MAX)
        row['max_fiber_km_worst_device'] = (T_MAX - base.max()) / (US_PER_KM * 1e-6)
        results.append(row)
    out = pd.DataFrame(results)
    os.makedirs(m.RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(m.RESULTS_DIR, 'delay_components.csv'), index=False)
    pd.set_option('display.width', 200)
    print(out.round(2).T)
    return out


def lambda_sweep(n_runs, lambdas_mbps=(0.1, 0.4, 0.8, 1.2, 1.62), fiber_km=20):
    """Refined URLLC latency vs arrival rate; checks the 1 ms budget at fiber_km."""
    rows = []
    orig = m.LAMBDA['URLLC']
    for D in D_LIST:
        for lm in lambdas_mbps:
            m.LAMBDA['URLLC'] = lm * 1e6
            tots, Ms = {False: [], True: []}, []
            for run_id in range(n_runs):
                np.random.seed(run_id * 100)            # same seeding as Fig. 3
                devices, orus = m.generate_positions(D)
                H = m.generate_channels(devices, orus, D)
                z = m.assign_service_classes(D)
                lam, Tmax, Pmax = m.get_device_params(z, D)
                _, R, M_c, alpha_c = m.AO_RA(D, H, z, lam, Tmax, Pmax)
                if R is None:
                    continue
                for rs in (False, True):
                    for c in urllc_components(D, devices, orus, z, lam, R, M_c, alpha_c, resize=rs):
                        tots[rs].append(c['tx_trans'] + c['tx_queue'] + c['proc'] + c['serialization']
                                        + c['prop_radio'] + fiber_km * US_PER_KM * 1e-6)
                        if rs:
                            Ms.append(c['M_urllc'])
            a0, a1 = np.array(tots[False]), np.array(tots[True])
            rows.append(dict(D=D, lambda_Mbps=lm,
                             same_M_mean_ms=a0.mean() * 1e3, same_M_meet_pct=100 * np.mean(a0 <= T_MAX),
                             resized_mean_ms=a1.mean() * 1e3, resized_max_ms=a1.max() * 1e3,
                             resized_meet_pct=100 * np.mean(a1 <= T_MAX), resized_M_mean=np.mean(Ms)))
    m.LAMBDA['URLLC'] = orig
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(m.RESULTS_DIR, f'delay_lambda_sweep_{fiber_km}km.csv'), index=False)
    print(out.round(3).to_string(index=False))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=m.MC_RUNS)
    ap.add_argument('--lambda-sweep', action='store_true')
    ap.add_argument('--sweep-runs', type=int, default=20)
    a = ap.parse_args()
    run(a.runs)
    if a.lambda_sweep:
        lambda_sweep(a.sweep_runs)