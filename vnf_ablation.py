"""
VNF-sizing ablation (response to Reviewer 2, Comment 2).

Rates R_d come from AO-RA and do NOT depend on the VNF sizing (SP4 does not
feed back into SP3), so AO-RA is run once per realization and the four sizing
rules are applied to the same allocation. URLLC latency is evaluated at packet
level exactly as in delay_components.py (L = 256 bit, link serialization,
radio propagation, 20 km of fiber).

Rules:
  lemma1_pkt : Lemma 1 with budget T_max/L (packet-level sizing, recommended)
  lemma1_unit: Lemma 1 with budget T_max   (unit-size packets, as in the model)
  minstable  : M_c = ceil(Lambda_c / mu_c)
  max        : M_c = M_MAX = 25

Usage:
    python vnf_ablation.py                 # 20 realizations per point
    python vnf_ablation.py --runs 5        # quick check
"""
import argparse
import os
import numpy as np
import pandas as pd
import ao_ra as m

L_BITS    = m.PKT_SIZE['URLLC'] * 8
C_RADIO   = 3e8
US_PER_KM = 5.0
N_LINKS   = 3
FIBER_KM  = 20
T_MAX     = m.T_MAX['URLLC']
RULES     = ['lemma1_pkt', 'lemma1_unit', 'minstable', 'max']


def lemma1_urllc(R, lam, z, D, alpha_u, budget):
    """Lemma 1 for the URLLC slice with delay budget `budget` (T_max or T_max/L)."""
    mu = m.MU_C_DEFAULT
    M_min = max(int(np.ceil(alpha_u / mu)), 1)
    cand = []
    for d in [i for i in range(D) if z[i] == 'URLLC']:
        num = alpha_u * (budget * R[d] - budget * lam[d] - 1)
        den = max((budget * mu - 3) * (R[d] - lam[d]) - mu, 1e-6)
        cand.append(m.M_MAX if num <= 0 else int(np.ceil(num / den)))
    return int(np.clip(max(cand), M_min, m.M_MAX))


def sizing(rule, R, lam, z, D, M_c, alpha_c):
    """Return the dict {class: M} for the given rule (M_c = unit-size Lemma 1)."""
    mu = m.MU_C_DEFAULT
    if rule == 'lemma1_unit':
        return dict(M_c)
    if rule == 'lemma1_pkt':
        out = dict(M_c)
        out['URLLC'] = lemma1_urllc(R, lam, z, D, alpha_c['URLLC'], T_MAX / L_BITS)
        return out
    if rule == 'minstable':
        return {sc: max(int(np.ceil(alpha_c[sc] / mu)), 1) for sc in m.SERVICE_CLASSES}
    if rule == 'max':
        return {sc: m.M_MAX for sc in m.SERVICE_CLASSES}


def urllc_latencies(D, devices, orus, z, lam, R, M_u, alpha_u):
    """Packet-level URLLC latency [s] of each URLLC device (inf if unstable)."""
    mu = m.MU_C_DEFAULT
    slack = mu - alpha_u / M_u
    out = []
    for d in [i for i in range(D) if z[i] == 'URLLC']:
        if slack <= 0 or R[d] <= lam[d]:
            out.append(np.inf)
            continue
        t_tx   = L_BITS / (R[d] - lam[d])              # transmission + queueing wait
        t_proc = L_BITS * 3.0 / slack
        t_ser  = N_LINKS * L_BITS / m.C_MAX_B
        d_max  = max(np.linalg.norm(devices[d] - orus[b]) for b in range(m.B))
        out.append(t_tx + t_proc + t_ser + d_max / C_RADIO + FIBER_KM * US_PER_KM * 1e-6)
    return out


def sweep(n_runs, D_list=(9, 18, 27), lambdas_mbps=(0.1, 0.4, 0.8, 1.2, 1.62)):
    rows = []
    orig = m.LAMBDA['URLLC']
    for D in D_list:
        for lm in lambdas_mbps:
            m.LAMBDA['URLLC'] = lm * 1e6
            lat  = {r: [] for r in RULES}
            inst = {r: [] for r in RULES}
            for run_id in range(n_runs):
                np.random.seed(run_id * 100)           # same seeding as Fig. 3
                devices, orus = m.generate_positions(D)
                H = m.generate_channels(devices, orus, D)
                z = m.assign_service_classes(D)
                lam, Tmax, Pmax = m.get_device_params(z, D)
                _, R, M_c, alpha_c = m.AO_RA(D, H, z, lam, Tmax, Pmax)
                if R is None:
                    continue
                for r in RULES:
                    M = sizing(r, R, lam, z, D, M_c, alpha_c)
                    lat[r] += urllc_latencies(D, devices, orus, z, lam, R,
                                              M['URLLC'], alpha_c['URLLC'])
                    inst[r].append(3 * sum(M.values()))   # 3 nodes x sum_c M_c
            for r in RULES:
                a = np.array(lat[r])
                fin = a[np.isfinite(a)]
                rows.append(dict(
                    D=D, lambda_Mbps=lm, rule=r,
                    compliance_pct=100 * np.mean(a <= T_MAX),
                    mean_latency_ms=(fin.mean() * 1e3 if len(fin) else np.nan),
                    max_latency_ms=(fin.max() * 1e3 if len(fin) else np.nan),
                    instances=np.mean(inst[r])))
    m.LAMBDA['URLLC'] = orig
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=20)
    a = ap.parse_args()
    df = sweep(a.runs)
    os.makedirs(m.RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(m.RESULTS_DIR, 'vnf_ablation_sweep.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))
    print("\n=== SUMMARY over the whole sweep (all D and lambda) ===")
    g = df.groupby('rule')
    summ = pd.DataFrame({
        'compliance_mean_%':  g.compliance_pct.mean(),
        'compliance_worst_%': g.compliance_pct.min(),
        'mean_latency_ms':    g.mean_latency_ms.mean(),
        'worst_latency_ms':   g.max_latency_ms.max(),
        'instances_mean':     g.instances.mean()}).loc[RULES]
    print(summ.round(3).to_string())
    summ.to_csv(os.path.join(m.RESULTS_DIR, 'vnf_ablation_summary.csv'))