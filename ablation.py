import sys
import numpy as np
import ao_ra as m
from scipy.stats import wilcoxon

D = int(sys.argv[2]) if len(sys.argv) > 2 else 18
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
V = ['baseline', 'gaa_random', 'dr', 'aora_frozen', 'slice', 'aora']
SLICE_PRBS = {'eMBB': range(0, 8), 'URLLC': range(8, 16), 'mMTC': range(16, 25)}
PRIO = {'URLLC': 0, 'eMBB': 1, 'mMTC': 2}


def nearest_assoc(devices, orus, D):
    x = np.zeros((m.B, D), dtype=int)
    for d in range(D):
        dists = [np.linalg.norm(devices[d] - orus[b]) for b in range(m.B)]
        x[int(np.argmin(dists)), d] = 1
    return x


def random_radio(x, Pmax, D):
    """Baseline radio: one random PRB per device, full power."""
    a = np.zeros((m.B, D, m.K)); p = np.zeros((D, m.K))
    for b in range(m.B):
        served = [d for d in range(D) if x[b, d] == 1]
        pool = list(range(m.K)); np.random.shuffle(pool)
        for i, d in enumerate(served):
            if i < m.K:
                a[b, d, pool[i]] = 1
                p[d, pool[i]] = Pmax[d]
    return p, a


def slice_radio(x, H, z, Pmax, D):
    """Slice-partition benchmark (inspired by [8]).
    PRBs are statically split among the classes (8/8/9). At each O-RU:
    phase 1 gives every device its best-gain free PRB inside its own slice
    (URLLC > eMBB > mMTC); phase 2 hands out the remaining PRBs of each slice
    round-robin (cap MAX_PRB_PER_DEVICE). P_max, C8 tracked, same as AO-RA."""
    a = np.zeros((m.B, D, m.K)); p = np.zeros((D, m.K))
    for b in range(m.B):
        served = [d for d in range(D) if x[b, d] == 1]
        free = {sc: set(SLICE_PRBS[sc]) for sc in m.SERVICE_CLASSES}
        cnt = {d: 0 for d in served}
        C_run = 0.0

        def give(d, k):
            nonlocal C_run
            r = m._rate(Pmax[d] * H[b, d, k] / m.N0, z[d], d)
            if r <= 0:
                return False
            a[b, d, k] = 1; p[d, k] = Pmax[d]
            free[z[d]].remove(k); cnt[d] += 1; C_run += r
            return True

        # phase 1: one PRB per device
        for d in sorted(served, key=lambda dd: PRIO[z[dd]]):
            if C_run >= m.C_MAX_B or not free[z[d]]:
                continue
            k = max(free[z[d]], key=lambda kk: H[b, d, kk])
            give(d, k)
        # phase 2: surplus PRBs of each slice, round-robin
        for sc in ['eMBB', 'URLLC', 'mMTC']:
            devs = [d for d in served if z[d] == sc]
            progress = True
            while progress and free[sc]:
                progress = False
                for d in sorted(devs, key=lambda dd: cnt[dd]):
                    if (cnt[d] >= m.MAX_PRB_PER_DEVICE or not free[sc]
                            or cnt[d] == 0):
                        continue
                    k = max(free[sc], key=lambda kk: H[b, d, kk])
                    r = m._rate(Pmax[d] * H[b, d, k] / m.N0, z[d], d)
                    if r <= 0 or C_run + r > m.C_MAX_B:
                        continue
                    give(d, k)
                    progress = True
                    break
    return p, a


def run_variant(v, D, H, z, lam, Tmax, Pmax, devices, orus):
    if v == 'baseline':
        return m.baseline_scheme(D, H, z, lam, Tmax, Pmax, devices, orus)[1]
    if v == 'dr':
        return m.dr_scheme(D, H, z, lam, Tmax, Pmax)[1]
    if v == 'aora':
        return m.AO_RA(D, H, z, lam, Tmax, Pmax)[1]
    if v == 'aora_frozen':
        m.RADIO_FROZEN = True
        try:
            return m.AO_RA(D, H, z, lam, Tmax, Pmax)[1]
        finally:
            m.RADIO_FROZEN = False
    if v == 'gaa_random':
        x = m.greedy_association(D, H, z, Pmax)
        p, a = random_radio(x, Pmax, D)
    elif v == 'slice':
        x = nearest_assoc(devices, orus, D)
        p, a = slice_radio(x, H, z, Pmax, D)
    return m.compute_rates(D, m.B, m.K, H, x, z, p, a, Pmax)


U = {v: [] for v in V}
unserved = {v: 0 for v in V}
seed = 0
while len(U['aora']) < N:
    seed += 1
    np.random.seed(seed * 1000 + D)
    devices, orus = m.generate_positions(D)
    H = m.generate_channels(devices, orus, D)
    z = m.assign_service_classes(D)
    lam, Tmax, Pmax = m.get_device_params(z, D)
    ok = True
    for d in range(D):
        b = int(np.argmin([np.linalg.norm(devices[d] - orus[k]) for k in range(m.B)]))
        if m._rate(Pmax[d] * H[b, d, d % m.K] / m.N0, z[d], d) <= lam[d]:
            ok = False
            break
    if not ok:
        continue
    for v in V:
        R_d = run_variant(v, D, H, z, lam, Tmax, Pmax, devices, orus)
        served = (R_d is not None) and not np.any(R_d == 0)
        U[v].append(0.0 if R_d is None else
                    sum(m.W[z[d]] * R_d[d] for d in range(D)) / 1e6)
        unserved[v] += (not served)

base = np.array(U['aora'])
print(f"{'aora':12s} U={base.mean():6.2f}")
for v in V:
    if v == 'aora':
        continue
    x = np.array(U[v]); diff = base - x; nz = diff[diff != 0]
    p = wilcoxon(nz, alternative='greater').pvalue if len(nz) else float('nan')
    print(f"{v:12s} U={x.mean():6.2f}  gain={(base.mean()-x.mean())/x.mean()*100:6.1f}%  "
          f"W/T/L={(diff>0).sum()}/{(diff==0).sum()}/{(diff<0).sum()}  "
          f"p={p:.1e}  unserved={unserved[v]/N*100:.0f}%")
np.savez(f'ablation_results_D{D}.npz', **{v: np.array(U[v]) for v in V})