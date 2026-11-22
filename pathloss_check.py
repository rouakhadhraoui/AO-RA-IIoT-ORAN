"""
Standardized path-loss check for Reviewer 3, Comment 5.
Run from the folder that contains ao_ra.py:   python pathloss_check.py
It does NOT modify ao_ra.py. It replaces the channel model in memory only.

Model: log-distance path loss with free-space reference (Goldsmith, Wireless
Communications, simplified path-loss model):
    PL(d) = FSPL(d0 = 1 m, fc = 3.5 GHz) + 10*alpha*log10(max(d, 10 m))   [dB]
plus a 5 dB receiver noise figure. alpha = 3.8 as in the paper.
Same seeds / same drops as the main experiment (seed = s*1000 + D, s = 1..100).
"""
import numpy as np
import ao_ra as m

FC = 3.5e9
LAM = 3e8 / FC
PL0 = 20 * np.log10(4 * np.pi / LAM)   # free-space loss at 1 m (dB)
DMIN = 10.0                            # minimum distance (m)
NF = 5.0                               # noise figure (dB)
m.N0 = m.N0 * 10 ** (NF / 10)


def gen(devices, orus, D):
    H = np.zeros((m.B, D, m.K))
    for b in range(m.B):
        for d in range(D):
            dist = max(np.linalg.norm(devices[d] - orus[b]), DMIN)
            pl = 10 ** (-(PL0 + 10 * m.ALPHA_PL * np.log10(dist)) / 10)
            g = (np.random.randn(m.K, m.J) + 1j * np.random.randn(m.K, m.J)) / np.sqrt(2)
            H[b, d, :] = pl * np.sum(np.abs(g) ** 2, axis=1)
    return H


m.generate_channels = gen
print(f"FSPL at 1 m = {PL0:.1f} dB, d_min = {DMIN} m, NF = {NF} dB, alpha = {m.ALPHA_PL}")

N = 100
for D in (9, 18, 27, 36):
    lo, hi = 1e9, -1e9
    ua, ud, c4a, c4d, c5a, c5d, c8 = [], [], [], [], [], [], []
    unserved = 0
    for s in range(1, N + 1):
        np.random.seed(s * 1000 + D)
        dev, oru = m.generate_positions(D)
        H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D)
        lam, Tm, Pm = m.get_device_params(z, D)
        for dd in range(D):
            b0 = int(np.argmax(H[:, dd, :].mean(1)))
            v = 10 * np.log10(Pm[dd] * H[b0, dd, :] / m.N0)
            lo, hi = min(lo, v.min()), max(hi, v.max())
        Rm = m.compute_dynamic_rmin(D, H, z, Pm, silent=True)
        _, Ra, Ma, al, det = m.AO_RA(D, H, z, lam, Tm, Pm, detail=True)
        U, Rd, Md, ald = m.dr_scheme(D, H, z, lam, Tm, Pm)
        ua.append(sum(m.W[z[d]] * Ra[d] for d in range(D)))
        ud.append(U)
        for R, Mc, a_, c4, c5 in ((Ra, Ma, al, c4a, c5a), (Rd, Md, ald, c4d, c5d)):
            c4.append(np.mean([R[d] >= Rm[z[d]] - 1e-6 for d in range(D)]))
            c5.append(np.mean([m.compute_delay(d, z[d], R, lam, Mc, a_) <= m.T_MAX[z[d]]
                               for d in range(D)]))
        x = det['x_best']
        c8.append(np.mean([sum(Ra[d] for d in range(D) if x[b, d]) <= m.C_MAX_B * 1.0001
                           for b in range(m.B)]))
        unserved += int(np.sum(Ra == 0))
    ua = np.array(ua) / 1e6
    ud = np.array(ud) / 1e6
    d = ua - ud
    from scipy import stats
    try:
        p = stats.wilcoxon(d[np.abs(d) > 1e-6], alternative='greater').pvalue
    except Exception:
        p = float('nan')
    print(f"D={D}: SNR {lo:.0f}..{hi:.0f} dB | AO-RA={ua.mean():.2f} DR={ud.mean():.2f} Mbps "
          f"gain={100*d.mean()/ud.mean():.2f}% W/T/L={int((d>1e-6).sum())}/"
          f"{int((abs(d)<=1e-6).sum())}/{int((d<-1e-6).sum())} Wilcoxon p={p:.1e} | "
          f"C4 dev AO={100*np.mean(c4a):.1f} DR={100*np.mean(c4d):.1f} | "
          f"C5 dev AO={100*np.mean(c5a):.1f} DR={100*np.mean(c5d):.1f} | "
          f"C8 O-RU AO={100*np.mean(c8):.1f} | unserved={unserved}", flush=True)