
"""
R3-6 / R3-7: fixed R_min, no discarded drops, multi-PRB Baselines,
AO-RA best vs last iterate against DR.

  python fixed_rmin_eval.py --calibrate
  python fixed_rmin_eval.py --rmin 10,2,1 --tag A
  python fixed_rmin_eval.py --rmin adaptive --tag chk

Schemes:
  aora_best : AO-RA, best iterate (the algorithm as in the paper)
  aora_last : AO-RA, last iterate (no best-iterate selection)
  dr        : DR benchmark
  base1     : single random PRB per device (original Baseline)
  baseN     : multi-PRB random Baseline (nearest O-RU, <=3 PRBs), C8 NOT enforced
  baseF     : same as baseN but PRBs are added only while the running
              fronthaul load stays below C_b^max (same treatment as AO-RA)
  baseF3    : multi-PRB baseline with C8 enforced and fill enabled
  baseF3N   : multi-PRB baseline with C8 not enforced and fill enabled
"""

import argparse
import numpy as np
from scipy import stats
import ao_ra as m

D_LIST = [9, 18, 27, 36]
TOL = 1e-6   # Mbps

S = [
    'aora_best', 'aora_last', 'dr',
    'base1', 'baseN', 'baseF', 'baseF3', 'baseF3N'
]


def init_assoc(D, H):
    x = np.zeros((m.B, D), int)
    for d in range(D):
        x[int(np.argmax([H[b, d].mean() for b in range(m.B)])), d] = 1
    return x


def nearest_assoc(D, dev, oru):
    x = np.zeros((m.B, D), int)
    for d in range(D):
        x[int(np.argmin([
            np.linalg.norm(dev[d] - oru[b])
            for b in range(m.B)
        ])), d] = 1
    return x


def precheck(D, H, z, lam, Pmax, dev, oru):
    # Diagnostic only, NOT a filter
    for d in range(D):
        b = int(np.argmin([
            np.linalg.norm(dev[d] - oru[bb])
            for bb in range(m.B)
        ]))
        if m._rate(
            Pmax[d] * H[b, d, d % m.K] / m.N0, z[d], d
        ) <= lam[d]:
            return False
    return True


def compliance(D, x, R, M, alp, z, lam, rmin):
    ok4 = np.array([R[d] >= rmin[z[d]] for d in range(D)])

    ok5 = np.array([
        m.compute_delay(d, z[d], R, lam, M, alp) <= m.T_MAX[z[d]]
        for d in range(D)
    ])  # unserved / unstable -> violation

    ok8 = (x @ R) <= m.C_MAX_B * (1 + 1e-9)

    return dict(c4=ok4, c5=ok5, c8=ok8)


def calibrate():
    m.RMIN_MODE = 'adaptive'

    for D in D_LIST:
        v = {sc: [] for sc in m.SERVICE_CLASSES}

        for att in range(1, 101):
            np.random.seed(att * 1000 + D)

            dev, oru = m.generate_positions(D)
            H = m.generate_channels(dev, oru, D)
            z = m.assign_service_classes(D)
            lam, Tmax, Pmax = m.get_device_params(z, D)

            r = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)

            for sc in v:
                v[sc].append(r[sc] / 1e6)

        print(
            f"D={D}: adaptive R_min [5%, 50%, 95%] Mbps:",
            {
                sc: np.percentile(x, [5, 50, 95]).round(2).tolist()
                for sc, x in v.items()
            }
        )


def run(D, n, rmin):
    U = {s: [] for s in S}
    C = {s: [] for s in S}

    changed, pre = [], []

    for att in range(1, n + 1):  # no drop is discarded
        np.random.seed(att * 1000 + D)

        dev, oru = m.generate_positions(D)
        H = m.generate_channels(dev, oru, D)
        z = m.assign_service_classes(D)

        lam, Tmax, Pmax = m.get_device_params(z, D)

        pre.append(precheck(D, H, z, lam, Pmax, dev, oru))

        _, Rb, Mb, Ab, info = m.AO_RA(
            D, H, z, lam, Tmax, Pmax, detail=True
        )

        Rl, Ml, Al, xl = info['last']

        _, R1, M1, A1 = m.baseline_scheme(
            D, H, z, lam, Tmax, Pmax, dev, oru
        )

        _, Rd, Md, Ad = m.dr_scheme(
            D, H, z, lam, Tmax, Pmax
        )

        seed_mp = att * 7919 + D

        _, RN, MN, AN = m.baseline_multiprb(
            D, H, z, lam, Tmax, Pmax, dev, oru,
            np.random.default_rng(seed_mp)
        )

        _, RF, MF, AF = m.baseline_multiprb(
            D, H, z, lam, Tmax, Pmax, dev, oru,
            np.random.default_rng(seed_mp), c8=True
        )

        _, RG, MG, AG = m.baseline_multiprb(
            D, H, z, lam, Tmax, Pmax, dev, oru,
            np.random.default_rng(seed_mp), c8=True, fill=True
        )

        # baseF3N: same multi-PRB fill procedure, without C8 enforcement
        _, RH, MH, AH = m.baseline_multiprb(
            D, H, z, lam, Tmax, Pmax, dev, oru,
            np.random.default_rng(seed_mp), c8=False, fill=True
        )

        x_dr = init_assoc(D, H)
        x_nn = nearest_assoc(D, dev, oru)

        changed.append(not np.array_equal(xl, x_dr))

        for s, (x, R, M, A) in {
            'aora_best': (info['x_best'], Rb, Mb, Ab),
            'aora_last': (xl, Rl, Ml, Al),
            'dr': (x_dr, Rd, Md, Ad),
            'base1': (x_nn, R1, M1, A1),
            'baseN': (x_nn, RN, MN, AN),
            'baseF': (x_nn, RF, MF, AF),
            'baseF3': (x_nn, RG, MG, AG),
            'baseF3N': (x_nn, RH, MH, AH),
        }.items():

            U[s].append(
                sum(m.W[z[d]] * R[d] for d in range(D)) / 1e6
            )

            C[s].append(
                compliance(D, x, R, M, A, z, lam, rmin)
            )

    return (
        {s: np.array(u) for s, u in U.items()},
        C,
        np.array(changed),
        np.mean(pre)
    )


def wtl(a, b):
    d = a - b

    return (
        int((d > TOL).sum()),
        int((np.abs(d) <= TOL).sum()),
        int((d < -TOL).sum())
    )


def gain(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]

    return (
        100 * (a.mean() - b.mean()) / b.mean()
        if len(a) else float('nan')
    )


def wil(a, b, alt):
    d = a - b
    d = d[np.abs(d) > TOL]

    return (
        stats.wilcoxon(d, alternative=alt).pvalue
        if len(d) else float('nan')
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--rmin', default='adaptive')
    ap.add_argument('--tag', default='run')
    ap.add_argument('--n', type=int, default=100)

    args = ap.parse_args()

    if args.calibrate:
        return calibrate()

    if args.rmin == 'adaptive':
        m.RMIN_MODE = 'adaptive'

        print(
            "adaptive mode: only U (regression check) is meaningful; "
            "C4 columns are against the default 20/2/2 Mbps and must be ignored"
        )

    else:
        e, u, t = [
            float(v) * 1e6 for v in args.rmin.split(',')
        ]

        m.RMIN_MODE = 'fixed'
        m.R_MIN_FIXED = {
            'eMBB': e,
            'URLLC': u,
            'mMTC': t
        }

    rmin = m.R_MIN_FIXED

    out = {}

    for D in D_LIST:
        U, C, chg, pre = run(D, args.n, rmin)

        out[D] = dict(
            U=U,
            C=C,
            changed=chg,
            pre=pre
        )

        print(
            f"\n=== D={D}  R_min={args.rmin} Mbps  "
            f"(pre-check passes in {100*pre:.0f}% of drops; none discarded)"
        )

        print(
            f"{'scheme':10s} {'U':>7s} | "
            f"{'C4 dev':>7s} {'C5 dev':>7s} {'C8 O-RU':>8s} | "
            f"{'C4 drop':>8s} {'C5 drop':>8s} "
            f"{'C8 drop':>8s} {'all drop':>8s}"
        )

        for s in S:
            c4 = np.mean(
                np.concatenate([r['c4'] for r in C[s]])
            ) * 100

            c5 = np.mean(
                np.concatenate([r['c5'] for r in C[s]])
            ) * 100

            c8 = np.mean(
                np.concatenate([r['c8'] for r in C[s]])
            ) * 100

            d4 = np.mean([r['c4'].all() for r in C[s]]) * 100
            d5 = np.mean([r['c5'].all() for r in C[s]]) * 100
            d8 = np.mean([r['c8'].all() for r in C[s]]) * 100

            da = np.mean([
                r['c4'].all() and r['c5'].all() and r['c8'].all()
                for r in C[s]
            ]) * 100

            print(
                f"{s:10s} {U[s].mean():7.2f} | "
                f"{c4:7.1f} {c5:7.1f} {c8:8.1f} | "
                f"{d4:8.1f} {d5:8.1f} {d8:8.1f} {da:8.1f}"
            )

        nt_best = np.abs(U['aora_best'] - U['dr']) > TOL

        print(
            f"AO-RA best  vs DR: W/T/L={wtl(U['aora_best'], U['dr'])}  "
            f"gain(all)={gain(U['aora_best'], U['dr']):.2f}%"
            f"  gain(non-tied)={gain(U['aora_best'], U['dr'], nt_best):.2f}%"
            f"  p(1-sided)={wil(U['aora_best'], U['dr'], 'greater'):.1e}"
        )

        print(
            f"AO-RA last  vs DR: W/T/L={wtl(U['aora_last'], U['dr'])}  "
            f"gain(all)={gain(U['aora_last'], U['dr']):.2f}%"
            f"  gain(assoc changed, n={chg.sum()})="
            f"{gain(U['aora_last'], U['dr'], chg):.2f}%"
            f"  p(2-sided)={wil(U['aora_last'], U['dr'], 'two-sided'):.1e}"
        )

        # Print gains over all requested baseline schemes
        for s in ('base1', 'baseN', 'baseF', 'baseF3', 'baseF3N'):
            print(
                f"AO-RA best  vs {s}: "
                f"gain={gain(U['aora_best'], U[s]):.1f}%  "
                f"W/T/L={wtl(U['aora_best'], U[s])}"
            )

    np.save(
        f'results/fixed_rmin_{args.tag}.npy',
        out,
        allow_pickle=True
    )

    if args.rmin == 'adaptive':
        # Regression: must match the saved MC for D=9,18
        r = np.load(
            'results/mc_results.npy',
            allow_pickle=True
        ).item()

        for D in (9, 18):
            ref = np.asarray(
                r['aora'][D]['utility_runs']
            ) / 1e6

            print(
                f"[check] D={D} max |U_best - saved| = "
                f"{np.abs(out[D]['U']['aora_best'] - ref).max():.2e}"
            )


if __name__ == '__main__':
    main()
