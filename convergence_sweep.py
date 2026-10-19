import hashlib, os, sys, time
import numpy as np, pandas as pd
import ao_ra as m

def scenario(seed, D):
    np.random.seed(seed)
    dev, orus = m.generate_positions(D)
    H = m.generate_channels(dev, orus, D)
    z = m.assign_service_classes(D)
    lam, Tmax, Pmax = m.get_device_params(z, D)
    return dev, orus, H, z, lam, Tmax, Pmax

def precheck(D, dev, orus, H, z, lam, Pmax):
    for d in range(D):
        b = int(np.argmin([np.linalg.norm(dev[d]-orus[k]) for k in range(m.B)]))
        if m._rate(Pmax[d]*H[b,d,d % m.K]/m.N0, z[d], d) <= lam[d]:
            return False
    return True

def traced(D, H, z, lam, Tmax, Pmax, i_max, eps):
    R_min = m.compute_dynamic_rmin(D, H, z, Pmax, silent=True)
    eta = np.array([max(R_min[z[d]], lam[d]+1.0/m.T_MAX[z[d]]) for d in range(D)])
    x = np.zeros((m.B, D), int)
    for d in range(D):
        x[int(np.argmax([np.mean(H[b,d,:]) for b in range(m.B)])), d] = 1
    h, mu = np.zeros(D), np.zeros(m.B)
    hist, xs, bestU, bestR, reason = [], [], -np.inf, None, 'i_max'
    for i in range(i_max):
        p, a, R, h, mu = m.lagrangian_kkt(D, m.B, m.K, H, x, z, lam, Tmax, Pmax, eta, h, mu)
        U = sum(m.W[z[d]]*R[d] for d in range(D))
        hist.append(U); xs.append(hashlib.md5(x.tobytes()).hexdigest())
        if U > bestU: bestU, bestR = U, R.copy()
        if len(hist) > 1 and abs(hist[-1]-hist[-2])/max(abs(hist[-1]), 1.0) < eps:
            reason = 'eps'; break
        x = m.greedy_association(D, H, z, Pmax)
    return hist, xs, bestU, bestR, reason

def valid_seeds(D, n, seed_base=0):
    """Same selection rule as run_monte_carlo, using the reference setting."""
    seeds, attempt = [], 0
    while len(seeds) < n:
        attempt += 1
        s = seed_base + attempt*1000 + D
        dev, orus, H, z, lam, Tmax, Pmax = scenario(s, D)
        if not precheck(D, dev, orus, H, z, lam, Pmax): continue
        _, _, _, R, _ = traced(D, H, z, lam, Tmax, Pmax, m.I_MAX, m.EPS_CONV)
        if R is None or np.any(R == 0): continue
        seeds.append(s)
    return seeds

def run(D, seeds, i_max, eps, alpha0):
    m.ALPHA0_STEP = alpha0
    rows = []
    for s in seeds:
        _, _, H, z, lam, Tmax, Pmax = scenario(s, D)
        hist, xs, U, _, reason = traced(D, H, z, lam, Tmax, Pmax, i_max, eps)
        rows.append(dict(
            U=U/1e6, iters=len(hist), hit_imax=(reason == 'i_max'),
            best_ne_last=(hist[-1] < U - 1e-9),
            nonmono=any(hist[i] < hist[i-1] - 1e-9 for i in range(1, len(hist))),
            x_const_after_2=(len(set(xs[1:])) <= 1)))
    m.ALPHA0_STEP = 0.1
    df = pd.DataFrame(rows)
    return dict(D=D, i_max=i_max, eps=eps, alpha0=alpha0, n=len(df),
                U=df.U.mean(), U_std=df.U.std(), iters=df.iters.mean(),
                pct_imax=100*df.hit_imax.mean(),
                pct_best_ne_last=100*df.best_ne_last.mean(),
                pct_nonmono=100*df.nonmono.mean(),
                pct_x_const=100*df.x_const_after_2.mean())

if __name__ == "__main__":
    # usage: python convergence_sweep.py [N] [D list] [tag]
    N      = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    D_LIST = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [9, 18, 27, 36]
    TAG    = sys.argv[3] if len(sys.argv) > 3 else ""
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/convergence_sweep{TAG}.csv"
    out, t0 = [], time.time()

    def go(D, seeds, i_max, eps, a0):
        out.append(run(D, seeds, i_max, eps, a0))
        r = out[-1]
        print(f"D={D} i_max={i_max} eps={eps} a0={a0} n={r['n']} -> "
              f"U={r['U']:.2f} iters={r['iters']:.2f} "
              f"nonmono={r['pct_nonmono']:.0f}% x_const={r['pct_x_const']:.0f}% "
              f"[{time.time()-t0:.0f}s]", flush=True)

    for D in D_LIST:
        print(f"--- D={D}: selecting {N} valid seeds ---", flush=True)
        seeds = valid_seeds(D, N)
        for i_max in (1, 2, 3, 5, 10, 50):
            go(D, seeds, i_max, 1e-3, 0.1)
        for eps in (1e-1, 1e-2, 1e-4, 1e-6):
            go(D, seeds, 50, eps, 0.1)
        for a0 in (0.01, 0.05, 0.5, 1.0):
            go(D, seeds, 50, 1e-3, a0)
        go(D, seeds[:30], 50, 0.0, 0.1)          # no early stop (30 seeds)
        go(D, seeds[:30], 50, 1e-3, 0.1)         # matched reference on the same 30 seeds
        pd.DataFrame(out).to_csv(csv_path, index=False)   # save after each D

    print(pd.DataFrame(out).round(3).to_string())