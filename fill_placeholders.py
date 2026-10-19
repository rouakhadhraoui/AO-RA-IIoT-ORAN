import json, numpy as np
import ao_ra as m

orig = m.AO_RA
kept = []
def wrapped(*a, **k):
    out = orig(*a, **k)
    hist, R = out[0], out[1]
    if R is not None and not np.any(R == 0):   # same filter as run_monte_carlo
        kept.append(hist)
    return out
m.AO_RA = wrapped

res, summary = None, {}
for D in [9, 18, 27, 36]:
    kept.clear()
    res = m.run_monte_carlo([D])
    ua = np.array(res['aora'][D]['utility_runs'])
    ud = np.array(res['dr'][D]['utility_runs'])
    assert len(kept) == len(ua), (len(kept), len(ua))
    tie = np.isclose(ua, ud, rtol=1e-9, atol=0)
    nt = ~tie
    same = [(max(h) - h[-1]) <= 1e-9 * max(abs(h[-1]), 1.0) for h in kept]
    summary[D] = dict(
        n=len(ua), wins=int((ua > ud * (1 + 1e-9)).sum()), ties=int(tie.sum()),
        losses=int((ua < ud * (1 - 1e-9)).sum()),
        gain_all=float((ua.mean() - ud.mean()) / ud.mean() * 100),
        gain_nontied=float((ua[nt].mean() - ud[nt].mean()) / ud[nt].mean() * 100) if nt.any() else None,
        best_equals_last_pct=float(100 * np.mean(same)),
        iters_mean=float(np.mean([len(h) for h in kept])),
        iters_max=int(max(len(h) for h in kept)))
    print(D, summary[D], flush=True)
json.dump(summary, open('results/fill_placeholders.json', 'w'), indent=1)