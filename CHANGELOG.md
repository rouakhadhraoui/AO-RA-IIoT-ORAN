# Changelog

## v2.0-revision (October 2026)

Code and documentation for the revised manuscript (IEEE Internet of Things
Journal, IoT-71590-2026). The numerical results of the original submission are
unchanged, except for rounding corrections in the text and the regenerated
Fig. 4. Changes are listed per reviewer comment.

### Reviewer 1

- **Comment 1 (intra-cell interference):** interference terms removed from the
  model; the SNR is interference-free under PRB exclusivity (C2). Power policy
  documented as a threshold (on/off) rule at `P_max`; PRB rule documented as a
  dual-guided greedy heuristic that enforces C2 by construction. Fig. 4
  regenerated from the current code.
- **Comment 2 (statistical significance):** new script `statistical_analysis.py`
  (bootstrap 95% confidence intervals, one-sided paired Wilcoxon tests,
  wins/ties/losses of AO-RA vs. DR).
- **Comment 3 (total data rate):** notation made explicit: per-PRB rate `r_d^k`
  and total rate `R_d` as the sum over assigned PRBs. No change to the
  numerical results.
- **Comment 4 (identical VNF nodes):** documented as a modeling simplification;
  generalization to node-specific rates described in the paper.
- **Comment 5 (delay components):** new script `delay_components.py`
  (explicit packet size, serialization, radio and fiber propagation); packet-level
  VNF sizing with budget `T_max / L`.

### Reviewer 2

- **Comment 1 (interference bounding):** new scripts `intercell_interference.py`
  and `intercell_ranking.py` (a posteriori evaluation under actual inter-cell
  interference and under a worst-case bound).
- **Comment 2 (benchmarks and ablation):** new scripts `ablation.py` (component-wise
  ablation and slice-partition benchmark), `vnf_ablation.py` (VNF-sizing
  ablation) and `check_duals.py` (values of the dual variables).
- **Comment 3 (dynamic labels, mobility):** new script `dynamic_labels.py`
  (preliminary evaluation of a service-label change).
- **Comment 4 (convergence):** new script `convergence_sweep.py` (sensitivity to
  `I_max`, `ε`, `α_0`; best-iterate statistics).
- **Comment 5 (runtime):** new script `runtime_benchmark.py` (wall-clock runtime
  per sub-problem and for the complete algorithm).
- **Comment 6:** no change (the list of references was removed from the
  decision letter).

### Documentation

- README updated (model scope, sub-problem table, figure and table mapping, new
  scripts, reproducibility details).
- `CITATION.cff` updated (version, repository URL, license, lab affiliation).

## v1.0-submitted (July 2026)

Reference implementation corresponding to the original submission.