"""
Figures replacing tables in the AO-RA paper, styled like fig1_convergence / fig2_utility_vs_D
(IEEE IoT Journal style): Times font, black/red/blue/green, open markers, boxed legend,
dotted grid, full frame. Same canvas size as your existing figures (6.8 x 4.4 in),
so LaTeX scales them to \\columnwidth exactly like Fig. 1 and Fig. 2.

Run:  python make_table_figures_v2.py   ->  writes PDFs into ./Figs/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

os.makedirs("Figs", exist_ok=True)

# Times New Roman on Windows; fall back to Liberation Serif / TeX Gyre Termes elsewhere
avail = {f.name for f in font_manager.fontManager.ttflist}
serif = next((f for f in ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "Nimbus Roman"]
              if f in avail), "serif")

plt.rcParams.update({
    "font.family": "serif", "font.serif": [serif], "mathtext.fontset": "stix",
    "font.size": 12, "axes.labelsize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "axes.linewidth": 1.0, "pdf.fonttype": 42,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
    "legend.framealpha": 1.0, "legend.borderpad": 0.5,
    "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.35, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

# same encoding as the paper: AO-RA black, DR red, Baseline blue, 4th dark green
COLS = ["black", "red", "blue", "darkgreen"]
LS = ["-", "--", "-.", ":"]
MK = ["o", "s", "^", "D"]
HATCH = ["", "///", "xx", ".."]
D = ["9", "18", "27", "36"]
XL = "Number of IIoT Devices ($D$)"
YL = "Weighted Sum Data Rate (Mbps)"
W, H = 6.8, 4.4


def save(fig, name):
    fig.tight_layout(pad=0.5)
    fig.savefig(f"Figs/{name}.pdf")
    plt.close(fig)
    print("saved", name)


def line(ax, xs, ys, i, label):
    ax.plot(xs, ys, color=COLS[i], ls=LS[i], marker=MK[i], ms=8, mfc="white", mew=1.4,
            lw=1.8, label=label)


def bars(ax, xs, ys, w, i, label, **kw):
    """Open-style bars: white fill + coloured hatch + coloured edge (readable in grayscale)."""
    ax.bar(xs, ys, w, label=label, facecolor="white", edgecolor=COLS[i], hatch=HATCH[i],
           linewidth=1.3, **kw)


def grouped(ax, series, labels, idx, width=0.78):
    x = np.arange(len(series[0]))
    n = len(series)
    w = width / n
    for k, (s, l) in enumerate(zip(series, labels)):
        bars(ax, x - width / 2 + w * (k + 0.5), s, w, idx[k], l)
    ax.set_xticks(x)


# ---- 1) Sensitivity: (a) I_max, (b) epsilon ---------------------------------
fig, ax = plt.subplots(1, 2, figsize=(W, 3.6))
grouped(ax[0], [[44.27, 70.98, 83.03, 87.06], [45.33, 74.87, 86.23, 88.76],
                [45.33, 74.87, 86.23, 88.76]],
        ["$I_{\\max}=1$", "$I_{\\max}=2$", "$I_{\\max}\\geq 3$"], [2, 1, 0])
ax[0].set_xticklabels(D)
ax[0].set_xlabel(XL)
ax[0].set_ylabel(YL)
ax[0].set_ylim(0, 125)
ax[0].legend(loc="upper left", fontsize=9.5)
ax[0].set_title("(a)", fontsize=12)

eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]
its = [[2.11, 2.29, 2.45, 2.50, 2.51], [2.20, 2.66, 2.78, 2.78, 2.78],
       [2.11, 2.67, 2.95, 2.98, 2.98], [2.03, 2.66, 2.92, 2.97, 2.97]]
for i, d in enumerate(D):
    line(ax[1], eps, its[i], i, f"$D={d}$")
ax[1].set_xscale("log")
ax[1].set_xlabel("Stopping Threshold ($\\varepsilon$)")
ax[1].set_ylabel("Mean Number of Outer Iterations")
ax[1].set_ylim(1.95, 3.5)
ax[1].legend(loc="upper left", ncol=2, fontsize=9.5)
ax[1].set_title("(b)", fontsize=12)
save(fig, "fig_sensitivity")

# ---- 2) Inter-cell interference ---------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(W, 3.6))
grouped(ax[0], [[45.3, 74.9, 86.2, 88.8], [16.2, 21.4, 24.7, 25.8], [4.3, 4.8, 4.8, 4.4]],
        ["Noise-limited", "Actual interference", "Worst case"], [0, 1, 2])
ax[0].set_xticklabels(D)
ax[0].set_xlabel(XL)
ax[0].set_ylabel(YL)
ax[0].set_ylim(0, 150)
ax[0].legend(loc="upper left", fontsize=9.5)
ax[0].set_title("(a) AO-RA", fontsize=12)

grouped(ax[1], [[16.2, 21.4, 24.7, 25.8], [16.3, 21.6, 24.7, 26.2], [12.3, 19.2, 23.1, 24.8]],
        ["Proposed AO-RA", "DR Scheme", "Baseline Scheme"], [0, 1, 2])
ax[1].set_xticklabels(D)
ax[1].set_xlabel(XL)
ax[1].set_ylabel(YL)
ax[1].set_ylim(0, 44)
ax[1].legend(loc="upper left", fontsize=9.5)
ax[1].set_title("(b) Under interference", fontsize=12)
save(fig, "fig_intercell")

# ---- 3) Ablation at D = 18 ---------------------------------------------------
names = ["Baseline", "GAA + random PRB", "Slice-partition", "DR Scheme",
         "AO-RA (duals frozen)", "Proposed AO-RA"]
vals = [30.56, 30.54, 70.96, 70.98, 74.87, 74.87]
ci = [2, 2, 3, 1, 0, 0]
fig, ax = plt.subplots(figsize=(W, 3.8))
y = np.arange(len(names))[::-1]
for yi, v, c, h in zip(y, vals, ci, [HATCH[2], HATCH[2], HATCH[3], HATCH[1], "", ""]):
    ax.barh(yi, v, 0.62, facecolor="white", edgecolor=COLS[c], hatch=h, linewidth=1.3)
    ax.text(v + 1, yi, f"{v:.1f}", va="center", fontsize=11)
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlabel("Weighted Sum Data Rate (Mbps), $D=18$")
ax.set_xlim(0, 88)
ax.grid(axis="y", visible=False)
save(fig, "fig_ablation")

# ---- 4) VNF-sizing ablation (log scale, 1 ms budget) ---------------------------
rules = ["Lemma 1\n($T^{\\max}/L$)", "Lemma 1\n($T^{\\max}$)", "Min-stable\n$\\lceil\\Lambda_c/\\mu_c\\rceil$",
         "Fixed\n$M_c=M_c^{\\max}$"]
mean_l = [0.524, 1.771, 1.771, 0.307]
worst_l = [0.776, 5.695, 5.695, 0.383]
inst = [20.2, 19.4, 19.4, 225]
fig, ax = plt.subplots(figsize=(W, 4.0))
xr = np.arange(4)
bars(ax, xr - 0.19, mean_l, 0.38, 0, "Mean latency")
bars(ax, xr + 0.19, worst_l, 0.38, 1, "Worst-case latency")
ax.axhline(1.0, color="black", ls="--", lw=1.3)
ax.text(3.48, 1.12, "1 ms budget", ha="right", fontsize=11)
for xi, n in zip(xr, inst):
    ax.text(xi, 9.5, f"{n:g} VNF inst.", ha="center", fontsize=10.5)
ax.set_yscale("log")
ax.set_ylim(0.2, 90)
ax.set_xticks(xr)
ax.set_xticklabels(rules)
ax.set_ylabel("URLLC End-to-End Latency (ms)")
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
save(fig, "fig_vnf_ablation")

# ---- 5) Delay components (stacked, microseconds) ---------------------------
comp = [("Processing (O-DU, O-CU-UP, UPF)", [174.55, 202.11, 240.00], 0),
        ("Transmission time $L/R_d$", [16.70, 23.95, 34.22], 1),
        ("Link serialization", [16.70, 16.70, 16.70], 2),
        ("Queueing + radio propagation", [0.23 + 0.95, 0.58 + 0.96, 1.11 + 0.97], 3)]
fig, ax = plt.subplots(figsize=(W, 4.2))
xx = np.arange(3)
bottom = np.zeros(3)
for lab, v, i in comp:
    bars(ax, xx, v, 0.5, i, lab, bottom=bottom)
    bottom += np.array(v)
for i, t in enumerate([209, 244, 293]):
    ax.text(i, t + 6, f"{t}", ha="center", fontsize=11)
ax.set_ylim(0, 430)
ax.set_xticks(xx)
ax.set_xticklabels(["9", "18", "27"])
ax.set_xlabel(XL)
ax.set_ylabel("Mean URLLC Latency ($\\mu$s)")
ax.legend(loc="upper left", fontsize=10)
save(fig, "fig_delay_components")

# ---- 6) Dynamic labels ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(W, 3.9))
xx = np.arange(3)
bars(ax, xx - 0.19, [14.3, 13.8, 16.6], 0.38, 1, "C5 violations (% of devices)")
bars(ax, xx + 0.19, [35, 33, 39], 0.38, 2, "C6 violations (% of realizations)")
ax.set_xticks(xx)
ax.set_xticklabels(["2", "4", "6"])
ax.set_xlabel("Number of Devices Changing Class ($n$)")
ax.set_ylabel("Constraint Violations (%)")
ax.set_ylim(0, 55)
ax.legend(loc="upper left")
save(fig, "fig_dynamic")

# ---- 7) Runtime breakdown ---------------------------------------------------------
total = np.array([230, 658, 1219, 1620])
sp1 = np.array([54, 102, 153, 204])
sp3 = np.array([70, 197, 356, 479]) * np.array([2.45, 2.78, 2.95, 2.92])  # per-iter x iterations
fig, ax = plt.subplots(figsize=(W, 4.2))
xx = np.arange(4)
bars(ax, xx, sp1, 0.5, 3, "SP1 (subscription + $R_c^{\\min}$)")
bars(ax, xx, sp3, 0.5, 0, "SP3 (all outer iterations)", bottom=sp1)
ax.axhline(1000, color="red", ls="--", lw=1.4)
ax.text(-0.42, 1040, "1 s (Near-RT RIC upper bound)", color="red", fontsize=10.5)
for i, t in enumerate(total):
    ax.text(i, t + 35, f"{t}", ha="center", fontsize=11)
ax.set_xticks(xx)
ax.set_xticklabels(D)
ax.set_xlabel(XL)
ax.set_ylabel("Mean Runtime (ms)")
ax.set_ylim(0, 2000)
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
save(fig, "fig_runtime")