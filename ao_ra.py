"""
AO-RA: Alternating Optimization-based Resource Allocation for Massive IIoT in O-RAN
====================================================================================

Reference implementation accompanying the paper:

    R. Khadhraoui, L. Ferdouse, and L. Nasraoui,
    "Subscription-based Slicing and Resource Allocation for Massive IIoT in O-RAN"

This module contains:
    - System parameters (Table III of the paper)
    - Rate & SINR models (Section IV-A, Eqs. 6-9, 17)
    - Scenario generation & SP1 service identification (Section III, V-B)
    - SP2: O-RU association via GAA (Algorithm 1)
    - SP3: PRB and power allocation via Lagrangian + KKT (Section V-D)
    - SP4: VNF sizing via closed-form Lemma 1 (Section V-E)
    - AO-RA main loop (Algorithm 2)
    - Baseline and Dynamic Resource (DR) benchmark schemes (Section VI-A)
    - Monte Carlo experiment and figure-generation utilities (Section VI)

All results in the paper are averaged over 100 Monte Carlo realizations with
deterministic seeding, and can be regenerated with `generate_figures.py`.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

# Reproducibility: fixed seed so all results in the paper can be regenerated
np.random.seed(42)

# Output directory for figures and Monte Carlo results
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# 1. SYSTEM PARAMETERS (Table III)
# ============================================================================

# Network topology
CELL_RADIUS = 250        # Cell radius (500 m diameter)
B           = 6          # Number of O-RUs
J           = 4          # Antennas per O-RU
K           = 25         # Number of PRBs per O-RU

# Radio / channel model
BW       = 180e3         # Bandwidth per PRB [Hz]
ALPHA_PL = 3.8           # Path-loss exponent (alpha)
N0_dBm_Hz = -174         # Noise PSD [dBm/Hz]
N0 = (10 ** (N0_dBm_Hz / 10)) * 1e-3 * BW   # Noise power over one PRB [W]

# Per-class device limits (constraint C1)
# Max transmit power P_d^max, converted dBm -> W
P_MAX = {
    'eMBB':  10 ** (33 / 10) * 1e-3,   # 33 dBm ~ 2.0 W
    'URLLC': 10 ** (33 / 10) * 1e-3,   # 33 dBm
    'mMTC':  10 ** (20 / 10) * 1e-3,   # 20 dBm = 0.1 W (low-power sensors)
}

# QoS requirements per service class (constraints C4, C5)
T_MAX = {'eMBB': 4e-3, 'URLLC': 1e-3, 'mMTC': 5e-3}   # Max E2E delay [s]
R_MIN = {'eMBB': 20e6, 'URLLC': 2e6,  'mMTC': 2e6}    # Min data rate [bit/s]

# Fronthaul (constraint C8)
C_MAX_B = 46e6           # Max fronthaul capacity per O-RU [bit/s]

# Traffic model
LAMBDA = {               # Mean packet arrival rate lambda_d [bit/s]
    'eMBB':  3e6,
    'URLLC': 0.2e6,
    'mMTC':  0.2e6,
}

# Finite-blocklength parameters (URLLC/mMTC short packets, Eq. 7)
PKT_SIZE = {'URLLC': 32,   'mMTC': 20}      # Packet size [bytes]
N_BLOCK  = {'URLLC': 32*8, 'mMTC': 20*8}    # Blocklength N_d [channel uses ~ bits]
EPSILON  = {'URLLC': 1e-5, 'mMTC': 1e-3}    # Target BLER epsilon_d

# VNF / queueing model (SP4, constraints C6, C7)
M_MAX        = 25        # Max VNF instances per slice (C7)
MU_C_DEFAULT = 5e6       # Service rate per VNF instance mu_c [bit/s]

# Objective weights
W = {'eMBB': 1/3, 'URLLC': 1/3, 'mMTC': 1/3}   # Equal priority weights w_c
SERVICE_CLASSES = ['eMBB', 'URLLC', 'mMTC']

# Algorithm settings (AO-RA, Algorithm 2)
I_MAX       = 50         # Max outer AO iterations
T_INNER     = 20         # Inner sub-gradient iterations (SP3)
EPS_CONV    = 1e-3       # Convergence threshold on weighted sum rate
MC_RUNS     = 100        # Monte Carlo realizations
ALPHA0_STEP = 0.1        # Initial sub-gradient step size alpha_0

MAX_PRB_PER_DEVICE = 3   # Cap on PRBs per device (fairness rule in SP3)


# ============================================================================
# 2. RATE & SINR MODELS (Section IV-A)
# ============================================================================

def _rate(sinr, sc, d):
    """
    Achievable data rate on one PRB for a device of service class `sc`.

    Implements Eq. (6) for eMBB (Shannon) and Eqs. (7)-(8) for URLLC/mMTC
    (finite-blocklength regime).

    Parameters
    ----------
    sinr : float   -- SINR on the PRB (linear scale)
    sc   : str     -- service class in {'eMBB', 'URLLC', 'mMTC'}
    d    : int     -- device index (kept for interface consistency)

    Returns
    -------
    float -- achievable rate [bit/s], clipped at 0
    """
    if sinr <= 0:
        return 0.0

    # Shannon capacity per PRB, Eq. (6)
    shannon = BW * np.log2(1.0 + sinr)
    if sc == 'eMBB':
        return shannon

    # Finite-blocklength penalty for short packets (URLLC/mMTC), Eq. (7)
    N_d    = N_BLOCK[sc]                       # blocklength [channel uses]
    eps_d  = EPSILON[sc]                       # target block error probability
    C_disp = 1.0 - 1.0 / (1.0 + sinr) ** 2     # channel dispersion V, Eq. (8)

    # Penalty term zeta = Bw * log2(e) * Q^{-1}(eps) * sqrt(V / N_d)
    # Q^{-1}(eps) = norm.ppf(1 - eps): inverse of the Gaussian tail function
    penalty = (BW * np.log2(np.e) * norm.ppf(1.0 - eps_d)
               * np.sqrt(C_disp / N_d))

    return max(0.0, shannon - penalty)         # rate cannot be negative


def sinr_lower_bound(d, b, k, p, H, x, D, P_max_arr):
    """
    Lower-bound SINR of device d at O-RU b on PRB k, Eq. (17).

    Interference is upper-bounded by assuming every other device
    associated with O-RU b (x[b, d2] = 1) transmits at its maximum
    power P_max. This makes the rate concave in p[d, k] (SP3).
    """
    # Worst-case interference: co-associated devices at full power
    I_bar = sum(
        P_max_arr[d2] * H[b, d2, k] * x[b, d2]
        for d2 in range(D) if d2 != d
    )
    # max(..., 1e-30) guards against division by zero
    return (p[d, k] * H[b, d, k]) / max(N0 + I_bar, 1e-30)


# ============================================================================
# 3. SCENARIO GENERATION & SP1: SERVICE IDENTIFICATION (Sections III, V-B)
# ============================================================================

def generate_positions(D):
    """
    Generate random device positions and fixed O-RU positions.

    Devices: uniform over the circular cell (r = R * sqrt(u) gives a
    uniform distribution over the disk area, not just the radius).
    O-RUs:  B = 6 units equally spaced on a ring of radius CELL_RADIUS/2.
    """
    r       = CELL_RADIUS * np.sqrt(np.random.rand(D))   # uniform over disk
    theta   = 2 * np.pi * np.random.rand(D)
    devices = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    oru_angles = np.linspace(0, 2 * np.pi, B, endpoint=False)
    oru_r  = 0.5 * CELL_RADIUS                            # O-RU ring radius
    orus   = np.column_stack([oru_r * np.cos(oru_angles),
                              oru_r * np.sin(oru_angles)])
    return devices, orus


def generate_channels(devices, orus, D):
    """
    Generate channel power gains H[b, d, k], Eqs. (3)-(4).

    Large-scale: path loss d^{-alpha} (distance floored at 1 m).
    Small-scale: Rayleigh fading, h_tilde ~ CN(0, I_J).
    MRC over J antennas: effective gain = pl * ||h_tilde||^2.
    """
    H = np.zeros((B, D, K))
    for b in range(B):
        for d in range(D):
            dist = max(np.linalg.norm(devices[d] - orus[b]), 1.0)
            pl   = dist ** (-ALPHA_PL)                    # path loss
            for k in range(K):                            # i.i.d. per PRB
                h_tilde    = (np.random.randn(J) + 1j * np.random.randn(J)) / np.sqrt(2)
                H[b, d, k] = pl * float(np.real(np.conj(h_tilde) @ h_tilde))
    return H


def assign_service_classes(D):
    """
    SP1 -- Subscription-based service identification (fixed labels z_{d,c}).

    Devices are split equally: first D/3 -> eMBB, next D/3 -> URLLC,
    remainder -> mMTC. Labels are fixed BEFORE optimization (no online
    inference), as enforced by the Near-RT RIC in the paper.
    """
    z         = {}
    per_class = D // 3
    for d in range(D):
        if   d < per_class:         z[d] = 'eMBB'
        elif d < 2 * per_class:     z[d] = 'URLLC'
        else:                       z[d] = 'mMTC'
    return z


def get_device_params(z, D):
    """
    Expand per-class parameters into per-device arrays.

    Returns
    -------
    lambda_d : (D,) packet arrival rates [bit/s]
    T_max_d  : (D,) max tolerable E2E delays [s]     (constraint C5)
    P_max_d  : (D,) max transmit powers [W]          (constraint C1)
    """
    lambda_d = np.array([LAMBDA[z[d]] for d in range(D)])
    T_max_d  = np.array([T_MAX[z[d]]  for d in range(D)])
    P_max_d  = np.array([P_MAX[z[d]]  for d in range(D)])
    return lambda_d, T_max_d, P_max_d


def compute_dynamic_rmin(D, H, z, P_max_d, silent=False):
    """
    Channel-adaptive per-class minimum rate requirement (feeds C4).

    For each service class: find the best interference-free rate each
    device could achieve (max over O-RUs and PRBs at full power), take
    the WEAKEST device in the class, and set the class requirement to
    50% of that value (floored at 1 kbps).
    """
    R_dyn = {}
    for sc in SERVICE_CLASSES:
        sc_devices = [d for d in range(D) if z[d] == sc]
        if not sc_devices:                     # class absent -> fixed value
            R_dyn[sc] = R_MIN[sc]
            continue
        min_rate_in_class = np.inf
        for d in sc_devices:
            best_rate_d = 0.0
            for b in range(B):
                for k in range(K):
                    # interference-free SINR at full power
                    sinr_max = P_max_d[d] * H[b, d, k] / N0
                    r = _rate(sinr_max, sc, d)
                    best_rate_d = max(best_rate_d, r)
            min_rate_in_class = min(min_rate_in_class, best_rate_d)
        R_dyn[sc] = max(min_rate_in_class * 0.5, 1e3)   # 50% margin, 1 kbps floor
    if not silent:
        print(f"  [Dynamic R_MIN] eMBB={R_dyn['eMBB']/1e6:.3f} Mbps | "
              f"URLLC={R_dyn['URLLC']/1e6:.3f} Mbps | "
              f"mMTC={R_dyn['mMTC']/1e6:.3f} Mbps")
    return R_dyn


# ============================================================================
# 4. SP2: O-RU ASSOCIATION (GAA, Algorithm 1)
# ============================================================================

def greedy_association(D, H, z, P_max_d):
    """
    SP2 -- Greedy Assignment Algorithm (GAA), Algorithm 1 of the paper.
    Executed at the Near-RT RIC. Determines x_{b,d} subject to C3, C8.

    Returns
    -------
    x : (B, D) binary association matrix (exactly one 1 per column -> C3)
    """
    x        = np.zeros((B, D), dtype=int)
    C_tilde  = np.full(B, C_MAX_B, dtype=float)   # residual fronthaul capacity (C8)
    n_served = np.zeros(B, dtype=int)             # devices already on each O-RU
    priority = {'URLLC': 0, 'eMBB': 1, 'mMTC': 2}
    order    = sorted(range(D), key=lambda d: priority[z[d]])  # URLLC first

    def estimate_rate(d, b):
        """Interference-free rate proxy, discounted by a crowding factor."""
        ch_mean    = np.mean(H[b, d, :])          # mean gain over the K PRBs
        sinr_est   = P_max_d[d] * ch_mean / N0    # full power, no interference
        r_est      = _rate(sinr_est, z[d], d)
        # Crowding: devices already served will compete for the same K PRBs
        crowd_factor = K / (K + n_served[b])
        return r_est * crowd_factor

    for d in order:
        # Pick the feasible O-RU with the highest weighted rate
        best_b, best_score, best_r = -1, -np.inf, 0.0
        for b in range(B):
            r_est    = estimate_rate(d, b)
            feasible = C_tilde[b] >= r_est         # fronthaul check (C8)
            score    = W[z[d]] * r_est if feasible else -np.inf
            if score > best_score:
                best_score, best_b, best_r = score, b, r_est

        # Fallback: no feasible O-RU -> largest residual capacity (C3)
        if best_b < 0:
            best_b = int(np.argmax(C_tilde))
            best_r = estimate_rate(d, best_b)

        # Commit the assignment and update residual capacity
        x[best_b, d]     = 1
        C_tilde[best_b]  = max(0.0, C_tilde[best_b] - best_r)
        n_served[best_b] += 1

    return x


# ============================================================================
# 5. SP3: PRB AND POWER ALLOCATION (Lagrangian + KKT, Section V-D)
# ============================================================================

def compute_rates(D, B, K, H, x, z, p, a, P_max_d):
    """
    Compute the TRUE achievable rate of each device, Eqs. (5)-(9),
    given a complete allocation (x, a, p).

    A device's total rate is the sum over its assigned PRBs.

    Returns
    -------
    R_d : (D,) achievable rates [bit/s]; 0.0 if a device has no PRB
    """
    R_d = np.zeros(D)
    for d in range(D):
        b_d      = int(np.argmax(x[:, d]))   # serving O-RU (C3: unique)
        sc       = z[d]
        assigned = False
        for k in range(K):
            if a[b_d, d, k] > 0.5:           # PRB k assigned to device d
                assigned = True
                # Actual interference from co-associated devices on PRB k
                I_bar = 0.0
                for d2 in range(D):
                    if d2 != d:
                        I_bar += p[d2, k] * H[b_d, d2, k] * x[b_d, d2]
                sinr    = p[d, k] * H[b_d, d, k] / max(N0 + I_bar, 1e-30)
                R_d[d] += _rate(sinr, sc, d)  # sum over assigned PRBs
        if not assigned:
            R_d[d] = 0.0                      # starved device
    return R_d


def lagrangian_kkt(D, B, K, H, x, z, lambda_d, T_max_d, P_max_d,
                   eta_d, h_dual_in, m_dual_in):
    """
    SP3 -- PRB and power allocation at the O-DU (Section V-D).

    Dual-guided two-phase greedy allocation per O-RU, iterated T_INNER
    times with sub-gradient updates of the dual variables:
      h_dual[d] : price of the merged rate bound eta_d (C4+C5)
      m_dual[b] : price of the fronthaul capacity (C8)

    Returns
    -------
    p, a   : power (D, K) and PRB assignment (B, D, K)
    R_d    : (D,) true rates for the final allocation
    h_dual, m_dual : updated dual variables (warm start for next call)
    """
    h_dual = h_dual_in.copy()
    m_dual = m_dual_in.copy()
    a      = np.zeros((B, D, K))
    p      = np.zeros((D, K))

    # Phase 1 serves strictest QoS first; phase 2 gives surplus PRBs
    # to eMBB first (highest rate demand), then URLLC, then mMTC.
    phase1_order = {'URLLC': 0, 'eMBB': 1, 'mMTC': 2}
    phase2_order = ['eMBB', 'URLLC', 'mMTC']

    for t in range(T_INNER):
        step  = ALPHA0_STEP / np.sqrt(t + 1)   # diminishing step (Sec. V-D)
        a_new = np.zeros((B, D, K))
        p_new = np.zeros((D, K))

        for b in range(B):
            served = [d for d in range(D) if x[b, d] == 1]
            if not served:
                continue

            # Threshold (bang-bang) power decision per device, Eq. (power_threshold)
            # Net dual weight: benefit (w + h_d) minus fronthaul price m_b.
            # Positive -> transmit at P_max; negative -> stay silent.
            p_cand = {}
            for d in served:
                A_d       = (W[z[d]] + h_dual[d]) - m_dual[b]
                p_cand[d] = P_max_d[d] if A_d >= 0 else 0.0

            C_b_running = 0.0                    # running fronthaul load (C8)
            taken_k     = set()                  # PRB exclusivity (C2)
            prb_count   = {d: 0 for d in served} # per-device cap tracking

            # PHASE 1: guarantee one best PRB per device
            for d in sorted(served, key=lambda dd: phase1_order[z[dd]]):
                if C_b_running >= C_MAX_B or p_cand[d] <= 0:
                    continue
                best_k, best_gain = -1, -np.inf
                for k in range(K):
                    if k not in taken_k and H[b, d, k] > best_gain:
                        best_gain, best_k = H[b, d, k], k
                if best_k >= 0:
                    sinr_cand           = p_cand[d] * H[b, d, best_k] / max(N0, 1e-30)
                    r_cand              = _rate(sinr_cand, z[d], d)
                    a_new[b, d, best_k] = 1
                    p_new[d, best_k]    = p_cand[d]
                    taken_k.add(best_k)
                    C_b_running        += r_cand
                    prb_count[d]       += 1

            # PHASE 2: surplus PRBs by KKT-inspired score Z, Eq. (19)
            for sc_priority in phase2_order:
                # Eligible: same class, under the PRB cap, transmitting
                sc_served = sorted(
                    [d for d in served
                     if z[d] == sc_priority
                     and prb_count[d] < MAX_PRB_PER_DEVICE
                     and p_cand[d] > 0],
                    key=lambda d: prb_count[d]
                )
                if not sc_served:
                    continue

                made_assignment = True
                while made_assignment:
                    made_assignment = False
                    for k in range(K):
                        if k in taken_k:
                            continue
                        if C_b_running >= C_MAX_B:   # fronthaul full (C8)
                            break

                        best_d, best_Z, best_r = None, -np.inf, 0.0
                        # Round-robin fairness: only devices with the
                        # current minimum PRB count may compete
                        min_prb_count = min(
                            prb_count[d] for d in sc_served
                            if prb_count[d] < MAX_PRB_PER_DEVICE
                        ) if sc_served else 0

                        for d in sc_served:
                            if prb_count[d] >= MAX_PRB_PER_DEVICE:
                                continue
                            if prb_count[d] > min_prb_count:
                                continue
                            sinr_cand = (p_cand[d] * H[b, d, k]
                                         / max(N0, 1e-30))
                            r_cand    = _rate(sinr_cand, z[d], d)
                            # Selection metric (Eq. 19 structure):
                            # weighted rate minus fronthaul dual cost
                            Z_d       = (r_cand * (W[z[d]] + h_dual[d])
                                         - m_dual[b] * H[b, d, k]
                                         * p_cand[d])
                            if Z_d > best_Z:
                                best_Z, best_d, best_r = Z_d, d, r_cand

                        if best_d is not None and best_r > 0:
                            if C_b_running + best_r > C_MAX_B:  # would break C8
                                continue
                            a_new[b, best_d, k] = 1
                            p_new[best_d, k]    = p_cand[best_d]
                            taken_k.add(k)
                            C_b_running        += best_r
                            prb_count[best_d]  += 1
                            made_assignment     = True
                            # Re-sort eligible set after each assignment
                            sc_served = sorted(
                                [d for d in sc_served
                                 if prb_count[d] < MAX_PRB_PER_DEVICE],
                                key=lambda d: prb_count[d]
                            )
                            break

        a   = a_new
        p   = p_new
        R_d = compute_rates(D, B, K, H, x, z, p, a, P_max_d)

        # Sub-gradient dual updates (Sec. V-D)
        # h_d grows while device d misses its rate bound eta_d (C4+C5)
        for d in range(D):
            grad      = (eta_d[d] - R_d[d]) / max(eta_d[d], 1.0)  # normalized
            h_dual[d] = max(0.0, h_dual[d] + step * grad)          # projection

        # m_b grows while O-RU b exceeds its fronthaul capacity (C8)
        for b in range(B):
            C_b       = sum(x[b, d] * R_d[d] for d in range(D))
            grad      = (C_b - C_MAX_B) / C_MAX_B                  # normalized
            m_dual[b] = max(0.0, m_dual[b] + step * grad)

    R_d = compute_rates(D, B, K, H, x, z, p, a, P_max_d)
    return p, a, R_d, h_dual, m_dual


# ============================================================================
# 6. SP4: VNF SIZING (Lemma 1) AND DELAY MODEL (Eqs. 10-14)
# ============================================================================

def compute_vnf(z, R_d, lambda_d, D, mu_c=MU_C_DEFAULT):
    """
    SP4 -- Closed-form VNF sizing per slice, Lemma 1 (at O-CU-UP/UPF).

    For each slice c, computes the minimum M_c such that every device of
    the slice satisfies its delay budget T_c^max (C5), clipped to the
    stability bound C6 (M_c >= Lambda_c / mu_c) and the budget C7
    (M_c <= M_MAX).

    Returns
    -------
    M_c     : dict {class -> number of VNF instances}
    alpha_c : dict {class -> aggregate arrival rate Lambda_c [bit/s]}
    """
    # Aggregate slice arrival rate: Lambda_c, Eq. (10)
    alpha_c = {sc: sum(lambda_d[d] for d in range(D) if z[d] == sc)
               for sc in SERVICE_CLASSES}
    M_c = {}

    for sc in SERVICE_CLASSES:
        T_sc  = T_MAX[sc]                              # delay budget T_c^max
        alp   = alpha_c[sc]
        M_min = max(int(np.ceil(alp / mu_c)), 1)       # C6: queue stability

        candidates = []
        for d in range(D):
            if z[d] != sc:
                continue
            R   = R_d[d]
            lam = lambda_d[d]
            # Invalid regime: rate below arrival rate -> T_tx undefined.
            # Conservatively request the maximum number of VNFs.
            if R <= lam + 1e-10:
                candidates.append(M_MAX)
                continue
            # Lemma 1 closed form
            num = alp * (T_sc * R - T_sc * lam - 1)
            den = max((T_sc * mu_c - 3) * (R - lam) - mu_c, 1e-6)  # guard /0
            if num <= 0:
                candidates.append(M_MAX)
            else:
                candidates.append(int(np.ceil(num / den)))

        # Max over devices: the slice's worst device drives the sizing (C5)
        M_star  = max(candidates) if candidates else M_min
        M_c[sc] = int(np.clip(M_star, M_min, M_MAX))   # clip to [C6, C7]

    return M_c, alpha_c


def compute_delay(d, sc, R_d, lambda_d, M_c, alpha_c,
                  mu_c=MU_C_DEFAULT):
    """
    End-to-end delay of device d, Eq. (14):
        T_tot = T_tx + 3 * T_proc
    where T_tx   = 1 / (R_d - lambda_d)           (uplink queuing)
          T_proc = 1 / (mu_c - Lambda_c / M_c)    (per O-RAN node; x3 for
                                                   O-DU, O-CU-UP, UPF)

    Returns np.inf when the queueing model is invalid.
    """
    R   = R_d[d]
    lam = lambda_d[d]
    if R <= lam + 1e-6 or M_c[sc] <= 0:
        return np.inf                      # queue unstable / no capacity
    T_tx    = 1.0 / (R - lam)              # transmission queuing delay
    denom_q = mu_c - alpha_c[sc] / M_c[sc] # per-node queueing slack
    T_proc  = 3.0 / denom_q if denom_q > 0 else np.inf   # 3 nodes in series
    return T_tx + T_proc                   # [seconds]


# ============================================================================
# 7. AO-RA MAIN ALGORITHM (Algorithm 2)
# ============================================================================

def AO_RA(D, H, z, lambda_d, T_max_d, P_max_d, mu_c=MU_C_DEFAULT):
    """
    AO-RA -- Algorithm 2: the complete alternating-optimization loop.

    Pre-processing: dynamic per-class R_min, merged rate bound eta_d
    (C4+C5), best-mean-channel initial association, zero-initialized duals.
    Loop: SP3 (PRB+power) -> SP4 (VNF sizing) -> SP2 (GAA association),
    tracking the weighted sum rate U until relative convergence.

    Returns
    -------
    history      : list of U values per outer iteration [bit/s]
    best_R_d     : (D,) rates of the best iterate
    best_M_c     : dict, VNF allocation of the best iterate
    best_alpha_c : dict, aggregate arrival rates Lambda_c
    """
    # Pre-processing (SP1 outputs + merged rate bound)
    R_min_dyn = compute_dynamic_rmin(D, H, z, P_max_d, silent=True)

    # eta_d = max(R_min, lambda + 1/T_max): one bound covering C4 and
    # the transmission part of C5 (see Sec. V, merged bound)
    eta_d = np.array([
        max(R_min_dyn[z[d]],
            lambda_d[d] + 1.0 / T_MAX[z[d]])
        for d in range(D)
    ])

    # Feasible initial association: best mean channel gain (Sec. VI)
    x = np.zeros((B, D), dtype=int)
    for d in range(D):
        best_b       = int(np.argmax([np.mean(H[b, d, :]) for b in range(B)]))
        x[best_b, d] = 1

    # Dual variables start at zero; warm-started across outer iterations
    h_dual = np.zeros(D)
    m_dual = np.zeros(B)

    history = []
    best_U, best_R_d, best_M_c, best_alpha_c = -np.inf, None, None, None

    for i in range(I_MAX):
        # SP3 (O-DU): PRB + power given x
        p, a, R_d, h_dual, m_dual = lagrangian_kkt(
            D, B, K, H, x, z, lambda_d, T_max_d, P_max_d,
            eta_d, h_dual, m_dual
        )
        # SP4 (O-CU-UP/UPF): VNF sizing given rates
        M_c, alpha_c = compute_vnf(z, R_d, lambda_d, D, mu_c)

        # Objective: weighted sum data rate (Eq. 15)
        U = sum(W[z[d]] * R_d[d] for d in range(D))
        history.append(U)

        # Keep the best iterate seen so far
        if U > best_U:
            best_U, best_R_d, best_M_c, best_alpha_c = \
                U, R_d.copy(), dict(M_c), dict(alpha_c)

        # Relative convergence test
        if len(history) > 1:
            diff = abs(history[-1] - history[-2]) / max(abs(history[-1]), 1.0)
            if diff < EPS_CONV:
                break

        # SP2 (Near-RT RIC): refresh association for next iteration
        x = greedy_association(D, H, z, P_max_d)

    return history, best_R_d, best_M_c, best_alpha_c


# ============================================================================
# 8. BENCHMARK SCHEMES: BASELINE & DR (Section VI-A)
# ============================================================================

def baseline_scheme(D, H, z, lambda_d, T_max_d, P_max_d,
                    devices, orus, mu_c=MU_C_DEFAULT):
    """
    Baseline benchmark (Sec. VI-A): lower performance bound.

    - Association: nearest O-RU by Euclidean distance
    - PRB: ONE random PRB per device (shuffled pool per O-RU)
    - Power: fixed at P_max on the assigned PRB
    - VNF: Lemma 1 (same as AO-RA)
    """
    # Nearest-O-RU association (geometric, load-blind)
    x = np.zeros((B, D), dtype=int)
    for d in range(D):
        dists = [np.linalg.norm(devices[d] - orus[b]) for b in range(B)]
        x[int(np.argmin(dists)), d] = 1

    # Random single-PRB assignment per O-RU (no channel knowledge)
    a = np.zeros((B, D, K))
    for b in range(B):
        served   = [d for d in range(D) if x[b, d] == 1]
        prb_pool = list(range(K))
        np.random.shuffle(prb_pool)          # consumes RNG state
        for idx, d in enumerate(served):
            if idx < K:                       # more devices than PRBs -> starved
                a[b, d, prb_pool[idx]] = 1

    # Full power on the assigned PRB
    p = np.zeros((D, K))
    for d in range(D):
        b_d = int(np.argmax(x[:, d]))
        for k in range(K):
            if a[b_d, d, k] > 0.5:
                p[d, k] = P_max_d[d]

    R_d          = compute_rates(D, B, K, H, x, z, p, a, P_max_d)
    M_c, alpha_c = compute_vnf(z, R_d, lambda_d, D, mu_c)
    U            = sum(W[z[d]] * R_d[d] for d in range(D))
    return U, R_d, M_c, alpha_c


def dr_scheme(D, H, z, lambda_d, T_max_d, P_max_d, mu_c=MU_C_DEFAULT):
    """
    Dynamic Resource (DR) benchmark (Sec. VI-A): SP2 ablation.

    Identical machinery to AO-RA (same eta_d, same SP3 Lagrangian+KKT,
    same SP4 Lemma 1) EXCEPT the association x is fixed once to the
    best-mean-channel O-RU and never refreshed by the GAA.
    """
    # Static association: strongest mean received channel (never updated)
    x = np.zeros((B, D), dtype=int)
    for d in range(D):
        x[int(np.argmax([np.mean(H[b, d, :]) for b in range(B)])), d] = 1

    # Same pre-processing as AO-RA
    R_min_dyn = compute_dynamic_rmin(D, H, z, P_max_d, silent=True)
    eta_d = np.array([
        max(R_min_dyn[z[d]],
            lambda_d[d] + 1.0 / T_MAX[z[d]])
        for d in range(D)
    ])

    h_dual = np.zeros(D)
    m_dual = np.zeros(B)

    best_U, best_R_d, best_M_c, best_alpha_c = -np.inf, None, None, None
    prev_U = None

    for _ in range(I_MAX):
        p, a, R_d, h_dual, m_dual = lagrangian_kkt(
            D, B, K, H, x, z, lambda_d, T_max_d, P_max_d,
            eta_d, h_dual, m_dual)
        M_c, alpha_c = compute_vnf(z, R_d, lambda_d, D, mu_c)
        U = sum(W[z[d]] * R_d[d] for d in range(D))

        if U > best_U:
            best_U, best_R_d = U, R_d.copy()
            best_M_c, best_alpha_c = dict(M_c), dict(alpha_c)

        # Same relative convergence test as AO-RA (x never changes here)
        if (prev_U is not None
                and abs(U - prev_U) / max(abs(U), 1.0) < EPS_CONV):
            break
        prev_U = U

    return best_U, best_R_d, best_M_c, best_alpha_c


# ============================================================================
# 9. MONTE CARLO EXPERIMENT (Section VI)
# ============================================================================

def run_monte_carlo(D_list, n_runs=MC_RUNS, seed_base=0):
    """
    Main Monte Carlo experiment (Sec. VI): paired evaluation of AO-RA,
    Baseline, and DR over n_runs valid realizations per device count.

    Feasibility filtering: a realization is kept only if (a) every device
    passes the nearest-O-RU full-power rate pre-check, and (b) AO-RA
    leaves no device with zero rate.
    """
    results = {s: {D: {} for D in D_list}
               for s in ['aora', 'baseline', 'dr']}

    for D in D_list:
        print(f"\n{'='*60}")
        print(f"  D = {D}  ({n_runs} Monte Carlo runs)")
        print(f"{'='*60}")

        u_ao, u_ba, u_dr       = [], [], []   # weighted sum rates
        d_ao, d_ba, d_dr       = [], [], []   # mean URLLC delays
        vnf_ao, vnf_ba, vnf_dr = [], [], []   # VNF counts per class
        histories              = []           # AO-RA convergence traces

        valid, attempt = 0, 0

        while valid < n_runs:
            attempt += 1
            # Deterministic, unique seed per (attempt, D)
            np.random.seed(seed_base + attempt * 1000 + D)

            # One shared scenario for all three schemes (paired)
            devices, orus   = generate_positions(D)
            H               = generate_channels(devices, orus, D)
            z               = assign_service_classes(D)
            lam, Tmax, Pmax = get_device_params(z, D)

            # Feasibility pre-check (discard infeasible realizations)
            # Necessary condition: rate above arrival rate from the nearest
            # O-RU at full power, on probe PRB (d mod K).
            ok = True
            for d in range(D):
                b_fa = int(np.argmin([np.linalg.norm(devices[d] - orus[b])
                                      for b in range(B)]))
                s_fa = Pmax[d] * H[b_fa, d, d % K] / N0
                if _rate(s_fa, z[d], d) <= lam[d]:
                    ok = False
                    break
            if not ok:
                continue

            # Proposed scheme; discard runs with starved devices
            hist, R_ao, M_ao, alp_ao = AO_RA(D, H, z, lam, Tmax, Pmax)
            if R_ao is None or np.any(R_ao == 0):
                continue

            # Benchmarks on the SAME scenario
            U_ba, R_ba, M_ba, alp_ba = baseline_scheme(
                D, H, z, lam, Tmax, Pmax, devices, orus)
            U_dr, R_dr, M_dr, alp_dr = dr_scheme(
                D, H, z, lam, Tmax, Pmax)

            u_ao.append(sum(W[z[d]] * R_ao[d] for d in range(D)))
            u_ba.append(U_ba)
            u_dr.append(U_dr)

            # Mean URLLC delay per scheme (finite values only)
            urllc = [d for d in range(D) if z[d] == 'URLLC']

            def _mean_delay(R_d, M_c, alp_c):
                vals = [compute_delay(d, 'URLLC', R_d, lam, M_c, alp_c)
                        for d in urllc]
                fin  = [v for v in vals if np.isfinite(v)]
                return np.mean(fin) if fin else np.nan

            d_ao.append(_mean_delay(R_ao, M_ao, alp_ao))
            d_ba.append(_mean_delay(R_ba, M_ba, alp_ba))
            d_dr.append(_mean_delay(R_dr, M_dr, alp_dr))

            vnf_ao.append([M_ao[sc] for sc in SERVICE_CLASSES])
            vnf_ba.append([M_ba[sc] for sc in SERVICE_CLASSES])
            vnf_dr.append([M_dr[sc] for sc in SERVICE_CLASSES])

            histories.append(hist)
            valid += 1
            if valid % 10 == 0:
                print(f"  {valid}/{n_runs} runs done (attempts: {attempt})")

        # Aggregate statistics per scheme
        def _store(key, u, d, vnf):
            r = results[key][D]
            r['utility_mean']     = float(np.nanmean(u))
            r['utility_std']      = float(np.nanstd(u))
            r['delay_urllc_mean'] = float(np.nanmean(d)) * 1e3   # -> ms
            r['delay_urllc_std']  = float(np.nanstd(d))  * 1e3
            arr = np.array(vnf)
            r['vnf_embb_mean']  = float(np.mean(arr[:, 0]))
            r['vnf_urllc_mean'] = float(np.mean(arr[:, 1]))
            r['vnf_mmtc_mean']  = float(np.mean(arr[:, 2]))

        _store('aora',     u_ao, d_ao, vnf_ao)
        _store('baseline', u_ba, d_ba, vnf_ba)
        _store('dr',       u_dr, d_dr, vnf_dr)

        # Mean convergence trace: pad shorter runs with their converged value
        max_len = max(len(h) for h in histories)
        padded  = [h + [h[-1]] * (max_len - len(h)) for h in histories]
        results['aora'][D]['history_mean'] = \
            np.mean(padded, axis=0).tolist()

        print(f"  AO-RA   : {results['aora'][D]['utility_mean']/1e6:.3f} Mbps")
        print(f"  DR      : {results['dr'][D]['utility_mean']/1e6:.3f} Mbps")
        print(f"  Baseline: {results['baseline'][D]['utility_mean']/1e6:.3f} Mbps")

    return results


# ============================================================================
# 10. FIGURE STYLING AND PLOTTING (Section VI figures)
# ============================================================================

def apply_ieee_style():
    """IEEE two-column journal style: serif fonts, STIX math, thin grid."""
    mpl.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'Times', 'Nimbus Roman',
                             'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size':        11,
        'axes.labelsize':   11,
        'axes.titlesize':   11,
        'legend.fontsize':  8.5,
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'axes.grid':        True,
        'grid.alpha':       0.35,
        'grid.linestyle':   ':',
        'grid.linewidth':   0.6,
        'axes.axisbelow':   True,
        'axes.linewidth':   0.9,
        'hatch.linewidth':  0.8,
        'savefig.dpi':      300,
        'savefig.bbox':     'tight',
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.fancybox':  False,
    })


BAR_STYLE = {
    'aora':     dict(facecolor='white', edgecolor='black', hatch='//',
                     linewidth=1.2, label='Proposed AO-RA'),
    'dr':       dict(facecolor='white', edgecolor='black', hatch='xx',
                     linewidth=1.2, label='DR Scheme'),
    'baseline': dict(facecolor='white', edgecolor='red',   hatch='..',
                     linewidth=1.2, label='Baseline Scheme'),
}

LINE_STYLE = {
    'aora':     dict(color='black', marker='o', linestyle='-',
                     markerfacecolor='white', markeredgecolor='black',
                     linewidth=1.4, markersize=6, label='Proposed AO-RA'),
    'dr':       dict(color='red',   marker='s', linestyle='--',
                     markerfacecolor='white', markeredgecolor='red',
                     linewidth=1.4, markersize=6, label='DR Scheme'),
    'baseline': dict(color='blue',  marker='^', linestyle='-.',
                     markerfacecolor='white', markeredgecolor='blue',
                     linewidth=1.4, markersize=6, label='Baseline Scheme'),
}

# Per-D curve styles
D_LINE_STYLE = [
    dict(color='black',     marker='o', linestyle='-'),
    dict(color='red',       marker='s', linestyle='--'),
    dict(color='blue',      marker='^', linestyle='-.'),
    dict(color='darkgreen', marker='D', linestyle=':'),
]


def _bar_labels(ax, bars, fmt='{:.0f}', fontsize=6.5, color='black',
                rotation=0, pad=0.5):
    """Annotate each bar with its value."""
    for rect in bars:
        h = rect.get_height()
        if not np.isfinite(h):
            continue
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, pad), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=fontsize, color=color, rotation=rotation)


def _save(fig, name):
    """Save a figure as PDF (submission) and PNG (preview)."""
    fig.savefig(os.path.join(RESULTS_DIR, name + '.pdf'))
    fig.savefig(os.path.join(RESULTS_DIR, name + '.png'))
    print(f'Saved -> {RESULTS_DIR}/{name}.pdf / .png')


def plot_system_model(devices, orus, z, show=False):
    """Deployment snapshot: device drops per class + O-RU positions."""
    colors  = {'eMBB': 'blue', 'URLLC': 'red',  'mMTC': 'green'}
    markers = {'eMBB': 'o',    'URLLC': 's',     'mMTC': '^'}
    fig, ax = plt.subplots(figsize=(7, 7))
    for sc in SERVICE_CLASSES:
        idx = [d for d in range(len(z)) if z[d] == sc]
        ax.scatter(devices[idx, 0], devices[idx, 1],
                   c=colors[sc], marker=markers[sc], label=sc, s=60)
    ax.scatter(orus[:, 0], orus[:, 1],
               c='black', marker='D', s=120, zorder=5, label='O-RU')
    circle = plt.Circle((0, 0), CELL_RADIUS, fill=False,
                        linestyle='--', color='gray')
    ax.add_patch(circle)
    ax.set_xlim(-CELL_RADIUS * 1.1, CELL_RADIUS * 1.1)
    ax.set_ylim(-CELL_RADIUS * 1.1, CELL_RADIUS * 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('O-RAN IIoT System Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'system_model')
    if show:
        plt.show()
    plt.close(fig)


def analyze_schemes(D, seed=42):
    """
    Run AO-RA, DR, and Baseline on one fixed scenario (fixed seed) and
    return the final allocation of each scheme (association x, rates R,
    per-device PRB counts) for the O-RU load comparison (Fig. 5).
    """
    np.random.seed(seed)
    devices, orus   = generate_positions(D)
    H               = generate_channels(devices, orus, D)
    z               = assign_service_classes(D)
    lam, Tmax, Pmax = get_device_params(z, D)
    R_min_dyn       = compute_dynamic_rmin(D, H, z, Pmax, silent=True)
    out = {}

    eta = np.array([
        max(R_min_dyn[z[d]], lam[d] + 1.0 / T_MAX[z[d]])
        for d in range(D)
    ])

    # --- Proposed AO-RA ---
    x = np.zeros((B, D), dtype=int)
    for d in range(D):
        x[int(np.argmax([np.mean(H[b, d, :]) for b in range(B)])), d] = 1
    h0, m0 = np.zeros(D), np.zeros(B)
    for _ in range(10):
        p, a, R_d, h0, m0 = lagrangian_kkt(D, B, K, H, x, z, lam, Tmax, Pmax, eta, h0, m0)
        M_c, alp_c = compute_vnf(z, R_d, lam, D)
        x = greedy_association(D, H, z, Pmax)
    p, a, R_d, h0, m0 = lagrangian_kkt(D, B, K, H, x, z, lam, Tmax, Pmax, eta, h0, m0)
    M_c, alp_c = compute_vnf(z, R_d, lam, D)
    rows = []
    for d in range(D):
        b_d   = int(np.argmax(x[:, d]))
        n_prb = sum(1 for k in range(K) if a[b_d, d, k] > 0.5)
        rows.append({'device': d, 'class': z[d], 'O-RU': b_d,
                     'n_PRBs': n_prb, 'rate_Mbps': R_d[d] / 1e6})
    out['aora'] = {'df': pd.DataFrame(rows), 'x': x.copy(), 'R': R_d.copy()}

    # --- DR benchmark (static best-channel association) ---
    x_dr = np.zeros((B, D), dtype=int)
    for d in range(D):
        x_dr[int(np.argmax([np.mean(H[b, d, :]) for b in range(B)])), d] = 1
    eta_dr = eta.copy()
    h0_dr, m0_dr = np.zeros(D), np.zeros(B)
    best_U_dr, best_R_dr, best_a_dr, prev_U_dr = -np.inf, None, None, None
    for _ in range(I_MAX):
        p_dr, a_dr, R_dr, h0_dr, m0_dr = lagrangian_kkt(
            D, B, K, H, x_dr, z, lam, Tmax, Pmax, eta_dr, h0_dr, m0_dr)
        U_dr = sum(W[z[d]] * R_dr[d] for d in range(D))
        if U_dr > best_U_dr:
            best_U_dr, best_R_dr, best_a_dr = U_dr, R_dr.copy(), a_dr.copy()
        if prev_U_dr is not None and abs(U_dr - prev_U_dr) / max(abs(U_dr), 1.0) < EPS_CONV:
            break
        prev_U_dr = U_dr
    rows_dr = []
    for d in range(D):
        b_d   = int(np.argmax(x_dr[:, d]))
        n_prb = sum(1 for k in range(K) if best_a_dr[b_d, d, k] > 0.5)
        rows_dr.append({'device': d, 'class': z[d], 'O-RU': b_d,
                        'n_PRBs': n_prb, 'rate_Mbps': best_R_dr[d] / 1e6})
    out['dr'] = {'df': pd.DataFrame(rows_dr), 'x': x_dr.copy(), 'R': best_R_dr.copy()}

    # --- Baseline benchmark (nearest O-RU, random single PRB) ---
    x_ba = np.zeros((B, D), dtype=int)
    for d in range(D):
        dists = [np.linalg.norm(devices[d] - orus[b]) for b in range(B)]
        x_ba[int(np.argmin(dists)), d] = 1
    a_ba = np.zeros((B, D, K))
    for b in range(B):
        srv  = [d for d in range(D) if x_ba[b, d] == 1]
        pool = list(range(K))
        np.random.shuffle(pool)
        for idx, d in enumerate(srv):
            if idx < K:
                a_ba[b, d, pool[idx]] = 1
    p_ba = np.zeros((D, K))
    for d in range(D):
        b_d = int(np.argmax(x_ba[:, d]))
        for k in range(K):
            if a_ba[b_d, d, k] > 0.5:
                p_ba[d, k] = Pmax[d]
    R_ba = compute_rates(D, B, K, H, x_ba, z, p_ba, a_ba, Pmax)
    rows_ba = []
    for d in range(D):
        b_d   = int(np.argmax(x_ba[:, d]))
        n_prb = sum(1 for k in range(K) if a_ba[b_d, d, k] > 0.5)
        rows_ba.append({'device': d, 'class': z[d], 'O-RU': b_d,
                        'n_PRBs': n_prb, 'rate_Mbps': R_ba[d] / 1e6})
    out['baseline'] = {'df': pd.DataFrame(rows_ba), 'x': x_ba.copy(), 'R': R_ba.copy()}
    return out, devices, orus, z


def compute_delay_vs_lambda(D_list_delay=(9, 18, 27), n_runs=100):
    """
    Sweep the URLLC mean arrival rate and measure the mean URLLC
    end-to-end delay, Eq. (14), for several device counts D (Fig. 3).
    """
    lambda_vals = np.linspace(0.1e6, 1.62e6, 10)
    delays_per_D = {}

    for D_fixed in D_list_delay:
        delays_ao = []
        for lam_val in lambda_vals:
            orig = LAMBDA['URLLC']
            LAMBDA['URLLC'] = lam_val
            u_ao_d = []
            for run in range(n_runs):
                np.random.seed(run * 100)
                devices, orus = generate_positions(D_fixed)
                H   = generate_channels(devices, orus, D_fixed)
                z   = assign_service_classes(D_fixed)
                lam, Tmax, Pmax = get_device_params(z, D_fixed)
                _, R_ao, M_ao, alp_ao = AO_RA(D_fixed, H, z, lam, Tmax, Pmax)
                urllc = [d for d in range(D_fixed) if z[d] == 'URLLC']
                if R_ao is not None:
                    vals = [compute_delay(d, 'URLLC', R_ao, lam, M_ao, alp_ao)
                            for d in urllc]
                    fin  = [v for v in vals if np.isfinite(v)]
                    if fin:
                        u_ao_d.append(np.mean(fin) * 1e3)
            delays_ao.append(np.nanmean(u_ao_d) if u_ao_d else np.nan)
            LAMBDA['URLLC'] = orig
        delays_per_D[D_fixed] = delays_ao

    lam_mbps = [l / 1e6 for l in lambda_vals]
    return lam_mbps, delays_per_D


def compute_utility_vs_power(D_fixed=18, n_runs=20):
    """
    Sweep the eMBB/URLLC maximum transmit power and measure the weighted
    sum data rate of AO-RA, DR, and Baseline (Fig. 4).
    """
    p_dbm_vals = np.arange(20, 36, 3)
    orig_embb  = P_MAX['eMBB']
    orig_urllc = P_MAX['URLLC']
    u_ao, u_dr, u_ba = [], [], []

    for p_dbm in p_dbm_vals:
        p_w            = 10 ** (p_dbm / 10) * 1e-3
        P_MAX['eMBB']  = p_w
        P_MAX['URLLC'] = p_w
        vals_ao, vals_dr, vals_ba = [], [], []
        for run in range(n_runs):
            np.random.seed(run * 100)
            devices, orus   = generate_positions(D_fixed)
            H               = generate_channels(devices, orus, D_fixed)
            z               = assign_service_classes(D_fixed)
            lam, Tmax, Pmax = get_device_params(z, D_fixed)
            _, R_ao, _, _ = AO_RA(D_fixed, H, z, lam, Tmax, Pmax)
            if R_ao is None or np.any(R_ao == 0):
                continue
            vals_ao.append(sum(W[z[d]] * R_ao[d] for d in range(D_fixed)))
            U_ba, _, _, _ = baseline_scheme(D_fixed, H, z, lam, Tmax, Pmax, devices, orus)
            U_dr, _, _, _ = dr_scheme(D_fixed, H, z, lam, Tmax, Pmax)
            vals_ba.append(U_ba)
            vals_dr.append(U_dr)
        u_ao.append(np.nanmean(vals_ao) / 1e6 if vals_ao else np.nan)
        u_dr.append(np.nanmean(vals_dr) / 1e6 if vals_dr else np.nan)
        u_ba.append(np.nanmean(vals_ba) / 1e6 if vals_ba else np.nan)

    P_MAX['eMBB']  = orig_embb
    P_MAX['URLLC'] = orig_urllc
    return p_dbm_vals, u_ao, u_dr, u_ba


def plot_fig1_convergence(results, D_list, n_show=15, show=False):
    """Fig. 1: mean AO-RA convergence trace, one curve per D."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, D in enumerate(D_list):
        h = list(results['aora'][D]['history_mean'])
        if len(h) < n_show:
            h = h + [h[-1]] * (n_show - len(h))   # pad with converged value
        h = h[:n_show]
        ax.plot(range(1, len(h) + 1), [u / 1e6 for u in h],
                markerfacecolor='white', markersize=6, linewidth=1.4,
                label=f'$D = {D}$', **D_LINE_STYLE[i])
    ax.set_xlabel('Number of Outer Iterations')
    ax.set_ylabel('Weighted Sum Data Rate (Mbps)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(40, 92)
    ax.legend(loc='center right', ncol=1)
    fig.tight_layout()
    _save(fig, 'fig1_convergence')
    if show:
        plt.show()
    plt.close(fig)


def plot_fig2_utility_vs_D(results, D_list, show=False):
    """Fig. 2: weighted sum data rate vs. D for the three schemes."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in ['aora', 'dr', 'baseline']:
        means = [results[s][D]['utility_mean'] / 1e6 for D in D_list]
        stds  = [results[s][D]['utility_std']  / 1e6 for D in D_list]
        ax.errorbar(D_list, means, yerr=stds, capsize=4, capthick=1.1,
                    elinewidth=1.1, **LINE_STYLE[s])
    ax.set_xlabel('Number of IIoT Devices ($D$)')
    ax.set_ylabel('Weighted Sum Data Rate (Mbps)')
    ax.set_xticks(D_list)
    ax.legend(loc='lower right')
    fig.tight_layout()
    _save(fig, 'fig2_utility_vs_D')
    if show:
        plt.show()
    plt.close(fig)


def plot_fig3_delay(lam_mbps, delays_per_D, D_list_delay, zoom=True, show=False):
    """
    Fig. 3: mean URLLC end-to-end delay vs. arrival rate, with a zoom
    inset on the low-delay region where the curves overlap.
    """
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, D in enumerate(D_list_delay):
        ax.plot(lam_mbps, delays_per_D[D],
                markerfacecolor='white', markersize=6, linewidth=1.4,
                label=f'AO-RA, $D = {D}$', **D_LINE_STYLE[i])
    ax.axhline(y=T_MAX['URLLC'] * 1e3, color='red', linestyle='--',
               linewidth=1.4, label=r'$T^{\max}_{\mathrm{URLLC}}$ = 1 ms')
    ax.set_xlabel(r'Mean Arrival Rate $\lambda$ (Mbps)')
    ax.set_ylabel('Mean End-to-End Delay (ms)')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='center left')

    if zoom:
        # Inset placed in the empty upper-right region of the axes
        axins = ax.inset_axes([0.55, 0.35, 0.42, 0.45])
        for i, D in enumerate(D_list_delay):
            axins.plot(lam_mbps, delays_per_D[D],
                       markerfacecolor='white', markersize=4,
                       linewidth=1.2, **D_LINE_STYLE[i])
        all_vals = [v for D in D_list_delay for v in delays_per_D[D]
                    if np.isfinite(v)]
        pad = (max(all_vals) - min(all_vals)) * 0.15
        axins.set_xlim(min(lam_mbps), max(lam_mbps))
        axins.set_ylim(min(all_vals) - pad, max(all_vals) + pad)
        axins.tick_params(labelsize=7)
        axins.grid(True, alpha=0.3, linestyle=':')
        ax.indicate_inset_zoom(axins, edgecolor='gray', linewidth=0.9)

    fig.tight_layout()
    _save(fig, 'fig3_delay_vs_lambda')
    if show:
        plt.show()
    plt.close(fig)


def plot_fig4_power(p_dbm_vals, u_ao, u_dr, u_ba, show=False):
    """Fig. 4: weighted sum data rate vs. maximum transmit power."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, vals in zip(['aora', 'dr', 'baseline'], [u_ao, u_dr, u_ba]):
        style = {k: v for k, v in LINE_STYLE[key].items()
                 if k not in ('markerfacecolor', 'markersize')}
        ax.plot(p_dbm_vals, vals, markerfacecolor='white', markersize=6, **style)
    ax.axvline(x=33, color='red', linestyle='--', linewidth=1.4,
               label=r'Default $P^{\max}_{\mathrm{eMBB/URLLC}}$ = 33 dBm')
    ax.set_xlabel(r'Maximum Transmit Power $P^{\max}$ (dBm)')
    ax.set_ylabel('Weighted Sum Data Rate (Mbps)')
    ax.legend(loc='center left')
    fig.tight_layout()
    _save(fig, 'fig4_utility_vs_power')
    if show:
        plt.show()
    plt.close(fig)


def plot_fig5_oru_load(D_fixed=27, seed=42, label_bars=True, show=False):
    """Fig. 5: devices served (left) and fronthaul load (right) per O-RU."""
    apply_ieee_style()
    res, devices, orus, z = analyze_schemes(D_fixed, seed)
    x_pos = np.arange(B)
    width = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, key in enumerate(['aora', 'dr', 'baseline']):
        df    = res[key]['df']
        x_mat = res[key]['x']
        R_vec = res[key]['R']

        n_devs = [df[df['O-RU'] == b].shape[0] for b in range(B)]
        bars0 = axes[0].bar(x_pos + i * width, n_devs, width, **BAR_STYLE[key])

        loads = [sum(R_vec[d] for d in range(D_fixed)
                     if x_mat[b, d] == 1) / 1e6 for b in range(B)]
        bars1 = axes[1].bar(x_pos + i * width, loads, width, **BAR_STYLE[key])

        if label_bars:
            _bar_labels(axes[0], bars0, fmt='{:.0f}', fontsize=7)
            _bar_labels(axes[1], bars1, fmt='{:.1f}', fontsize=6,
                        rotation=0, pad=1.5)

    axes[0].axhline(y=D_fixed / B, color='red', linestyle='--',
                    linewidth=1.4, label=f'Fair share = {D_fixed/B:.1f}')
    axes[0].set_xticks(x_pos + width)
    axes[0].set_xticklabels([f'O-RU {b}' for b in range(B)], fontsize=8)
    axes[0].set_xlabel('O-RU Index')
    axes[0].set_ylabel('Number of Devices Served')
    axes[0].set_ylim(0, max(D_fixed / B, 7) + 1.5)
    axes[0].grid(axis='y'); axes[0].grid(False, axis='x')
    axes[0].legend(loc='upper right', fontsize=8)

    axes[1].axhline(y=C_MAX_B / 1e6, color='red', linestyle='--',
                    linewidth=1.4,
                    label=r'$C_b^{\max}$ = ' + f'{C_MAX_B/1e6:.0f} Mbps')
    axes[1].set_xticks(x_pos + width)
    axes[1].set_xticklabels([f'O-RU {b}' for b in range(B)], fontsize=8)
    axes[1].set_xlabel('O-RU Index')
    axes[1].set_ylabel('Aggregate Fronthaul Rate (Mbps)')
    axes[1].set_ylim(0, C_MAX_B / 1e6 * 1.45)
    axes[1].grid(axis='y'); axes[1].grid(False, axis='x')
    axes[1].legend(loc='upper right', ncol=2, fontsize=8)

    fig.tight_layout()
    _save(fig, 'fig5_oru_load')
    if show:
        plt.show()
    plt.close(fig)
