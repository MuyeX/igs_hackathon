"""
stl_threshold_learning.py

Joint learning of STL thresholds (β1..β11 for IOB, and β21 for BG) across multiple simulation files.

Input: a Python list `simulations` where each element is a dict:
    {"data": pd.DataFrame with columns ['bg','IOB','rate'],
     "abnormal": 1D array-like of 0/1 values (same length as data)}

Output:
 - learned_thresholds.csv (columns: idx, initial, learned)
 - tmee_loss_evolution.png
 - histograms of mu for a couple of thresholds (displayed)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from data_preprocessing import SimglucoseData

# -----------------------------
# TMEE loss and helpers
# -----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tmee_per_sample_and_grad_r(r):
    """
    Given r (numpy array), returns (loss_sum, dL_dr array)
    L(r) = exp(-r) + r - sigmoid(2r)
    dL/dr = -exp(-r) + 1 - 2*s*(1-s) with s = sigmoid(2r)
    """
    s = sigmoid(2.0 * r)
    loss_vec = np.exp(-r) + r - s
    loss = np.sum(loss_vec)
    dL_dr = -np.exp(-r) + 1.0 - 2.0 * s * (1.0 - s)
    return loss, dL_dr

# -----------------------------
# Compute deltas (approx derivatives)
# -----------------------------
def compute_deltas(df):
    df = df.copy().reset_index(drop=True)
    # forward differences (current - previous); first row diff = 0
    df['delBg'] = df['bg'].diff().fillna(0.0)
    df['delIOB'] = df['IOB'].diff().fillna(0.0)
    df['delInsulinRate'] = df['rate'].diff().fillna(0.0)
    return df

# -----------------------------
# Predicate (antecedent) evaluation
# -----------------------------
def evaluate_antecedents(df, bgTarget=140):
    """
    Evaluate antecedent boolean masks for STL rules 1..12 (Table I).
    Returns dict mapping 'row_N' -> boolean mask (numpy boolean array)
    """
    df2 = df.copy().reset_index(drop=True)
    bg = df2['bg'].values
    delBg = df2['delBg'].values
    iob = df2['IOB'].values
    delIob = df2['delIOB'].values

    eps = 1e-6
    masks = {}

    # Rule 1: (BG>BGT & BG'>0 & IOB'<0)
    masks['row_1'] = (bg > bgTarget) & (delBg > eps) & (delIob < -eps)

    # Rule 2: (BG>BGT & BG'>0 & IOB'=0)
    masks['row_2'] = (bg > bgTarget) & (delBg > eps) & np.isclose(delIob, 0.0, atol=eps)

    # Rule 3: (BG>BGT & BG'<0 & IOB'>0)
    masks['row_3'] = (bg > bgTarget) & (delBg < -eps) & (delIob > eps)

    # Rule 4: (BG>BGT & BG'<0 & IOB'<0)
    masks['row_4'] = (bg > bgTarget) & (delBg < -eps) & (delIob < -eps)

    # Rule 5: (BG>BGT & BG'<0 & IOB'=0)
    masks['row_5'] = (bg > bgTarget) & (delBg < -eps) & np.isclose(delIob, 0.0, atol=eps)

    # Rule 6: (BG<BGT & BG'<0 & IOB'>0)
    masks['row_6'] = (bg < bgTarget) & (delBg < -eps) & (delIob > eps)

    # Rule 7: (BG<BGT & BG'<0 & IOB'<0)
    masks['row_7'] = (bg < bgTarget) & (delBg < -eps) & (delIob < -eps)

    # Rule 8: (BG<BGT & BG'<0 & IOB'=0)
    masks['row_8'] = (bg < bgTarget) & (delBg < -eps) & np.isclose(delIob, 0.0, atol=eps)

    # Rule 9: (BG>BGT)
    masks['row_9'] = (bg > bgTarget)

    # Rule 10: (BG<β21) – antecedent is always true, β21 applied later
    masks['row_10'] = (bg < bgTarget)

    # Rule 11: (BG>BGT & BG'>0 & IOB'<=0)
    masks['row_11'] = (bg > bgTarget) & (delBg > eps) & (delIob <= eps)

    # Rule 12: (BG<BGT & BG'<0 & IOB'>=0)
    masks['row_12'] = (bg < bgTarget) & (delBg < -eps) & (delIob >= -eps)

    return masks

# -----------------------------
# Map rules to threshold indices and quantities
# -----------------------------

def rule_to_index_mapping():
    """
    Map each rule (1–12) to (beta_index, direction, quantity)
    β1–β11: IOB thresholds, β21: BG threshold
    """
    mapping = {
        'row_1':  (0,  'lt', 'IOB'),  # β1, H2
        'row_2':  (1,  'lt', 'IOB'),  # β2, H2
        'row_3':  (2,  'lt', 'IOB'),  # β3, H2
        'row_4':  (3,  'lt', 'IOB'),  # β4, H2
        'row_5':  (4,  'lt', 'IOB'),  # β5, H2
        'row_6':  (5,  'gt', 'IOB'),  # β6, H1
        'row_7':  (6,  'gt', 'IOB'),  # β7, H1
        'row_8':  (7,  'gt', 'IOB'),  # β8, H1
        'row_9':  (8,  'lt', 'IOB'),  # β9, H2
        'row_10': (11, 'lt', 'BG'),   # β21 (note index 11 corresponds to β21), H1
        'row_11': (9,  'lt', 'IOB'),  # β10, H2
        'row_12': (10, 'gt', 'IOB'),  # β11, H1
    }
    return mapping
# -----------------------------
# Collect μ values for hazardous antecedents across a single simulation
# -----------------------------
def collect_mu_for_simulation(sim_dict, mapping, bgTarget=140):
    """
    Collect predicate variable (μ) samples for each STL rule in Table I,
    evaluated only at hazardous times (fault_period == 1).

    sim_dict: {"data": df, "fault_period": array-like of 0/1}
    mapping:  rule_to_index_mapping()
    Returns:  dict idx -> numpy array of μ values corresponding to that rule
    """
    df = sim_dict['data']
    fault_arr = np.asarray(sim_dict['hazards'], dtype=int)
    assert len(fault_arr) == len(df), "fault_period length must match dataframe length"

    # --- Step 1: compute derivatives ---
    df2 = compute_deltas(df)
    masks = evaluate_antecedents(df2, bgTarget=bgTarget)

    # --- Step 2: get base arrays ---
    bg = df2['bg'].values
    iob = df2['IOB'].values
    # hazard_mask = fault_arr == 1
    hazard_mask_1 = fault_arr == 1
    hazard_mask_2 = fault_arr == 2

    # --- Step 3: initialize storage ---
    max_idx = max(v[0] for v in mapping.values())
    mu_by_idx = {idx: [] for idx in range(max_idx + 1)}

    # --- Step 4: evaluate each rule’s antecedent ---
    for rule_key, antecedent_mask in masks.items():
        if rule_key not in mapping:
            continue
        idx, direction, quantity = mapping[rule_key]

        # antecedent ∧ hazard
        if idx in [0, 1, 2, 3, 4, 8, 9]:
            mask = antecedent_mask & hazard_mask_2
        else:
            mask = antecedent_mask & hazard_mask_1
        # mask = antecedent_mask & hazard_mask
        if not np.any(mask):
            continue

        # μ value is whichever variable appears in the rule’s inequality
        if quantity == 'IOB':
            mu_vals = iob[mask]
        elif quantity == 'BG':
            mu_vals = bg[mask]
        else:
            raise ValueError(f"Unknown quantity type: {quantity}")

        mu_by_idx[idx].extend(mu_vals.tolist())

    # --- Step 5: convert lists to numpy arrays ---
    for k in mu_by_idx.keys():
        mu_by_idx[k] = np.array(mu_by_idx[k], dtype=float)

    return mu_by_idx

# -----------------------------
# Aggregate across multiple simulations
# -----------------------------
def aggregate_mu_from_simulations(simulations, bgTarget=140):
    mapping = rule_to_index_mapping()
    total_mu = {}
    # initialize keys
    max_idx = max([v[0] for v in mapping.values()])
    for k in range(max_idx+1):
        total_mu[k] = []

    for sim in simulations:
        mu_sim = collect_mu_for_simulation(sim, mapping, bgTarget=bgTarget)
        for k, arr in mu_sim.items():
            if arr.size > 0:
                total_mu[k].extend(arr.tolist())

    # convert to numpy arrays
    for k in list(total_mu.keys()):
        total_mu[k] = np.array(total_mu[k], dtype=float)
    return total_mu, mapping

# -----------------------------
# Joint loss & gradient over all indices (beta vector)
# -----------------------------
def joint_loss_and_grad(beta_vec, mu_sets, direction_map, penalty_gamma=1e4):
    """
    beta_vec: numpy array length = number of betas
    mu_sets: dict idx->numpy array of mu samples
    direction_map: dict idx -> 'lt' or 'gt'
    Return: (loss_scalar, grad_vector)
    """
    beta_vec = np.asarray(beta_vec).astype(float)
    loss_total = 0.0
    grad = np.zeros_like(beta_vec)

    for idx, mu in mu_sets.items():
        if mu.size == 0:
            continue

        dirn = direction_map.get(idx, 'lt')
        if dirn == 'lt':
            # predicate is mu < beta -> r = beta - mu (we want r>0)
            r = np.clip(beta_vec[idx] - mu, -20, 20)  # vector
            loss, dL_dr = tmee_per_sample_and_grad_r(r)
            loss_total += loss
            # penalty for r <= 0
            neg = np.clip(-r, a_min=0.0, a_max=None)
            penalty = penalty_gamma * np.sum(neg**2)
            loss_total += penalty
            # derivative of penalty wrt r: 2*gamma*r for r < 0
            dpenalty_dr = np.where(r < 0, 2.0 * penalty_gamma * r, 0.0)
            dtotal_dr = dL_dr + dpenalty_dr
            # dr/dbeta = +1 => grad contribution is sum(dtotal_dr)
            grad[idx] += np.sum(dtotal_dr)
        else:
            # direction == 'gt': predicate mu > beta -> r = mu - beta
            r = np.clip(mu - beta_vec[idx], -20, 20)
            loss, dL_dr = tmee_per_sample_and_grad_r(r)
            loss_total += loss
            neg = np.clip(-r, a_min=0.0, a_max=None)
            penalty = penalty_gamma * np.sum(neg**2)
            loss_total += penalty
            dpenalty_dr = np.where(r < 0, 2.0 * penalty_gamma * r, 0.0)
            dtotal_dr = dL_dr + dpenalty_dr
            # dr/dbeta = -1 => grad contribution is -sum(dtotal_dr)
            grad[idx] -= np.sum(dtotal_dr)

    return float(loss_total), grad

# -----------------------------
# Main learning function across multiple simulations
# -----------------------------
def learn_thresholds_from_simulations(simulations,
                                      bgTarget=140,
                                      thresholds_len=12,
                                      beta_init=None,
                                      bounds=None,
                                      penalty_gamma=1e4,
                                      save_csv="learned_thresholds.csv",
                                      loss_plot="tmee_loss_evolution.png",
                                      verbose=True):
    """
    simulations: list of {"data": df, "fault_period": arr_like}
    thresholds_len: total number of betas (default 12: 11 IOB + 1 BG)
    beta_init: initial vector of length thresholds_len
    bounds: list of (low,high) pairs length thresholds_len. If None, use defaults:
            IOB betas (0..10) bounds [-5, 5]; last beta index (11 - BG) bounds [0, 500]
    """
    # 1) aggregate mu sets
    mu_sets, mapping = aggregate_mu_from_simulations(simulations, bgTarget=bgTarget)
    direction_map = {v[0]: v[1] for k, v in rule_to_index_mapping().items()}

    if verbose:
        print("Aggregated μ counts per idx:")
        for k in sorted(mu_sets.keys()):
            print(f"  idx {k}: {mu_sets[k].size}")

    # Normalize μ
    mu_stats = {}
    mu_scaled = {}

    for idx, mu in mu_sets.items():
        if len(mu) == 0:
            mu_mean, mu_std = 0.0, 1.0
            mu_scaled[idx] = mu
        else:
            mu_mean = np.mean(mu)
            mu_std = np.std(mu) + 1e-6
            mu_scaled[idx] = (mu - mu_mean) / mu_std
        mu_stats[idx] = (mu_mean, mu_std)

    # 2) init betas
    if beta_init is None:
        # init in normalized scale
        beta_init = np.zeros(len(mu_scaled))
    else:
        beta_init = np.asarray(beta_init, dtype=float)
        if beta_init.size != thresholds_len:
            raise ValueError("beta_init length mismatch")

    # 3) bounds
    if bounds is None:
        # default scaled bounds for all betas: +/- 3 std devs
        scaled_default = (-3.0, 3.0)
        bounds = [scaled_default] * thresholds_len
        # Low BG threshold range: [60, 80]
        bg_mean = mu_stats[11][0]
        bg_std = mu_stats[11][1]
        bounds[-1] = ((60 - bg_mean) / bg_std, (80 - bg_mean) / bg_std)
    if len(bounds) != thresholds_len:
        raise ValueError("bounds length mismatch")

    # Diagnostic: initial loss & grad
    init_loss, init_grad = joint_loss_and_grad(beta_init, mu_scaled, direction_map, penalty_gamma=penalty_gamma)
    grad_norm = np.linalg.norm(init_grad)
    if verbose:
        print(f"Initial loss (scaled-space): {init_loss:.6e}, grad norm: {grad_norm:.6e}")

    # If gradient huge or NaN, bail out with message
    if not np.isfinite(init_loss) or not np.isfinite(grad_norm) or grad_norm > 1e12:
        raise RuntimeError(
            "Initial loss/gradient numerically unstable. "
            "Check mu_sets, scaling, or reduce penalty_gamma. "
            f"init_loss={init_loss}, grad_norm={grad_norm}"
        )

    # record loss evolution
    loss_history = []
    x_history = []

    # define objective and jac
    def obj_and_jac(x):
        L, g = joint_loss_and_grad(x, mu_scaled, direction_map, penalty_gamma=penalty_gamma)
        return L, g

    # callback for scipy to record intermediate values
    def callback(xk):
        Lk, _ = joint_loss_and_grad(xk, mu_scaled, direction_map, penalty_gamma=penalty_gamma)
        loss_history.append(Lk)
        x_history.append(xk.copy())

    # run optimization
    if verbose:
        print("Starting optimization with L-BFGS-B...")

    res = minimize(fun=lambda x: obj_and_jac(x)[0],
                   x0=beta_init,
                   jac=lambda x: obj_and_jac(x)[1],
                   bounds=bounds,
                   method='L-BFGS-B',
                   callback=callback,
                   options={'ftol':1e-12, 'maxiter':5000})

    beta_learned = np.array([
        res.x[i] * mu_stats[i][1] + mu_stats[i][0]
        for i in range(len(mu_scaled))
    ])

    if verbose:
        print("Optimization finished. success:", res.success, "message:", res.message)
        print("Learned betas (12):", beta_learned[:thresholds_len])

    # Save CSV of initial vs learned
    rows = []
    for i in range(thresholds_len):
        learned_real = float(beta_learned[i])
        rows.append({
            'idx': i,
            'initial': mu_stats[i][0],
            'learned_real': learned_real,
            'direction': direction_map.get(i, '')
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(save_csv, index=False)
    if verbose:
        print(f"Saved learned thresholds to {save_csv}")

    # Plot loss evolution
    if len(loss_history) > 0:
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(len(loss_history)), loss_history, '-o')
        plt.xlabel('LBFGS iteration (callback calls)')
        plt.ylabel('TMEE total loss')
        plt.title('TMEE loss evolution')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_plot)
        if verbose:
            print(f"Saved loss evolution plot to {loss_plot}")
        plt.show()
    else:
        print("No intermediate loss history recorded (callback not invoked).")

    # Plot histograms for a couple of representative indices
    try:
        plotted = 0
        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        for idx in sorted(mu_sets.keys()):
            if mu_sets[idx].size == 0:
                continue
            ax = axes[plotted]
            ax.hist(mu_sets[idx], bins=30, alpha=0.6)
            ax.axvline(beta_init[idx], color='k', linestyle='--', label='init')
            ax.axvline(beta_learned[idx], color='r', linestyle='-', label='learned')
            ax.set_title(f'idx {idx} dir={direction_map.get(idx)} count={mu_sets[idx].size}')
            ax.legend()
            plotted += 1
            if plotted >= 2:
                break
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting histograms failed:", e)

    return beta_learned, res, mu_sets, mapping, df_out


def detect_hazards(df, learned_thresholds, bgTarget=140):
    """
    Detect abnormal time points based on STL rules (Table I) and learned thresholds.
    learned_thresholds: array-like of length 12 (β₁–β₁₁ for IOB, β₂₁ for BG)
    Returns: numpy array of 0/1 same length as df.
    """
    df2 = compute_deltas(df)
    masks = evaluate_antecedents(df2, bgTarget=bgTarget)
    bg, iob = df2['bg'].values, df2['IOB'].values

    # learned thresholds
    betas = np.asarray(learned_thresholds).flatten()
    assert len(betas) == 12, "Expected 12 thresholds (β₁–β₁₁ + β₂₁)"

    abnormal = np.zeros(len(df2), dtype=int)

    # --- H2 hazard rules (1–5, 9, 11)
    # (IOB < β)
    for rule, idx in [('row_1',0), ('row_2',1), ('row_3',2), ('row_4',3), ('row_5',4), ('row_9',8), ('row_11',9)]:
        mask = masks[rule] & (iob < betas[idx])
        abnormal[mask] = 1

    # --- H1 hazard rules (6–8, 10, 12)
    # (IOB > β) or (BG < β₂₁)
    for rule, idx in [('row_6',5), ('row_7',6), ('row_8',7), ('row_12',10)]:
        mask = masks[rule] & (iob > betas[idx])
        abnormal[mask] = 1

    # Rule 10: (BG < β₂₁)
    mask = (bg < betas[11])
    abnormal[mask] = 1

    return abnormal

# -----------------------------
# Training for personalized thresholds (if run as script)
# -----------------------------
if __name__ == "__main__":
    data_path = [r"../datasets/simglucose/Simulation_OpenAPS_training_naive_faults"]
    testbed = 'simglucose'
    thresholds_group = 'simglucose_hazard'
    save_dir = f"../logs/results/thresholds/{thresholds_group}"
    os.makedirs(save_dir, exist_ok=True)

    if testbed == 'simglucose':
        patient_list = [f"Patient_{i}" for i in range(20)]
    elif testbed == 'apstestbed':
        patient_list = ['patientA', 'patientB', 'patientC', 'patientD', 'patientE', 'patientF',
                    'patientG', 'patientH', 'patientI', 'patientJ']
    else:
        print("Please input valid testbed: simglucose or apstestbed")

    simulations = []
    for patient in patient_list:
        print("Processing patient:", patient)
        if testbed == 'simglucose':
            sim_data = SimglucoseData(data_path, [patient], norm=False)
            simulations = sim_data.samples

        learned_beta, res, mu_sets, mapping, df_out = learn_thresholds_from_simulations(
            simulations,
            bgTarget=140,
            thresholds_len=12,
            beta_init=None,
            bounds=None,
            penalty_gamma=100,
            save_csv=os.path.join(save_dir, f"learned_thresholds_{patient}.csv"),
            loss_plot=os.path.join(save_dir, f"tmee_loss_evolution_{patient}.png"),
            verbose=True
        )

        abnormal_list = detect_hazards(simulations[0]['data'], learned_beta)

    print("Done. Learned betas:", learned_beta)
