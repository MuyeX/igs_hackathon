import copy

import numpy as np
from utils.affiliation.metrics import pr_from_events
from utils.affiliation.generics import convert_vector_to_events
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.util import binary_to_events
import json
import os
import math
from tqdm import tqdm
from utils import util
import pandas as pd
from pathlib import Path
import random
import psutil
import time
from scipy import stats

random.seed(42)


def sample_evaluator(test_samples, predicts, running_results, tolerance=12):
    # Evaluate detection accuracy, temporal performance
    standard_scores = []
    range_scores = []
    affinity_scores = []

    for i, sample in enumerate(test_samples):
        preds = predicts[i]  # predicted hazard flags (array-like)
        trues = sample["faults"]  # ground truth faults flags (array-like)

        standard_f = f_scores(preds, trues)
        range_f = f_range(preds, trues, tolerance=tolerance)
        affinity_f = f_affinity(preds, trues)

        standard_scores.append(standard_f)
        range_scores.append(range_f)
        affinity_scores.append(affinity_f)

    # compute mean ± 95% CI
    mean_std, ci_std = mean_confidence_interval(standard_scores)
    mean_rng, ci_rng = mean_confidence_interval(range_scores)
    mean_aff, ci_aff = mean_confidence_interval(affinity_scores)

    # Evaluate computational efficiency, resource usage
    latencies = running_results['latencies']
    ram_usages = running_results['ram_usages']
    gpu_peaks = running_results['gpu_peaks']
    input_tokens = running_results['input_tokens']
    output_tokens = running_results['output_tokens']

    ram_usages = np.array(ram_usages)
    gpu_peaks = np.array(gpu_peaks) if gpu_peaks else np.array([0.0])
    input_tokens = np.array(input_tokens) if input_tokens else np.array([0.0])
    output_tokens = np.array(output_tokens) if output_tokens else np.array([0.0])

    peak_ram = np.max(ram_usages)
    peak_gpu = np.max(gpu_peaks)

    latency_summary = median_iqr_p95(latencies)
    input_tokens_summary = median_iqr_p95(input_tokens)
    output_tokens_summary = median_iqr_p95(output_tokens)

    return {
        'Standard F-score - Mean': mean_std,
        'Standard F-score - 95%CI': ci_std,
        'Range F-score - Mean': mean_rng,
        'Range F-score - 95%CI': ci_rng,
        'Affinity F-score - Mean': mean_aff,
        'Affinity F-score - 95%CI': ci_aff,
        'Inference Latency (s)': latency_summary,
        'Memory Footprint - Peak RAM (MB)': peak_ram,
        'Memory Footprint - Peak GPU (MB)': peak_gpu
    }


def overall_evaluator(all_preds, all_trues, all_hazards, skip_paths=None, type='forecasting',  tolerance=0):
    standard_f = f_scores(all_preds, all_trues)
    range_f = f_range(all_preds, all_trues, tolerance=tolerance)
    affinity_f = f_affinity(all_preds, all_trues)
    faults_hazards_results = compute_fault2hazard_metrics(all_preds, all_trues, all_hazards, time_step=5.0)
    all_hazards_results = compute_hazard_alert_metrics(all_preds, all_hazards, time_step=5.0)

    faults_hazards_summary = summarize_hazard_detection(faults_hazards_results)
    all_hazards_summary = summarize_hazard_detection(all_hazards_results)

    return {
        'standard_f': standard_f,
        'range_f': range_f,
        'affinity_f': affinity_f,
        'fault_to_hazard_reaction': faults_hazards_summary,
        'all_hazard_reaction': all_hazards_summary
    }


def compute_fault2hazard_metrics(prediction, attacks, hazards, time_step=5.0 , max_fault_gap=12*60):
    """
    Compute reaction times for hazard episodes (continuous hazard segments).

    Args:
        prediction (array-like): binary array (1=alert issued)
        attacks (array-like): binary array (1=fault active)
        hazards (array-like): binary array (1=hazard active)
        time_step (float): duration per sample (e.g., 1 min)
        max_fault_gap (float): maximum allowed time (unit: minute) between
                       fault and hazard start to consider association.
                       e.g. 12*60 for 1 day.

    Returns:
        time_to_hazard: list of float, time between last fault and hazard start
        reaction_time: list of float, time between first alert after fault and hazard start
    """
    prediction = np.asarray(prediction).astype(int)
    attacks = np.asarray(attacks).astype(int)
    hazards = np.asarray(hazards).astype(int)
    n = len(hazards)

    # --- Step 1: find continuous hazard windows (start & end indices)
    hazard_starts = []
    hazard_ends = []
    in_hazard = False
    for i in range(n):
        if hazards[i] != 0 and not in_hazard:
            hazard_starts.append(i)
            in_hazard = True
        elif hazards[i] == 0 and in_hazard:
            hazard_ends.append(i)
            in_hazard = False
    if in_hazard:  # hazard lasts until end of array
        hazard_ends.append(n)

    time_to_hazard = []
    reaction_time = []
    hazard_durations = []
    detected_flags = []
    last_hzd = 0

    # --- Step 2: compute for each hazard start ---
    for hz_start, hz_end in zip(hazard_starts, hazard_ends):
        # Find most recent fault before hazard start
        fault_indices = np.where(attacks[last_hzd:hz_start] == 1)[0]
        if len(fault_indices) == 0:
            continue  # skip if no preceding fault
        fault_idx = fault_indices[-1] + last_hzd

        # --- Compute time to hazard ---
        t_hazard = (hz_start - fault_idx) * time_step
        # skip if too far away
        if max_fault_gap is not None and t_hazard > max_fault_gap:
            continue

        time_to_hazard.append(t_hazard)

        # --- Compute hazard duration ---
        hazard_duration = (hz_end - hz_start) * time_step
        hazard_durations.append(hazard_duration)

        # --- Find first alert after fault but before hazard ---
        preds_between = np.where(prediction[fault_idx:hz_start] == 1)[0]
        if len(preds_between) == 0:
            reaction_time.append(0.0)
            detected_flags.append(0)
        else:
            first_alert_idx = fault_idx + preds_between[0]
            t_react = (hz_start - first_alert_idx) * time_step
            reaction_time.append(t_react)
            detected_flags.append(1)

        last_hzd = copy.copy(hz_start)

    return {
        "time_to_hazard": time_to_hazard,
        "hazard_durations": hazard_durations,
        "reaction_time": reaction_time,
        "detected_flags": detected_flags
    }


def compute_hazard_alert_metrics(prediction, hazards, time_step=5.0, pre_warning_window=3*60.0):
    """
    Compute alert performance for *all* hazard periods, not just fault-related ones.

    Args:
        prediction (array-like): Binary array, 1 = alert triggered
        hazards (array-like): Binary array, 1 = hazard period
        time_step (float): Duration per sample (e.g., 1 minute)
        pre_warning_window (float): Optional window (in same units as time_step)
            allowing alerts *slightly before* hazard start to count as correct.

    Returns:
        hazard_durations: list of hazard durations
        reaction_times: list of times between hazard start and first alert (0 if no alert)
        detected_flags: list of 0/1 indicating whether an alert was triggered for each hazard
    """
    prediction = np.asarray(prediction).astype(int)
    hazards = np.asarray(hazards).astype(int)
    n = len(hazards)

    hazard_durations = []
    reaction_times = []
    detected_flags = []

    # Identify start and end indices of each hazard period
    in_hazard = False
    start_idx = None
    for i in range(n):
        if hazards[i] != 0 and not in_hazard:
            start_idx = i
            in_hazard = True
        elif hazards[i] == 0 and in_hazard:
            end_idx = i
            in_hazard = False

            # Compute hazard duration
            hazard_duration = (end_idx - start_idx) * time_step
            hazard_durations.append(hazard_duration)

            # Look for alerts during hazard window (or slightly before)
            pre_start = max(0, start_idx - int(pre_warning_window / time_step))
            alert_indices = np.where(prediction[pre_start:end_idx] == 1)[0]

            if len(alert_indices) > 0:
                first_alert_idx = pre_start + alert_indices[0]
                reaction_time = (start_idx - first_alert_idx) * time_step
                reaction_times.append(reaction_time)
                detected_flags.append(1)
            else:
                reaction_times.append(0.0)
                detected_flags.append(0)

    # handle case if hazard ends at the very end
    if in_hazard:
        end_idx = n
        hazard_duration = (end_idx - start_idx) * time_step
        hazard_durations.append(hazard_duration)
        pre_start = max(0, start_idx - int(pre_warning_window / time_step))
        alert_indices = np.where(prediction[pre_start:end_idx] == 1)[0]
        if len(alert_indices) > 0:
            first_alert_idx = pre_start + alert_indices[0]
            reaction_time = max(0.0, (first_alert_idx - start_idx) * time_step)
            reaction_times.append(reaction_time)
            detected_flags.append(1)
        else:
            reaction_times.append(0.0)
            detected_flags.append(0)

    return {
        "hazard_durations": hazard_durations,
        "reaction_time": reaction_times,
        "detected_flags": detected_flags
    }


def f_scores(y_pred, y_true):
    # point-wise
    # y_pred & y_true: list of anomaly of every point [0, ..., 1]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    #     {
    #     "precision":precision,
    #     "recall": recall,
    #     "f1": f1
    # }
    return f1



def f_range(y_pred, y_true, tolerance=0):
    """
    Compute range-based F1 score with a tolerance window (in time steps).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_ranges = convert_vector_to_events(y_true)
    pred_ranges = convert_vector_to_events(y_pred)

    tp = 0
    matched_true = set()

    for pred_start, pred_end in pred_ranges:
        for i, (true_start, true_end) in enumerate(true_ranges):
            # Only count 1 TP for each ground truth
            if i in matched_true:
                continue

            # Expand the true range with tolerance
            # tol_start = max(0, true_end - tolerance)
            tol_start = true_start
            tol_end = min(len(y_true) - 1, true_end + tolerance)

            if pred_end >= tol_start and pred_start <= tol_end:
                tp += 1
                matched_true.add(i)
                break

    fp = len(pred_ranges) - tp
    fn = len(true_ranges) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1



def f_affinity(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Edge case 1: no true events and no predicted events ---
    if np.all(y_true == 0) and np.all(y_pred == 0):
        # Perfectly correct: no event predicted when no event exists
        return 1.0

    # --- Edge case 2: no true events, but model predicted something ---
    if np.all(y_true == 0) and np.any(y_pred == 1):
        # False alarms only → F1 = 0
        return 0.0

    # --- Edge case 3: true events exist, but no prediction ---
    if np.any(y_true == 1) and np.all(y_pred == 0):
        # Missed all events → F1 = 0
        return 0.0

    # --- Normal case: both have events ---
    try:
        true_ranges = convert_vector_to_events(y_true)
        pred_ranges = convert_vector_to_events(y_pred)

        pr_results = pr_from_events(pred_ranges, true_ranges, (0, len(y_true)))

        precision = pr_results['precision']
        recall = pr_results['recall']
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    except ValueError as e:
        # Fallback in case of unexpected empty input
        print(f"[Warning] f_affinity skipped due to: {e}")
        return 0.0

    return f1


def mean_confidence_interval(data, confidence=0.95):
    """Return mean and half-width of the CI."""
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]  # remove NaNs if any
    n = len(data)
    if n == 0:
        return np.nan, np.nan
    mean = np.mean(data)
    sem = stats.sem(data)  # standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, h


def median_iqr_p95(result_list):
    arr = np.array(result_list, dtype=float)
    median = np.median(arr)
    q1, q3 = np.percentile(arr, [25, 75])
    p95 = np.percentile(arr, 95)
    min_a = np.min(arr)
    max_a = np.max(arr)
    return {
        "median": median,
        'q1': q1,
        'q3': q3,
        'p95': p95,
        'min': min_a,
        'max': max_a
        }

def summarize_hazard_detection(results):
    """
    Compute detection statistics from hazard results.

    Args:
        results (dict): {
            "hazard_durations": list[float],
            "reaction_time": list[float],
            "detected_flags": list[int]
        }

    Returns:
        dict with detection_rate and reaction_time_summary
    """
    detected_flags = np.asarray(results["detected_flags"])
    reaction_times = np.asarray(results["reaction_time"], dtype=float)

    # --- 1. Detection rate ---
    detection_rate = 100.0 * detected_flags.mean() if len(detected_flags) > 0 else np.nan

    # --- 2. Reaction time summary (only detected cases) ---
    if detected_flags.sum() > 0:
        detected_rt = reaction_times[detected_flags == 1]
        rct_time_summary = median_iqr_p95(detected_rt)
    else:
        rct_time_summary = np.nan

    results_summary = {
        "detection_rate": detection_rate,
        "reaction_time (min)": rct_time_summary
    }

    return results_summary






