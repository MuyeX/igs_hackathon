from datetime import datetime
import json
import os
import logging
import numpy as np
import torch
from darts import TimeSeries
from tqdm import tqdm
import copy
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
import logging


def to_darts_timeseries_list(data_dicts, profile_info=None):
    ts_list = []
    covar_list = []
    for sample in data_dicts:
        df = sample['data']
        if profile_info is None:
            profile_info = sample['static_covar']
        ts = TimeSeries.from_dataframe(df, static_covariates=profile_info)
        ts_list.append(ts)

        covar_meal = TimeSeries.from_series(sample['covar_meal'])
        covar_exercise = TimeSeries.from_series(sample['covar_exercise'])
        covars = covar_meal.stack(covar_exercise)
        covar_list.append(covars)

    return ts_list, covar_list


def log_all_metrics(patient_id, all_metrics, prefix="Evaluation Metrics"):
    """
    Log all key-value pairs from all_metrics for a given patient.
    Automatically formats numbers and handles nested dicts.
    """
    lines = [f"{patient_id}'s {prefix}:"]
    for key, value in all_metrics.items():
        # format floats nicely
        if isinstance(value, (float, int)):
            lines.append(f"  {key}: {value:.4f}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"  {key}: {value}")
        elif isinstance(value, dict):
            lines.append(f"  {key}:")
            for subk, subv in value.items():
                if isinstance(subv, (float, int)):
                    lines.append(f"    {subk}: {subv:.4f}")
                else:
                    lines.append(f"    {subk}: {subv}")
        else:
            lines.append(f"  {key}: {value}")
    log_text = "\n".join(lines)
    logging.info("\n" + log_text + "\n")

def interval_to_binary(intervals, length=150):
    """Convert a list of [start, end] intervals into a binary array of given length."""
    binary_array = np.zeros(length, dtype=int)
    start = intervals[0]
    end = intervals[1]
    binary_array[start:end+1] = 1  # +1 because end is inclusive
    return binary_array

def stitch_predictions(predictions, window_size, stride, total_length, mode="max"):
    """
    Reconstruct a full-length prediction array from overlapping sliding windows.

    Args:
        predictions: list of np.ndarray, each of shape (window_size,)
        window_size: int, number of samples per window
        stride: int, step size between consecutive windows
        total_length: int, length of the full signal
        mode: "max" | "mean" | "vote"

    Returns:
        full_pred: np.ndarray of shape (total_length,)
    """
    full_pred = np.zeros(total_length)
    counts = np.zeros(total_length)

    for i, pred in enumerate(predictions):
        start = i * stride
        end = start + window_size
        if end > total_length:
            break  # skip incomplete tail window if any
        if mode == "max":
            full_pred[start:end] = np.maximum(full_pred[start:end], pred)
        else:
            full_pred[start:end] += pred
            counts[start:end] += 1

    if mode in ["mean", "vote"]:
        counts[counts == 0] = 1
        full_pred = full_pred / counts
        if mode == "vote":
            full_pred = (full_pred >= 0.5).astype(int)

    return full_pred.astype(int)


def context_to_column(context_dict, df: pd.DataFrame, col_name: str):
    for m in context_dict:
        idx = int(round(m["time"] / 5))
        if 0 <= idx < len(df):
            df.loc[idx, col_name] = m[col_name.split("_")[1]]


def format_time_info(mins: float) -> Tuple[int, str]:
    """
    Convert absolute minute index into (day_index, HH:MM).
    Day index starts at 1 and increases monotonically (no weekly reset).
    """
    mins = float(mins)
    day = int(mins // 1440) + 1
    time_of_day = mins % 1440
    hours = int(time_of_day // 60)
    minutes = int(time_of_day % 60)
    return day, f"{hours:02d}:{minutes:02d}"


def extract_carb_events(payload: Dict[str, Any], person_idx: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    if "meal_carb" in payload["inputs"]:
        mags = payload["inputs"]["meal_carb"]["magnitude"][person_idx]
        times = payload["inputs"]["meal_carb"]["start_time"][person_idx]
        meal_types = ["breakfast", "lunch", "dinner"]
        for idx, (carbs, t) in enumerate(zip(mags, times)):
            day, time_str = format_time_info(t)
            events.append({
                "time": t, "day": day, "time_str": time_str,
                "carbs": carbs, "meal_type": meal_types[idx % 3]
            })

    if "snack_carb" in payload["inputs"]:
        mags = payload["inputs"]["snack_carb"]["magnitude"][person_idx]
        times = payload["inputs"]["snack_carb"]["start_time"][person_idx]
        snack_types = ["morning_snack", "afternoon_snack"]
        for idx, (carbs, t) in enumerate(zip(mags, times)):
            day, time_str = format_time_info(t)
            events.append({
                "time": t, "day": day, "time_str": time_str,
                "carbs": carbs, "meal_type": snack_types[idx % 2]
            })

    events.sort(key=lambda x: x["time"])
    return events


def extract_exercise_events(payload: Dict[str, Any], person_idx: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for source, label in [("running_speed", "running"), ("cycling_power", "cycling")]:
        if source not in payload["inputs"]:
            continue
        magnitudes = payload["inputs"][source]["magnitude"][person_idx]
        start_times = payload["inputs"][source]["start_time"][person_idx]
        durations = payload["inputs"][source]["duration"][person_idx]
        for mag, t, d in zip(magnitudes, start_times, durations):
            if mag == 0:
                continue
            day, time_str = format_time_info(t)
            events.append({
                "time": t, "day": day, "time_str": time_str,
                "duration": d, "magnitude": mag, "exercise_type": label
            })
    events.sort(key=lambda x: x["time"])
    return events

def mUmin_to_Uhr(mUmin):
    return mUmin / 1000 * 60

def concentration_mmolL_to_mgdL(mmolL):
    return mmolL*18

def hazard_tags(csv_file_paths, hazard_column='hazard_flag'):
    hazard_flags = []

    for path in csv_file_paths:
        df = pd.read_csv(path)

        if hazard_column not in df.columns:
            raise ValueError(f"'{hazard_column}' column not found in file: {path}")

        hazard_flags.extend(df[hazard_column].astype(int).tolist())

    return hazard_flags


def to_timeseries(files_path, features, min_length=10, normal=True):
    # Index 'Unnamed: 0' used as time index
    columns = copy.deepcopy(features)
    columns.append('Unnamed: 0')
    ts_list = []
    data_points = 0
    for file_path in files_path:
        df = pd.read_csv(file_path)
        if normal:
            # Get non-faulty data
            if len(df[df['faultinjection'] == True]) != 0:
                # Find segments where fault injection occurs
                fault_starts = df.index[(df['faultinjection'].shift(1) == 0) & (df['faultinjection'] == 1)].tolist()
                fault_ends = df.index[(df['faultinjection'].shift(1) == 1) & (df['faultinjection'] == 0)].tolist()

                # Add beginning and end of dataframe to create complete segments
                before_fault = df.iloc[:fault_starts[0]][columns]
                if len(before_fault) > min_length:
                    series_before = TimeSeries.from_dataframe(before_fault, time_col="Unnamed: 0")
                    ts_list.append(series_before)
                    data_points += len(before_fault)

                if len(fault_ends) != 0:
                    after_fault = df.iloc[fault_ends[0]:][columns]
                    if len(after_fault) > min_length:
                        series_after = TimeSeries.from_dataframe(after_fault, time_col="Unnamed: 0")
                        ts_list.append(series_after)
                        data_points += len(after_fault)
            else:
                if len(df) > 10:
                    series = TimeSeries.from_dataframe(df[columns], time_col="Unnamed: 0")
                    ts_list.append(series)
                    data_points += len(series)
        else:
            if len(df) > 10:
                series = TimeSeries.from_dataframe(df[columns], time_col="Unnamed: 0")
                ts_list.append(series)
                data_points += len(series)

    # print("Total {} episodes".format(len(ts_list)))
    # print("Total {} points".format(data_points))
    return ts_list

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    # Add a filter to exclude "HTTP Request" logs
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda record: "HTTP Request" not in record.getMessage())


def save_path(log_dir, tag, type='log'):
    log_file = os.path.join(log_dir, f"{tag}.{type}")
    return log_file

def save_predictions(log_path, log_tag, predict_list, target_list):
    """Save prediction and target lists to a JSON file."""
    if not log_tag:
        log_tag = datetime.now().strftime("%Y%m%d_%H%M")
    predictions_file = os.path.join(log_path, f"{log_tag}_predictions.json")

    predict_list = np.array(predict_list).tolist()
    target_list = np.array(target_list).tolist()

    with open(predictions_file, 'w') as f:
        json.dump({
            'predict_list': predict_list,
            'target_list': target_list
        }, f, indent=4)


def save_LLM_record(log_path, results):
    with open(log_path, 'a', encoding='utf-8') as file:
        json.dump(results, file)
        print("LLM Data saved to {}!".format(log_path))


def binary_to_events(binary_seq, threshold=0.5):
    """
    Convert a binary sequence into a list of anomaly periods [start, end].
    """
    binary_seq = (binary_seq >= threshold).int()
    binary_seq = binary_seq.tolist() if isinstance(binary_seq, torch.Tensor) else binary_seq
    events = []
    start = None

    for i, value in enumerate(binary_seq):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            events.append((start, i))
            start = None

    if start is not None:
        if start == len(binary_seq) - 1:
            # avoid point anomaly error
            events.append((start-1, len(binary_seq)-1))
        else:
            events.append((start, len(binary_seq) - 1))

    # if len(events) == 0:
    #     events = [(0, 1)]

    return events