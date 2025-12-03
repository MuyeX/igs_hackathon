import argparse
import json
import csv
import os
import time
import random
import logging
import warnings
from datetime import datetime

import numpy as np
import torch
import psutil

from data_preprocessing import SimglucoseData
from models.baseline_anomaly_detection import ForecastingAnomaly
from evaluation import sample_evaluator, overall_evaluator
from models.baseline_specification import detect_hazards  # not used in hyperopt, kept for completeness
from utils import util

# -------------------------
# Global setup for Apple Silicon (MPS) and reproducibility
# -------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
torch.set_default_dtype(torch.float32)
warnings.filterwarnings("ignore")
random.seed(42)
process = psutil.Process(os.getpid())
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# -------------------------
# Device selection
# -------------------------
def select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# -------------------------
# Helper casting to float32 for Darts TimeSeries lists
# -------------------------
def to_float32_timeseries(series_list, covar_list=None):
    series_list = [s.astype(np.float32) for s in series_list]
    if covar_list is not None:
        covar_list = [c.astype(np.float32) for c in covar_list]
    return series_list, covar_list

# -------------------------
# Robust metric extraction for model selection
# -------------------------
def extract_score(overall_metrics):
    """
    Try common keys for F1; if missing, attempt to compute F1 from confusion counts.
    As a fallback, try precision/recall harmonic mean; last resort: accuracy if present.
    Returns a float score; higher is better.
    """
    if not isinstance(overall_metrics, dict):
        return float("-inf")
    # Common F1 keys
    for key in ["F1", "f1", "F1_score", "f1_score"]:
        if key in overall_metrics and overall_metrics[key] is not None:
            try:
                return float(overall_metrics[key])
            except Exception:
                pass
    # Confusion-based F1
    tp = overall_metrics.get("TP", None)
    fp = overall_metrics.get("FP", None)
    fn = overall_metrics.get("FN", None)
    if tp is not None and fp is not None and fn is not None:
        tp, fp, fn = float(tp), float(fp), float(fn)
        denom = (2 * tp + fp + fn)
        if denom > 0:
            return (2 * tp) / denom
    # Precision/Recall harmonic mean
    prec = overall_metrics.get("precision", None)
    rec = overall_metrics.get("recall", None)
    if prec is not None and rec is not None:
        try:
            prec = float(prec)
            rec = float(rec)
            if (prec + rec) > 0:
                return 2 * prec * rec / (prec + rec)
        except Exception:
            pass
    # Accuracy fallback
    acc = overall_metrics.get("accuracy", None)
    if acc is not None:
        try:
            return float(acc)
        except Exception:
            pass
    # If no metric found
    return float("-inf")

# -------------------------
# Training and validation evaluation for a single configuration
# -------------------------
def train_and_eval_config(
    method,
    input_chunk_length,
    training_length,
    epochs,
    train_data,
    valid_data,
    window_size,
    stride,
    log_dir
):
    """
    Train ForecastingAnomaly with given hyperparameters and evaluate on validation patients.
    Returns:
        result = {
            "score": float,
            "overall_metrics_per_patient": list of dicts,
            "sample_metrics_per_patient": list of dicts,
            "model_save_path": str or None,
            "error": str or None
        }
    """
    result = {
        "score": float("-inf"),
        "overall_metrics_per_patient": [],
        "sample_metrics_per_patient": [],
        "model_save_path": None,
        "error": None
    }

    # Basic validity check
    if training_length <= input_chunk_length:
        result["error"] = f"Invalid config: training_length ({training_length}) must be > input_chunk_length ({input_chunk_length})."
        return result

    try:
        # Convert training/validation to Darts TimeSeries and cast to float32
        train_series, train_covar = util.to_darts_timeseries_list(train_data.samples)
        valid_series, valid_covar = util.to_darts_timeseries_list(valid_data.samples)

        train_series, train_covar = to_float32_timeseries(train_series, train_covar)
        valid_series, valid_covar = to_float32_timeseries(valid_series, valid_covar)

        # Initialize forecaster
        forecaster = ForecastingAnomaly(
            model=method,
            scorer="difference",
            input_chunk_length=input_chunk_length,
            training_length=training_length,
            epochs=epochs
        )

        # Fit the forecasting model
        forecaster.model.fit(
            series=train_series,
            past_covariates=train_covar,
            val_series=valid_series,
            val_past_covariates=valid_covar
        )

        # Save trained model checkpoint for this config
        model_name = f"{method}_icl{input_chunk_length}_tl{training_length}_ep{epochs}.pt"
        model_save_path = os.path.join(log_dir, model_name)
        try:
            forecaster.model.save(model_save_path)
            result["model_save_path"] = model_save_path
        except Exception as e_save:
            logging.warning(f"Failed to save model for config {model_name}: {e_save}")

        # Evaluate on validation patients by running anomaly detection on their sliding windows
        for p_data in valid_data.samples:
            window_predicts = []
            patient_profile = p_data.get("static_covar", None)

            for sample in p_data["slide_samples"]:
                # Prepare input for this window
                test_series, test_covar = util.to_darts_timeseries_list([sample], patient_profile)
                test_series, test_covar = to_float32_timeseries(test_series, test_covar)

                # Inference and binary anomaly sequence
                abnormal_ts_list = forecaster.anomaly_detection(test_series, test_covar)

                # Convert to numpy vector of length window_size
                pred_vec = abnormal_ts_list[0].to_dataframe().reindex(range(window_size), fill_value=0).to_numpy().flatten()
                window_predicts.append(pred_vec)

            # Stitch windows back to continuous patient timeline
            stitched_pred = util.stitch_predictions(
                window_predicts,
                window_size,
                stride,
                len(p_data["faults"]),
                mode="max"
            )

            # Evaluate metrics
            overall_metrics = overall_evaluator(
                stitched_pred,
                p_data["faults"],
                p_data["hazards"],
                tolerance=12  # you can make this a CLI arg if needed
            )

            # Also compute sample-level metrics (optional for model selection; useful for logging)
            sample_metrics = sample_evaluator(
                p_data["slide_samples"],
                window_predicts,
                {
                    "latencies": [],
                    "ram_usages": [],
                    "gpu_peaks": [],
                    "input_tokens": [],
                    "output_tokens": []
                },
                tolerance=12
            )

            result["overall_metrics_per_patient"].append(overall_metrics)
            result["sample_metrics_per_patient"].append(sample_metrics)

        # Aggregate macro average score across patients
        scores = [extract_score(m) for m in result["overall_metrics_per_patient"]]
        scores = [s for s in scores if np.isfinite(s)]
        if len(scores) > 0:
            result["score"] = float(np.mean(scores))
        else:
            result["score"] = float("-inf")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result

# -------------------------
# Main hyperparameter optimization routine
# -------------------------
def main(args):
    device = select_device()

    # Logging setup
    if not args.log_tag:
        args.log_tag = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"hyperopt_{args.method}_{args.log_tag}"
    log_dir = os.path.join(args.log_dir, "hyperopt", run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = util.save_path(log_dir, run_name, type="log")
    util.setup_logging(log_file)

    logging.info("========== Hyperopt Configuration ==========")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
    logging.info(f"Selected device: {device}")
    logging.info("============================================")

    # Normalization policy
    norm = True
    if args.method in ["baseline_rule", "baseline_LLM"]:
        norm = False

    # Patient splits (train/valid as in your original code)
    if args.testbed != "simglucose":
        logging.error("Testbed is not in the valid set: ['simglucose']")
        return

    patient_list = [f"Patient_{i}" for i in range(20)]
    patients_train = patient_list[:16]
    patients_valid = patient_list[16:20]

    # Load datasets
    logging.info("Loading training and validation datasets...")
    try:
        train_data = SimglucoseData(
            args.data_path, patients_train, window_size=args.window_size, step=args.stride, norm=norm
        )
        valid_data = SimglucoseData(
            args.data_path, patients_valid, window_size=args.window_size, step=args.stride, norm=norm
        )
    except Exception as e_data:
        logging.exception(f"Failed to load datasets: {e_data}")
        return

    # Build search space
    input_chunk_grid = args.input_chunks
    training_length_grid = args.training_lengths
    epoch_grid = args.epochs_grid

    logging.info(f"Search grid sizes: icl={len(input_chunk_grid)}, tl={len(training_length_grid)}, ep={len(epoch_grid)}")

    # CSV to record all trials
    trials_csv = os.path.join(log_dir, "hyperopt_trials.csv")
    with open(trials_csv, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "method", "input_chunk_length", "training_length", "epochs",
            "score", "model_path", "error"
        ])

    best = {
        "score": float("-inf"),
        "config": None,
        "model_path": None,
        "overall_metrics_per_patient": []
    }

    # Iterate through the grid
    total_trials = 0
    start_time = time.time()

    for icl in input_chunk_grid:
        for tl in training_length_grid:
            # Skip invalid combos early
            if tl <= icl:
                logging.warning(f"Skipping invalid combo: training_length ({tl}) <= input_chunk_length ({icl})")
                # Record trial as invalid
                with open(trials_csv, "a", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow([args.method, icl, tl, None, None, None, "invalid_combo"])
                continue

            for ep in epoch_grid:
                total_trials += 1
                logging.info(f"Trial {total_trials}: method={args.method}, icl={icl}, tl={tl}, ep={ep}")

                result = train_and_eval_config(
                    method=args.method,
                    input_chunk_length=icl,
                    training_length=tl,
                    epochs=ep,
                    train_data=train_data,
                    valid_data=valid_data,
                    window_size=args.window_size,
                    stride=args.stride,
                    log_dir=log_dir
                )

                # Record trial
                with open(trials_csv, "a", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow([
                        args.method, icl, tl, ep,
                        result["score"],
                        result["model_save_path"],
                        result["error"]
                    ])

                # Update best if improved
                if result["error"] is None and result["score"] > best["score"]:
                    best["score"] = result["score"]
                    best["config"] = {
                        "method": args.method,
                        "input_chunk_length": icl,
                        "training_length": tl,
                        "epochs": ep,
                        "window_size": args.window_size,
                        "stride": args.stride
                    }
                    best["model_path"] = result["model_save_path"]
                    best["overall_metrics_per_patient"] = result["overall_metrics_per_patient"]

    elapsed = time.time() - start_time
    logging.info(f"Hyperopt completed in {elapsed:.2f}s over {total_trials} trials.")

    # Save best hyperparameters
    best_json = os.path.join(log_dir, "best_hyperparams.json")
    with open(best_json, "w") as f_json:
        json.dump({
            "best_score": best["score"],
            "best_config": best["config"],
            "best_model_path": best["model_path"],
            "elapsed_seconds": elapsed
        }, f_json, indent=2)
    logging.info(f"Best hyperparameters saved to {best_json}")

    # Log evaluation results for best parameters
    if best["config"] is None:
        logging.error("No valid configuration found. See trials CSV for errors.")
        print("No valid configuration found. Check logs and hyperopt_trials.csv.")
        return

    logging.info("========== Best Configuration ==========")
    for k, v in best["config"].items():
        logging.info(f"{k}: {v}")
    logging.info(f"Best score (macro F1 or fallback): {best['score']:.6f}")
    logging.info(f"Best model path: {best['model_path']}")
    logging.info("========================================")

    # Print evaluation summary for best parameters (per patient overall metrics)
    summary_txt = os.path.join(log_dir, "best_overall_metrics.txt")
    with open(summary_txt, "w") as f_sum:
        for i, m in enumerate(best["overall_metrics_per_patient"]):
            f_sum.write(f"Patient_{16 + i} metrics:\n")  # validation patients are 16..19
            for k, v in m.items():
                f_sum.write(f"  {k}: {v}\n")
            f_sum.write("\n")
    logging.info(f"Best evaluation metrics per validation patient saved to {summary_txt}")

    # Also print to console
    print("Best hyperparameters:")
    print(json.dumps(best["config"], indent=2))
    print(f"Best score: {best['score']:.6f}")
    print("Per-patient overall metrics (validation):")
    for idx, m in enumerate(best["overall_metrics_per_patient"]):
        print(f"- Patient_{16 + idx}:")
        for k, v in m.items():
            print(f"  {k}: {v}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Anomaly Detection (simglucose)")
    parser.add_argument("--data_path", type=str, nargs="+",
                        default=["datasets/simglucose/Simulation_OpenAPS_testing_all_faults"],
                        help="Paths to the input attacked data file.")
    parser.add_argument("--log_dir", type=str, default="logs/", help="Directory to save logs and results.")
    parser.add_argument("--log_tag", type=str, default=None, help="Experiment identifier")
    parser.add_argument("--testbed", type=str, default="simglucose", choices=["simglucose"], help="Dataset/testbed")
    parser.add_argument("--method", type=str, default="LSTM", choices=["RNN", "LSTM", "GRU"], help="Forecasting model")
    parser.add_argument("--window_size", type=int, default=150, help="Number of samples per window")
    parser.add_argument("--stride", type=int, default=6, help="Step size between consecutive windows")

    # Search grids (customize as needed)
    parser.add_argument("--input_chunks", type=int, nargs="+", default=[24, 30, 60, 90],
                        help="Grid for input_chunk_length")
    parser.add_argument("--training_lengths", type=int, nargs="+", default=[30, 36, 72, 120],
                        help="Grid for training_length (must be > input_chunk_length)")
    parser.add_argument("--epochs_grid", type=int, nargs="+", default=[5, 10, 20],
                        help="Grid for number of epochs")

    args = parser.parse_args()
    main(args)