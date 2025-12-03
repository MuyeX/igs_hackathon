import argparse
from data_preprocessing import SimglucoseData
from evaluation import sample_evaluator, overall_evaluator
from models.baseline_anomaly_detection import ForecastingAnomaly
from utils import util
import random
import torch
import psutil, os, time
from datetime import datetime
from models.baseline_specification import detect_hazards
import warnings
import logging

warnings.filterwarnings("ignore")
process = psutil.Process(os.getpid())
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

random.seed(42)

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''Set up logging '''
    if not args.log_tag:
        args.log_tag = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{args.method}_{args.log_tag}"
    log_dir = os.path.join(args.log_dir, args.mode, run_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = util.save_path(log_dir, run_name, type='log')
    util.setup_logging(log_file)

    # ---- Log all arguments ----
    logging.info("========== Configuration ==========")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
    logging.info("===================================")

    logging.info("Starting anomaly detection pipeline...")

    '''Load and preprocess the dataset'''
    logging.info("Reading data from: %s", args.data_path)

    # rule-based methods no normalization, data-driven methods need normalization
    norm = True
    if args.method in ['baseline_rule', 'baseline_LLM']:
        norm = False

    if args.testbed == 'simglucose' and args.mode == 'train':
        patient_list = [f"Patient_{i}" for i in range(20)]
        patients_train = patient_list[:16]
        patients_valid = patient_list[16:20]

    elif args.testbed == 'simglucose' and args.mode == 'test':
        patient_list = [f"Patient_{i}" for i in range(10)]
        patients_test = patient_list[0:10]
    else:
        print("Testbed is not in the valid set: ['simglucose']")
        return


    # Train the model
    if args.mode == 'train':
        train_data = SimglucoseData(args.data_path, patients_train, window_size=args.window_size, step=args.stride,
                                    norm=norm)
        valid_data = SimglucoseData(args.data_path, patients_valid, window_size=args.window_size, step=args.stride,
                                    norm=norm)
        
        print(train_data)

        if args.method in ['RNN', 'LSTM', 'Transformer', 'RandomForest', 'VARIMA', 'KalmanFilter',
                           'MovingAverageFilter']:
            # Transfer data into TimeSeries objects
            train_series, train_covar = util.to_darts_timeseries_list(train_data.samples)
            valid_series, valid_covar = util.to_darts_timeseries_list(valid_data.samples)
            forecaster = ForecastingAnomaly(model=args.method, scorer='difference', input_chunk_length=277,
                                          training_length=336, epochs=10)
            forecaster.model.fit(
                series=train_series,
                past_covariates=train_covar,
                val_series=valid_series,
                val_past_covariates=valid_covar
            )

            model_save_path = os.path.join(log_dir, 'model.pt')
            forecaster.model.save(model_save_path)
            logging.info("Model saved to %s", model_save_path)


    # Test the model
    elif args.mode == 'test':
        # evaluating
        logging.info(f"Running anomaly detection using {args.method}...")

        test_data = SimglucoseData(args.data_path, patients_test, window_size=args.window_size, step=args.stride,
                                   norm=norm)
        logging.info(f"Total {len(test_data.samples)} patient and each patient have "
                     f"{len(test_data.samples[0]['slide_samples'])} samples for testing.")

        for p_data in test_data.samples:
            running_results = {
                'latencies': [],
                'ram_usages': [],
                'gpu_peaks': [],
                'input_tokens': [],
                'output_tokens': []
            }
            window_predicts = []
            patient_profile = p_data['static_covar']

            for sample in p_data['slide_samples']:
                # ---- memory before ----
                mem_before = process.memory_info().rss / (1024 ** 2) # MB

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                # ---- inference start ----
                if args.method == 'baseline_rule':
                    start_time = time.perf_counter()
                    pred = detect_hazards(sample['data'], p_data['thresholds'])
                    end_time = time.perf_counter()
                elif args.method in ['RNN', 'LSTM', 'GRU', 'Transformer']:
                    anomaly_detector = ForecastingAnomaly(model=args.method,
                                                    input_chunk_length=277,
                                                    training_length=336)
                    pretrained_model = anomaly_detector.model.load(args.model_path)
                    anomaly_detector.update_model(pretrained_model)

                    test_series, test_covar = util.to_darts_timeseries_list([sample], patient_profile)

                    start_time = time.perf_counter()
                    pred = anomaly_detector.anomaly_detection(test_series, test_covar)
                    end_time = time.perf_counter()

                    pred = pred[0].to_dataframe().reindex(range(150), fill_value=0).to_numpy().flatten()

                else:
                    print(f'Method {args.method} not implemented')
                # ---- inference end ----

                # ---- memory after ----
                mem_after = process.memory_info().rss / (1024 ** 2)
                running_results['ram_usages'].append(mem_after - mem_before)

                # ---- record latency ----
                running_results['latencies'].append(end_time - start_time)

                # ---- record GPU memory ----
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                    running_results['gpu_peaks'].append(gpu_peak)

                window_predicts.append(pred)

            # Stitch sliding window samples back to continuous time series
            # mode: "max" | "mean" | "vote"
            stitch_mode = 'max'
            stitched_pred = util.stitch_predictions(window_predicts, args.window_size, args.stride, len(p_data['faults']), mode="max")
            logging.info(f"Stitching sliding window samples back to continuous time series using mode: {stitch_mode}")

            # Evaluate detection and hazard reaction time performance at real-time anomaly detection level
            overall_metrics = overall_evaluator(stitched_pred, p_data['faults'], p_data['hazards'], tolerance=12)
            # Evaluate detection and running performance in each slide data sample
            sample_metrics = sample_evaluator(p_data['slide_samples'], window_predicts, running_results, tolerance=12)

            util.log_all_metrics(p_data['patient_id'], overall_metrics, prefix="Overall Evaluation Metrics")
            util.log_all_metrics(p_data['patient_id'], sample_metrics, prefix="Sample-level Evaluation Metrics")

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection for Artificial Pancreas Systems")

    # Add arguments for user inputs
    parser.add_argument('--data_path', type=str, nargs='+', default=["datasets/simglucose/Simulation_OpenAPS_testing_all_faults"]   # D:\GlucoseSimulatedDatasets\APS-faultydata\simulationCollection_newmodel
                        , help="Paths to the input attacked data file.")
    parser.add_argument('--model_path', type=str, default='logs/train/RNN_20251113_1904/model.pt'
                        , help="Paths to trained model.")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                        help="Anomaly detection method to use.")
    parser.add_argument('--testbed', type=str, default='simglucose', choices=['simglucose'],
                        help="Anomaly detection method to use.")
    parser.add_argument('--method', type=str, default='baseline_rule', choices=['RNN', 'LSTM', 'GRU', 'baseline_rule'],
                        help="Anomaly detection method to use.")
    parser.add_argument('--log_dir', type=str, default="logs/", help="Directory to save log files.")
    parser.add_argument('--log_tag', type=str, default=None, help="Experiments identifier")
    parser.add_argument('--tolerance_window', type=int, default=48,
                        help="predicted attacks can earlier or later than true attacks happen in a tolerance window")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size")
    parser.add_argument('--bg_target', type=int, default=140,
                        help="Target blood glucose level")
    parser.add_argument('--window_size', type=int, default=150,
                        help="Number of samples per window")
    parser.add_argument('--stride', type=int, default=6,
                        help="Step size between consecutive windows")

    args = parser.parse_args()
    main(args)
