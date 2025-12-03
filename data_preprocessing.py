import torch
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from utils import util
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


def compute_bg_risk(bg_values):
    """Compute individual BG risk scores."""
    f = 1.509 * ((np.log(bg_values)) ** 1.084 - 5.381)
    risk = 10 * (f ** 2)
    return risk


def label_hazards_bg_risk(df, window_size=12, lbgi_thresh=5, hbgi_thresh=9):
    """
    Label time-series BG data as:
      0 - Normal
      1 - Hypoglycemia (LBGI high)
      2 - Hyperglycemia (HBGI high)
    Based on Kovatchev BG risk indices.

    Parameters:
        df: DataFrame with 'bg' column
        window_size: number of samples per window (e.g. 12 for 5-min data = 1 hour)
        lbgi_thresh: threshold for LBGI
        hbgi_thresh: threshold for HBGI
    """
    bg = df['bg'].values
    n = len(bg)

    # Compute risk scores
    risk = compute_bg_risk(bg)

    # Split low and high risks: (ln(BG))1.084−5.381=0 -> BG=e4.722≈112.5
    rl = np.where(bg < 112.5, risk, 0)
    rh = np.where(bg > 112.5, risk, 0)

    # Compute moving averages
    lbgi = pd.Series(rl).rolling(window=window_size, min_periods=1).mean()
    hbgi = pd.Series(rh).rolling(window=window_size, min_periods=1).mean()

    # Compute gradients to detect increasing risk
    d_lbgi = np.gradient(lbgi)
    d_hbgi = np.gradient(hbgi)

    # Labeling logic
    labels = np.zeros(n, dtype=int)
    labels[(lbgi > lbgi_thresh) & (d_lbgi > 0)] = 1
    labels[(hbgi > hbgi_thresh) & (d_hbgi > 0)] = 2

    # Add to dataframe
    df['LBGI'] = lbgi
    df['HBGI'] = hbgi
    df['hazard_label'] = labels

    return df


class SimglucoseData(Dataset):
    def __init__(self, data_paths, target_patients, window_size=150, step=6, norm=False,
                 thresholds_path='logs/results/thresholds/simglucose_hazard'):
        """
        raw_data_all: dict[int or str, pd.DataFrame]
            Each DataFrame must contain columns:
            ['BG', 'rate', 'IOB', 'meal_carbs', 'exercise_magnitude', 'faults_label']
        window_size: int
            Number of time steps per sample (e.g., 12 for 1 hour if 5-min data)
        step: int
            Sliding step between windows
        """
        self.window_size = window_size
        self.step = step
        self.data_paths = data_paths
        self.patient_list = target_patients
        self.norm = norm
        self.thresholds_path = thresholds_path

        self.samples = []
        self.slide_data = []
        self.slide_faults = []
        self.slide_hazards = []
        self.slide_meal = []
        self.slide_exercise = []

        for path in self.data_paths:
            prep_data = self.preprocess_data(path)
            self.samples.extend(prep_data)
            for p_data in prep_data:
                for sample in p_data['slide_samples']:
                    self.slide_data.append(np.array(sample['data']))
                    self.slide_faults.append(sample['faults'])
                    self.slide_hazards.append(sample['hazards'])
                    self.slide_meal.append(np.array(sample['covar_meal']))
                    self.slide_exercise.append(np.array(sample['covar_exercise']))

        self.slide_data = torch.tensor(np.array(self.slide_data), dtype=torch.float32)
        self.slide_faults = torch.tensor(np.array(self.slide_faults), dtype=torch.float32)
        self.slide_hazards = torch.tensor(np.array(self.slide_hazards), dtype=torch.float32)
        self.slide_meal = torch.tensor(np.array(self.slide_meal), dtype=torch.float32)
        self.slide_exercise = torch.tensor(np.array(self.slide_exercise), dtype=torch.float32)

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        return self.slide_data[idx], self.slide_faults[idx], self.slide_hazards[idx], self.slide_meal[idx], self.slide_exercise[idx]

    def preprocess_data(self, data_path):
        """
        Preprocess the simulation data from pymgipsim test bed for model training or analysis.

        Notes
        -----
        - The function assumes 5-minute sampling intervals in the state trajectories.
        - All numeric values are rounded to 3 decimal places for consistency.
        """
        # Construct full file paths relative to base directory
        state_path = os.path.join(data_path, "model_state_results.xlsx")
        insulin_path = os.path.join(data_path, "insulin_input.csv")
        context_path = os.path.join(data_path, "simulation_settings.json")
        iob_path = os.path.join(data_path, "iob.csv")

        # Load insulin delivery data, simulation states, and contextual metadata
        insulin_input = pd.read_csv(insulin_path)
        iob = pd.read_csv(iob_path)
        sheets = pd.read_excel(state_path, sheet_name=None, index_col=0)
        with open(context_path, "r") as f:
            payload = json.load(f)

        print(f"Loaded data from {data_path}, preprocessing data for {self.patient_list}...")

        raw_data_all = []
        demographic_info = payload['patient']['demographic_info']
        demo_factors = list(demographic_info.keys())

        # Process each patient (each Excel sheet represents one simulation)
        for i, (name, df) in enumerate(sheets.items()):
            # Skip patient not in the target dataset
            if name not in self.patient_list:
                continue

            # --- key features: Historical glucose data, insulin administration, active insulin on board ---
            df['rate'] = insulin_input[str(i)].round(3)  # U/h
            df['bg'] = util.concentration_mmolL_to_mgdL(df['IG (mmol/L)']).round(3)  # mg/dL
            df['IOB'] = iob[str(i)].round(3)

            # --- First-order temporal derivatives ---
            df['detBG'] = df['bg'].diff()
            df['detrate'] = df['rate'].diff()
            df['detIOB'] = df['IOB'].diff()

            # --- Contextual signals: meals and exercise ---
            carb_events = util.extract_carb_events(payload, i)
            exercise_events = util.extract_exercise_events(payload, i)
            df["meal_carbs"] = 0.0
            df["exercise_magnitude"] = 0.0
            util.context_to_column(carb_events, df, "meal_carbs")
            util.context_to_column(exercise_events, df, "exercise_magnitude")

            # --- Static demographic/physiological features ---
            profile_dict = {
                key: round(demographic_info[key][i], 3)
                for key in demo_factors
                if (
                        isinstance(demographic_info[key], list)  # must be a list
                        and len(demographic_info[key]) == len(sheets.items())  # length matches number of patients
                        and demographic_info[key][i] is not None  # value for this patient is not None
                )
            }
            profile = pd.Series(profile_dict, name=f"Patient_{i}")

            # --- Fault labels ---
            df['faults_label'] = (df['faults_label'].fillna(0) != 0).astype(int)

            # --- Hazards labels ---
            # Add new columns:  'LBGI', 'HBGI', 'hazard_label'
            df = label_hazards_bg_risk(df, window_size=12)

            # Learned thresholds
            th_csv = pd.read_csv(f'{self.thresholds_path}/learned_thresholds_{name}.csv')
            thresholds = th_csv['learned_real']

            covar_meal = df[df["meal_carbs"] != 0.0]["meal_carbs"]
            covar_exercise = df[df["exercise_magnitude"] != 0.0]["exercise_magnitude"]

            # --- Aggregate raw data for data-driven methods---
            if self.norm:
                target = df[['bg', 'IOB', 'rate']]
                target_scaler = StandardScaler().fit(target)
                target_scaled = target_scaler.transform(target)
                target_scaled_df = pd.DataFrame(target_scaled, columns=["bg", "IOB", "rate"]).round(3)
                raw_data = {
                    'data': target_scaled_df,
                    'faults': np.array(df['faults_label']),
                    'hazards': np.array(df['hazard_label']),
                    'covar_meal': df["meal_carbs"],
                    'covar_exercise': df["exercise_magnitude"],
                    'target_scaler': target_scaler,
                    'static_covar': profile,
                    'patient_id': name
                }
                # Store per-patient processed data

            else:
                # --- Aggregate raw data for rule-based methods---
                data_df = df[['bg', 'rate', 'IOB', 'detBG', 'detrate', 'detIOB']]
                raw_data = {
                    'data': data_df,
                    'faults': np.array(df['faults_label']),
                    'hazards': np.array(df['hazard_label']),
                    'covar_meal': df["meal_carbs"],
                    'covar_exercise': df["exercise_magnitude"],
                    'static_covar': profile,
                    'thresholds': thresholds,
                    'patient_id': name
                }

            slide_sample = self.slide_window(raw_data)
            raw_data['slide_samples'] = slide_sample

            raw_data_all.append(raw_data)

        return raw_data_all

    def slide_window(self, ts_dict):
        data = ts_dict['data'].copy()
        abnormal = ts_dict['faults']
        hazard = ts_dict['hazards']
        meal = ts_dict['covar_meal']
        exercise = ts_dict['covar_exercise']

        samples = []
        for start in range(0, len(data) - self.window_size + 1, self.step):
            end = start + self.window_size
            window_data = data.iloc[start:end].copy()
            window_faults = abnormal[start:end]
            window_hazards = hazard[start:end]
            window_meal = meal[start:end]
            window_exercise = exercise[start:end]

            sample = {
                'data': window_data,
                'faults': window_faults,
                'hazards': window_hazards,
                'covar_meal': window_meal,
                'covar_exercise': window_exercise
            }
            samples.append(sample)

        return samples



if __name__ == "__main__":
    # base directory
    sim_data = r"datasets/simglucose/Simulation_OpenAPS_testing_all_faults"

    patient_list = [f"Patient_{i}" for i in range(16, 20)]

    # preprocessed_data = preprocess_pymgipsim_data(sim_data, patient_list, norm=False)
    #
    # data_samples = []
    # for p_data in preprocessed_data:
    #     sample = slide_window(p_data, window_size=150, stride=6)
    #     data_samples.extend(sample)

