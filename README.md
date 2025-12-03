# 🧠 Anomaly Detection for Automatic Insulin Delivery (AID) System

This repository implements a **modular anomaly detection pipeline** for detecting anomaly events in **closed-loop AID systems** using simulated AID data with faults injection from HealthLoopQA testbed: https://github.com/schlevik/py-mgipsim/tree/anomaly_detection/pymgipsim.  

It supports both **data-driven** and **rule-based** anomaly detection methods, with end-to-end functionality for:
- Data preprocessing  
- Model training and evaluation  
- Real-time hazard detection and reaction time analysis  
- Performance logging and reporting

## ⚙️ Installation

### 1. Clone the repository
### 2. Create a virtual environment

### 3. Install dependencies

    pip install -r requirements.txt


### 4. Install Darts 
Refer to their installation guide: https://unit8co.github.io/darts/quickstart/00-quickstart.html



## 🚀 Usage

### Train a Model

    python main.py \
      --mode train \
      --method LSTM \
      --testbed simglucose \
      --data_path datasets/simglucose/Simulation_OpenAPS_testing_all_faults \
      --log_dir logs/

You can incorporate new forecasting-based or filtering-based anomaly detection models implemneted by Darts: 
https://github.com/unit8co/darts

### Test a Model

    python main.py \
      --mode test \
      --method LSTM \
      --model_path logs/train/LSTM_20251113_1240/model.pt \
      --data_path datasets/simglucose/Simulation_OpenAPS_testing_all_faults \
      --log_dir logs/

### Logging & Outputs
Logs and models are automatically saved under:

    logs/{train|test}/{method}_{timestamp}/

Each run stores:

model.pt — Trained model checkpoint

run.log — Full training/testing log

Metric summaries:

* Affinity/Range/Point-wise detection F1, precision, recall

* Hazard reaction time distributions

* Latency and memory footprint per sample