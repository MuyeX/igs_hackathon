CUDA_VISIBLE_DEVICES=1 python main_MPS.py \
  --mode train \
  --method RNN \
  --testbed simglucose \
  --data_path ./datasets/simglucose/Simulation_OpenAPS_training_normal_scenario1 ./datasets/simglucose/Simulation_OpenAPS_training_normal_scenario2 \
  --log_dir logs/