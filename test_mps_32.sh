CUDA_VISIBLE_DEVICES=0 python main_MPS.py \
  --mode test \
  --method LSTM \
  --model_path ./logs/train/RNN_20251203_0630/model.pt      \
  --data_path ./datasets/simglucose/Simulation_OpenAPS_testing_all_faults/ \
  --log_dir logs/