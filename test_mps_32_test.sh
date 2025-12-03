CUDA_VISIBLE_DEVICES=0 python main_MPS_test.py \
  --mode test \
  --method LSTM \
  --model_path ./logs/train/LSTM_20251203_0758/model.pt      \
  --data_path ./datasets/simglucose/test-data/ \
  --log_dir logs/ \
  --window_size 277