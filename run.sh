.venv/bin/python -u train.py --data_path /media/pc/HD1/aibirder_model_data/combined.parquet --model_scale 0.75 --no_amp --no_yearly --resume checkpoints/checkpoint_latest.pt --num_epochs 250 --patience 50

run2
.venv/bin/python -u train.py --data_path /media/pc/HD1/aibirder_model_data/combined.parquet --model_scale 1 --no_amp --no_yearly  --num_epochs 350 --patience 50 --jitter --lr_warmup 5 

