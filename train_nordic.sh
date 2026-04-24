.venv/bin/python train.py --data_path /media/pc/HD1/aibirder_model_data/combined_nordic.parquet --taxonomy taxonomy.csv --checkpoint_dir checkpoints/nordic/ --run_name run1 --no_amp --patience 50 --lr 2e-4 --jitter --no_yearly --min_obs_per_species 10 --num_epochs 300 --batch_size 1000 --model_scale 2.0 --data_path /media/pc/HD1/aibirder_model_data/combined_nordic.parquet 

