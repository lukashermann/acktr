CUDA_VISIBLE_DEVICES=1 python main.py --env-id Jaco-v1 --kl-desired 0.004 --lr-vf 0.001 --seed 1 --max-timesteps 100000000 --timesteps-per-batch 8000 --use-pixels True
