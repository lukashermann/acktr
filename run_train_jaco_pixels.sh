<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=1 python main.py --env-id Jaco-v1 --kl-desired 0.004 --lr-vf 0.001 --seed 1 --max-timesteps 100000000 --timesteps-per-batch 8000 --use-pixels True
=======
CUDA_VISIBLE_DEVICES=0 python main.py --env-id JacoPixel-v1 --kl-desired 0.002 --lr-vf 0.0001 --cold-lr-vf 0.0001 --seed 1 --max-timesteps 100000000 --timesteps-per-batch 8000 --use-pixels True --is-rgb True --is-depth True
>>>>>>> resume_training_test
