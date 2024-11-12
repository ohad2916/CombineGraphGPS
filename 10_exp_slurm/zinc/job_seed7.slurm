#!/bin/bash
#SBATCH --job-name=GPS+RWSE-zinc-seed7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --output=/home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/zinc/output_seed7.out
#SBATCH --error=/home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/zinc/error_seed7.err
#SBATCH --time=4300
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4

echo 'Starting job...'
/home/yandex/MLWG2024/michaelbest/anaconda3/etc/profile.d/conda.sh activate graphgps
echo 'Activated environment: $CONDA_PREFIX'

cd /home/yandex/MLWG2024/michaelbest/CombineGraphGPS
python main.py --cfg /config/GPS/zinc-GPS+RWSE.yaml wandb.use False --repeat 1 seed 7 out_dir /home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/zinc
