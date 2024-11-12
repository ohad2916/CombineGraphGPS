#!/bin/bash
#SBATCH --job-name=GPS+RWSE-ogbg-molpcba-seed2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --output=/home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/ogbg-molpcba/output_seed2.out
#SBATCH --error=/home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/ogbg-molpcba/error_seed2.err
#SBATCH --time=4300
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4

echo 'Starting job...'
/home/yandex/MLWG2024/michaelbest/anaconda3/etc/profile.d/conda.sh activate graphgps
echo 'Activated environment: $CONDA_PREFIX'

cd /home/yandex/MLWG2024/michaelbest/CombineGraphGPS
python main.py --cfg /config/GPS/ogbg-molpcba-GPS+RWSE.yaml wandb.use False --repeat 1 seed 2 out_dir /home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/ogbg-molpcba
