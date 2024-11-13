#!/bin/sh
#SBATCH --job-name=GPS+RWSE-ogbg-molhiv-seed5
#SBATCH --partition=studentbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --output=/home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/ogbg-molhiv/output_seed5.out
#SBATCH --error=/home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/ogbg-molhiv/error_seed5.err
#SBATCH --time=4300
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4

echo 'Starting job...'
/home/yandex/MLWG2024/michaelbest/anaconda3/etc/profile.d/conda.sh activate graphgps
Activated environment: $CONDA_PREFIX

cd /home/yandex/MLWG2024/michaelbest/CombineGraphGPS
python main.py --cfg /home/yandex/MLWG2024/michaelbest/CombineGraphGPS/configs/GPS/ogbg-molhiv-GPS+RWSE.yaml wandb.use False seed 5 out_dir /home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/ogbg-molhiv
