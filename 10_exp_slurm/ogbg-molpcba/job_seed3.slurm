#!/bin/sh
#SBATCH --job-name=GPS+RWSE-ogbg-molpcba-seed3
#SBATCH --partition=studentbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --output=/home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/ogbg-molpcba/output_seed3.out
#SBATCH --error=/home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/ogbg-molpcba/error_seed3.err
#SBATCH --time=2000
#SBATCH --gpus=3
#SBATCH --cpus-per-task=4

echo 'Starting job...'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/yandex/MLWG2024/michaelbest/anaconda3/etc/profile.d/conda.sh activate graphgps
Activated environment: $CONDA_PREFIX

cd /home/yandex/MLWG2024/michaelbest/CombineGraphGPS
python main.py --cfg /home/yandex/MLWG2024/michaelbest/CombineGraphGPS/configs/GPS/ogbg-molpcba-GPS+RWSE.yaml wandb.use False seed 3 out_dir /home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/ogbg-molpcba
