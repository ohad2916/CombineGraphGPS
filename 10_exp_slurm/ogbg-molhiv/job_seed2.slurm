#!/bin/bash
#SBATCH --job-name=GPS+RWSE-ogbg-molhiv-seed2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --output=results_full_exp/ogbg-molhiv/output_seed2.out
#SBATCH --error=results_full_exp/ogbg-molhiv/error_seed2.err
#SBATCH --time=4300 --gres=gpu:2 --cpus-per-task=4 

echo 'Starting job...'
/home/yandex/MLWG2024/michaelbest/anaconda3/etc/profile.d/conda.sh activate graphgps
echo 'Activated environment: $CONDA_PREFIX'

cd /home/yandex/MLWG2024/michaelbest/CombineGraphGPS
python main.py --cfg  --repeat 1 seed 2 out_dir results_full_exp/ogbg-molhiv name_tag GPSwRWSE.10run
