function run_repeats {
    dataset=$1
    cfg_suffix=$2

    main="python main.py --cfg /home/yandex/MLWG2024/michaelbest/CombineGraphGPS/configs/GPS/${dataset}-${cfg_suffix}.yaml wandb.use False"
    out_dir="/home/yandex/MLWG2024/michaelbest/CombineGraphGPS/results_10_exp/${dataset}"  # Set the output dir
  
    echo "Run program: ${main}"
    echo "Output dir: ${out_dir}"

    # Create the directory for the .slurm files if it doesn't exist

    # Loop through each seed (0 to 9)
    for SEED in {0..9}; do
        # Generate unique output and error file names for each seed
        out_file="${out_dir}/output_seed${SEED}.out"
        err_file="${out_dir}/error_seed${SEED}.err"

        # Create a .slurm file for each seed
        slurm_file="${dataset}/job_seed${SEED}.slurm"

        # Create the Slurm job script for this seed
        echo "#!/bin/sh" > ${slurm_file}
        echo "#SBATCH --job-name=${cfg_suffix}-${dataset}-seed${SEED}" >> ${slurm_file}
        echo "#SBATCH --partition=studentbatch" >> ${slurm_file}
        echo "#SBATCH --nodes=1" >> ${slurm_file}
        echo "#SBATCH --ntasks=1" >> ${slurm_file}
        echo "#SBATCH --mem=50000" >> ${slurm_file}
        echo "#SBATCH --output=${out_file}" >> ${slurm_file}
        echo "#SBATCH --error=${err_file}" >> ${slurm_file}
	    echo "#SBATCH --time=4300" >> ${slurm_file}
        echo "#SBATCH --gres=gpu:3" >> ${slurm_file}
	    echo "#SBATCH --cpus-per-task=4" >> ${slurm_file}
        echo "" >> ${slurm_file}
        echo "echo 'Starting job...'" >> ${slurm_file}
        echo "/home/yandex/MLWG2024/michaelbest/anaconda3/etc/profile.d/conda.sh activate graphgps" >> ${slurm_file}
        echo "Activated environment: \$CONDA_PREFIX" >> ${slurm_file}
        echo "" >> ${slurm_file}
        echo "cd /home/yandex/MLWG2024/michaelbest/CombineGraphGPS" >> ${slurm_file}
        echo "${main} seed ${SEED} out_dir ${out_dir}" >> ${slurm_file}

        # Echo and submit the job
        echo "Created Slurm file: ${slurm_file}"
    done
}

# Datasets and slurm directives
DATASETS=("zinc" "ogbg-molhiv" "ogbg-molpcba")

# Loop through each dataset and create the Slurm files and submit the jobs

for DATASET in ${DATASETS[@]}; do
    if [ ${DATASET} == "zinc" ]; then
        cfg_dir="configs/GPS"
        cfg_suffix="GPS+RWSE"
        cfg_overrides="name_tag GPSwRWSE.10run"
    elif [ ${DATASET} == "ogbg-molhiv" ]; then
        cfg_dir="configs/GPS"
        cfg_suffix="GPS+RWSE"
        cfg_overrides="name_tag GPSwRWSE.GatedGCN+Trf.10run"
    elif [ ${DATASET} == "ogbg-molpcba" ]; then
        cfg_dir="configs/GPS"
        cfg_suffix="GPS+RWSE"
        cfg_overrides="name_tag GPSwRWSE.GatedGCN+Trf.10run"
    fi

    run_repeats ${DATASET} ${cfg_suffix} 
done



