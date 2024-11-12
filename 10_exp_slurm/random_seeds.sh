function run_repeats {
    dataset=$1
    cfg_suffix=$2
    cfg_overrides=$3

    main="python main.py --cfg ${cfg_file}"
    out_dir="results_full_exp/${dataset}"  # Set the output dir
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "Output dir: ${out_dir}"

    # Create the directory for the .slurm files if it doesn't exist
s
    # Loop through each seed (0 to 9)
    for SEED in {0..9}; do
        # Generate unique output and error file names for each seed
        out_file="${out_dir}/output_seed${SEED}.out"
        err_file="${out_dir}/error_seed${SEED}.err"

        # Create a .slurm file for each seed
        slurm_file="${dataset}/job_seed${SEED}.slurm"

        # Create the Slurm job script for this seed
        echo "#!/bin/bash" > ${slurm_file}
        echo "#SBATCH --job-name=${cfg_suffix}-${dataset}-seed${SEED}" >> ${slurm_file}
        echo "#SBATCH --partition=studentbatch"
        echo "#SBATCH --nodes=1" >> ${slurm_file}
        echo "#SBATCH --ntasks=1" >> ${slurm_file}
        echo "#SBATCH --mem=50000" >> ${slurm_file}
        echo "#SBATCH --output=${out_file}" >> ${slurm_file}
        echo "#SBATCH --error=${err_file}" >> ${slurm_file}
        echo "#SBATCH ${slurm_directive}" >> ${slurm_file}
        echo "" >> ${slurm_file}
        echo "echo 'Starting job...'" >> ${slurm_file}
        echo "/home/yandex/MLWG2024/michaelbest/anaconda3/etc/profile.d/conda.sh activate graphgps" >> ${slurm_file}
        echo "echo 'Activated environment: \$CONDA_PREFIX'" >> ${slurm_file}
        echo "" >> ${slurm_file}
        echo "cd /home/yandex/MLWG2024/michaelbest/CombineGraphGPS" >> ${slurm_file}
        echo "python main.py --cfg ${cfg_file} --repeat 1 seed ${SEED} ${common_params}" >> ${slurm_file}

        # Echo and submit the job
        echo "Created Slurm file: ${slurm_file}"
    done
}

# Datasets and slurm directives
DATASETS=("zinc" "ogbg-molhiv" "ogbg-molpcba")

# The slurm directives
slurm_directive="--time=4300 --gres=gpu:2 --cpus-per-task=4 "
run_repeats zinc GPS+RWSE "name_tag GPSwRWSE.10run"
run_repeats ogbg-molhiv GPS+RWSE "name_tag GPSwRWSE.10run"
run_repeats ogbg-molpcba GPS+RWSE "name_tag GPSwRWSE.10run"



