#!/bin/bash

# Function to run repeats with SLURM job submission
function run_repeats {
    dataset=$1
    cfg_suffix=$2
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python main.py --cfg ${cfg_file}"
    out_dir="results_10_exp/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    # Run each repeat as a separate job
    for SEED in {0..9}; do
        # Define the output and error file paths based on the dataset name and seed
        slurm_directive="--time=0-15:00:00 --mem=16G --gres=gpu:1 --cpus-per-task=4 --partition=studentbatch"
        
        # Generate output and error filenames with the seed number
        out_file="/home/yandex/MLWG2024/michaelbest/results/${dataset}_GPS_seed${SEED}.out"
        err_file="/home/yandex/MLWG2024/michaelbest/results/${dataset}_GPS_seed${SEED}.err"

        # Add output and error files to the SLURM directive
        slurm_directive+=" --output=${out_file} --error=${err_file}"

        # Construct the full SLURM script to activate Conda environment and run the job
        script="
        #!/bin/bash
        #SBATCH ${slurm_directive}

        # Activate Conda environment
        /home/yandex/MLWG2024/michaelbest/anaconda3/etc/profile.d/conda.sh activate graphgps
        echo 'Activated environment: $CONDA_PREFIX'

        # Navigate to the working directory and run the job
        cd ../CombineGraphGPS
        ${main} --repeat 1 seed ${SEED} ${common_params}
        "

        # Submit the job
        echo "$script" | sbatch
    done
}

# Set configuration directory
cfg_dir="configs"

# Run for Zinc dataset
DATASET="zinc"
run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.10run"

# Run for OGBG-MolPCBA dataset
DATASET="ogbg-molpcba"
run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.10run"

# Run for OGBG-MolHiv dataset
DATASET="ogbg-molhiv"
run_repeats ${DATASET} GPS+RWSE "name_tag GPSwRWSE.GatedGCN+Trf.10run"
