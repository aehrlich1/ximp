#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT_PATH="experiments.py"

# Argument arrays
depths=(2 3 4 5)
models=("GIN" "GCN")
activations=("relu")
num_clean=(5)
num_dirty=(5)
prune_percentage=(0.1 0.2 0.4 0.6 0.7 0.5 0.3 0.9 0.8)
datasets=("MUTAG" "AIDS" "PTC_FM" "PTC_MR" "NCI1" "PROTEINS" "ENZYMES" "MSRC_9" "MSRC_21C" "IMDB-BINARY")

current_combination=0
new_combination=0
start_time=$(date +%s)

# OUTERMOST: datasets
for ds in "${datasets[@]}"; do
  for d in "${depths[@]}"; do
    for m in "${models[@]}"; do
      for a in "${activations[@]}"; do
        for nc in "${num_clean[@]}"; do
          for nd in "${num_dirty[@]}"; do
            for ppc in "${prune_percentage[@]}"; do

              ((current_combination++))
              ppc_int=$(printf "%.0f" "$(echo "$ppc * 100" | bc)")

              fname="model_${m}_depth_${d}_activation_${a}_numClean_${nc}_numDirty_${nd}_prunePerc_${ppc_int}_dataset_${ds}.json"
              path="./results"
              full_path="${path}/${fname}"

              if [ -e "$full_path" ]; then
                echo "File exists at path: $full_path"
              else
                ((new_combination++))
                cat > submit_${new_combination}.submit << EOF
#!/bin/bash

#SBATCH --job-name="ICML Rebuttal: D=$d M=$m A=$a NC=$nc ND=$nd PP=$ppc DS=$ds"
#SBATCH --comment="FG Data Mining / Lorenz Kummer"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --constraint="module-miniforge"
#SBATCH --partition=p_low
#SBATCH --requeue

if [ -e "$full_path" ]; then
    echo "File exists at path: $full_path"
    exit 0
fi

export ENV_MODE=permanant
export ENV_NAME="lorenz_bitflips"
module load miniforge

python $PYTHON_SCRIPT_PATH -d $d -m $m -a $a -nc $nc -nd $nd -ppc $ppc -ds $ds

module purge
EOF
              fi

            done
          done
        done
      done
    done
  done
done

echo "All combinations have been created."

