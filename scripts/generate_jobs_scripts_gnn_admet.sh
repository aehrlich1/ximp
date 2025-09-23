#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# generate_jobs.sh  –  Slurm job file generator for Polaris (no file checks)
# ---------------------------------------------------------------------------

PYTHON_SCRIPT="main.py"

# ──────────────────  Hyper-parameter grids (from YAML)  ──────────────────
TARGET_TASKS=("MLM" "HLM" "KSOL" "LogD" "MDR1-MDCKII")

BATCH_SIZES=(64 128)
LRS=(1e-3)
WEIGHT_DECAYS=(1e-4)

HIDDEN_CHANNELS=(16 32)
OUT_CHANNELS=(16 32)
NUM_LAYERS=(1 2 3)

#JT_COARSITY=(1 2 3)
#RG_EMBEDDING_DIMS=(16 32)

DROPOUT=(0.1)
PROJ_HIDDEN_DIM=(16 32)
EPOCHS=(50 100 150)

REPR_MODEL=("GIN" "GCN" "GAT" "GraphSAGE" "HIMP")

# ───────────────────  Constants (single-valued YAML)  ────────────────────
TASK="admet"
ENCODING_DIM=8
NUM_CV_FOLDS=5
NUM_CV_BINS=10
SCAFFOLD_SPLIT_VAL_SZ=0.1

OUT_DIM=1

# ───────────────────  Slurm defaults – tweak as needed  ──────────────────
SLURM_TIME="1-06:00:00"
SLURM_PARTITION="p_low"
SLURM_CPUS=8
SLURM_MEM_PER_CPU="6G"

# ─────────────────────────────────────────────────────────────────────────
idx=0
for tgt in "${TARGET_TASKS[@]}"; do
  for bs in "${BATCH_SIZES[@]}";    do
    for lr in "${LRS[@]}";          do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        for hc in "${HIDDEN_CHANNELS[@]}";  do
          for oc in "${OUT_CHANNELS[@]}";   do
            for nl in "${NUM_LAYERS[@]}";   do
              #for fr in "${JT_COARSITY[@]}"; do
                for rm in "${REPR_MODEL[@]}"; do
                  for ep in "${EPOCHS[@]}"; do
                    for phd in "${PROJ_HIDDEN_DIM[@]}"; do
                      for dout in "${DROPOUT[@]}"; do

                    ((idx++))
                    sub="submit_${idx}.submit"

cat > "$sub" << EOF
#!/bin/bash
#SBATCH --job-name="Polaris: ${tgt} bs=${bs} lr=${lr} hc=${hc}"
#SBATCH --comment="FG Data Mining"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=${SLURM_TIME}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem-per-cpu=${SLURM_MEM_PER_CPU}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --requeue

export ENV_MODE=permanant
export ENV_NAME="ximp"
module load miniforge

python ${PYTHON_SCRIPT} \
  --task ${TASK} \
  --target_task "${tgt}" \
  --batch_size ${bs} \
  --epochs ${ep} \
  --lr ${lr} \
  --weight_decay ${wd} \
  --num_cv_folds ${NUM_CV_FOLDS} \
  --num_cv_bins ${NUM_CV_BINS} \
  --scaffold_split_val_sz ${SCAFFOLD_SPLIT_VAL_SZ} \
  --encoding_dim ${ENCODING_DIM} \
  --repr_model ${rm} \
  --hidden_channels ${hc} \
  --out_channels ${oc} \
  --num_layers ${nl} \
  --dropout ${dout} \
  --proj_hidden_dim ${phd} \
  --out_dim ${OUT_DIM}
EOF

                        chmod +x "$sub"
                        echo "Created $sub"
                      done
                    done
                  done
                done
              #done
            done
          done
        done
      done
    done
  done
done

echo
echo "Generated ${idx} submit files."
