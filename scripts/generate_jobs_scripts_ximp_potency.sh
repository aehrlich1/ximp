#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# generate_jobs.sh  –  Slurm job file generator for Polaris (no file checks)
# ---------------------------------------------------------------------------

PYTHON_SCRIPT="main.py"

# ──────────────────  Hyper-parameter grids (from YAML)  ──────────────────
TARGET_TASKS=("pIC50 (MERS-CoV Mpro)" "pIC50 (SARS-CoV-2 Mpro)")

BATCH_SIZES=(64 128)
LRS=(1e-3)
WEIGHT_DECAYS=(1e-4)

HIDDEN_CHANNELS=(16 32)
OUT_CHANNELS=(16 32)
NUM_LAYERS=(1 2 3)

JT_COARSITY=(1 2 3) # Needs to be 1 if USE_JT=False to avoid sampling bias
RG_EMBEDDING_DIMS=(16 32)

DROPOUT=(0.1)
PROJ_HIDDEN_DIM=(16 32)
EPOCHS=(50 100 150)

USE_JT=(True) # False combinations need to be ran seperately with JT_COARSITY adjusted accordingly to avoid sampling bias
USE_ERG=(True) # False

# ───────────────────  Constants (single-valued YAML)  ────────────────────
TASK="potency"
ENCODING_DIM=8
NUM_CV_FOLDS=5
NUM_CV_BINS=10
SCAFFOLD_SPLIT_VAL_SZ=0.1
REPR_MODEL=("XIMP_a" "XIMP_b" "XIMP_c")
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
              for jc in "${JT_COARSITY[@]}"; do
                for rg in "${RG_EMBEDDING_DIMS[@]}"; do
                  for ep in "${EPOCHS[@]}"; do
                    for phd in "${PROJ_HIDDEN_DIM[@]}"; do
                      for dout in "${DROPOUT[@]}"; do
                        for USE_JT in "${USE_JT[@]}"; do
                          for use_erg in "${USE_ERG[@]}"; do
                            for rm in "${REPR_MODEL[@]}"; do

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
  --use_erg ${use_erg} \
  --use_jt ${use_jt} \
  --jc_coarsity ${jc} \
  --rg_embedding_dim ${rg} \
  --proj_hidden_dim ${phd} \
  --out_dim ${OUT_DIM}
EOF

                                chmod +x "$sub"
                              echo "Created $sub"
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo
echo "Generated ${idx} submit files."
