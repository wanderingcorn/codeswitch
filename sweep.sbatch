#!/bin/bash
#SBATCH --job-name=cs-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=9-12
#SBATCH --output=logs/%x_%A_%a.out

source ~/codeswitch/.venv/bin/activate
cd ~/codeswitch

LINE=$(grep -vE '^\s*(#|$)' grid.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r NAME MODEL OVERRIDES <<< "$LINE"
echo "== $NAME | $MODEL | $OVERRIDES =="
python -u run.py --name "$NAME" --model "$MODEL" --set $OVERRIDES