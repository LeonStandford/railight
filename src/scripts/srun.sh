#!/bin/bash
#SBATCH -J beelab.railight.exp3
#SBATCH -o /local/stacy.en14/models/railight/logs/beelab.railight.exp3.%j.log
#SBATCH -e /local/stacy.en14/models/railight/logs/beelab.railight.exp3.%j.log
#SBATCH -p defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gpu:1
#SBATCH --mem=100G
#SBATCH --time=5-00:00:00

set -e

REPO_ROOT=/home/a00161/stacy.en14/Long/railight

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate railight

cd "$REPO_ROOT/src/scripts"
bash train.sh
