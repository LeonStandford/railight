#!/bin/bash
#SBATCH -J beelab.railight.exp3.da.batch16.h100.train
#SBATCH -o /home/a00161/stacy.en14/models/railight/logs/beelab.railight.exp3.da.batch16.h100.train.%j.log
#SBATCH -e /home/a00161/stacy.en14/models/railight/logs/beelab.railight.exp3.da.batch16.h100.train.%j.log
#SBATCH -p defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gpu:2
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00

set -euo pipefail

REPO_ROOT=/home/a00161/stacy.en14/Long/railight
set +u
source /etc/profile.d/modules.sh
set -u

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate railight
set -u

cd "$REPO_ROOT/src/scripts"
bash train.sh
