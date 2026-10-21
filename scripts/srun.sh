#!/bin/bash
#SBATCH -J beelab.railight.train
#SBATCH -o /home/a00161/stacy.en14/outputs/railight/logs/beelab.railight.train.%j.log
#SBATCH -e /home/a00161/stacy.en14/outputs/railight/logs/beelab.railight.train.%j.log
#SBATCH -p h200q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gpu:1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

set -euo pipefail

if [ -n "${CONFIG:-}" ] || [ -n "${SCRIPT:-}" ]; then
    echo "ERROR: the CONFIG and SCRIPT environment variables are no longer read." >&2
    echo "       Edit the values at the top of this script instead." >&2
    exit 2
fi

STAGE=train.sh
CONDA_ENV=railight

cd "$(dirname "$0")"

set +u
source /etc/profile.d/modules.sh
set -u

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate "$CONDA_ENV"
set -u

bash "$STAGE"
