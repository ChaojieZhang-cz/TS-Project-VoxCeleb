#!/bin/bash

#SBATCH --job-name=CNN_LSTM_Spe_2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=cz2064@nyu.edu

module purge
module load anaconda3/5.3.1
source activate /scratch/cz2064/envs/dl4med

srun python -u CNN_LSTM_Spectrogram_2.py
