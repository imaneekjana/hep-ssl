#!/bin/bash
#SBATCH --job-name=only_dihiggs_training_events_1500
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --qos=regular
#SBATCH --account=m4474_g
#SBATCH --output=job.%j.out

echo "Loading conda"

module load conda

echo "Loaded conda"

echo "Activating colliderml-env"

conda activate colliderml-env

echo "Activated colliderml-env"

cd /global/cfs/cdirs/m4474/aneek/particlemind_aneek

echo "Changed directory"


echo "Trying to run the script"

srun python -m src.train_gnn_colliderml_3