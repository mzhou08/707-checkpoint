#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=56G
#SBATCH --time=6:00:00
#SBATCH --exclude=grogu-1-40,grogu-1-3,grogu-2-22
module load anaconda3-2019
cd /home/mhzhou/vllama2
source activate /home/acl2/mambaforge/envs/vgpt

export PYTHONUNBUFFERED=1
exec $@
