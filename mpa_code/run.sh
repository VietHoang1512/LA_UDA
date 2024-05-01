#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --job-name=UDA
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/python
nvidia-smi

cd /scratch/hvp2011/implement/UDA/mpa_code

python main.py --data_root /vast/hvp2011/data/image-clef --dataset ImageCLEF --radius .0 
# python main.py --data_root /vast/hvp2011/data/office_home/ --dataset OfficeHome
# python main.py --data_root /vast/hvp2011/data/domainnet/ --dataset DomainNet
# python main.py --data_root /vast/hvp2011/data/office-31/ --dataset Office31

