#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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

cd /scratch/hvp2011/implement/UDA/LA/

args=("$@")
echo $# arguments passed
echo ${args[0]} ${args[1]} ${args[2]} 
radius=${args[0]}
tradeoff=${args[1]}
align=${args[2]}
python main.py  --entropy_tradeoff .0 --radius $radius --tradeoff $tradeoff --align $align --M1 16 --M2 16 --threshold .4 --data_root /vast/hvp2011/data/image-clef --dataset ImageCLEF

# python main.py  --entropy_tradeoff .0 --radius .0001 --tradeoff .1 --align .0 --M1 16 --M2 16 --threshold .4 --data_root /vast/hvp2011/data/image-clef --dataset ImageCLEF
# python main.py --data_root /vast/hvp2011/data/office_home/ --dataset OfficeHome
# python main.py --data_root /vast/hvp2011/data/domainnet/ --dataset DomainNet
# python main.py --data_root /vast/hvp2011/data/office-31/ --dataset Office31

