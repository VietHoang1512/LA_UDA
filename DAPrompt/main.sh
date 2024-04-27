#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --job-name=DAPL
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/python
nvidia-smi

cd /home/hvp2011/implement/UDA/DAPrompt/

# custom config
DATA=/vast/hvp2011/data/ 

TRAINER=LAPA
radius=0.001
align=10.0
tradeoff=.5


DATASET=officehome # name of the dataset
CFG=ep25-32 # config file
SEED=1

# for source in art clipart product real_world
# do
#     for target in art clipart product real_world
#     do
#         DIR=output/${DATASET}/${TRAINER}/${CFG}/${source}2${target}/radius=$radius/align=$align/tradeoff=$tradeoff/seed_${SEED}/
#         # DIR=output/${DATASET}/${TRAINER}/${CFG}/${source}2${target}/seed_${SEED}
#         echo "Run this job and save the output to ${DIR}"
#         mkdir -p ${DIR}
#         python train.py \
#         --root ${DATA} \
#         --seed ${SEED} \
#         --trainer ${TRAINER} \
#         --dataset-config-file configs/datasets/${DATASET}.yaml \
#         --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#         --source-domains $source \
#         --target-domains $target \
#         --output-dir ${DIR} \
#         TRAINER.LAPA.radius $radius \
#         TRAINER.LAPA.align $align \
#         TRAINER.LAPA.tradeoff $tradeoff
#     done
# done
radius=0.001
align=10.0
tradeoff=.5
DATA=/vast/hvp2011/data/ 
DATASET=officehome # name of the dataset
CFG=ep25-32 # config file
SEED=1
TRAINER=LAMPA

DIR=output/${DATASET}/${TRAINER}/${CFG}/art/radius=$radius/align=$align/tradeoff=$tradeoff/seed_${SEED}/
echo "Run this job and save the output to ${DIR}"
mkdir -p ${DIR}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--source-domains clipart product real_world  \
--target-domains art \
--output-dir ${DIR} \
DATALOADER.TRAIN_X.SAMPLER RandomDomainSampler \
TRAINER.LAMPA.radius $radius \
TRAINER.LAMPA.align $align \
TRAINER.LAMPA.tradeoff $tradeoff \
DATALOADER.TRAIN_X.BATCH_SIZE 36

DIR=output/${DATASET}/${TRAINER}/${CFG}/clipart/radius=$radius/align=$align/tradeoff=$tradeoff/seed_${SEED}/
echo "Run this job and save the output to ${DIR}"
mkdir -p ${DIR}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--source-domains art product real_world \
--target-domains clipart  \
--output-dir ${DIR} \
DATALOADER.TRAIN_X.SAMPLER RandomDomainSampler \
TRAINER.LAMPA.radius $radius \
TRAINER.LAMPA.align $align \
TRAINER.LAMPA.tradeoff $tradeoff  \
DATALOADER.TRAIN_X.BATCH_SIZE 36

DIR=output/${DATASET}/${TRAINER}/${CFG}/product/radius=$radius/align=$align/tradeoff=$tradeoff/seed_${SEED}/
echo "Run this job and save the output to ${DIR}"
mkdir -p ${DIR}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--source-domains art clipart real_world \
--target-domains  product  \
--output-dir ${DIR} \
DATALOADER.TRAIN_X.SAMPLER RandomDomainSampler \
TRAINER.LAMPA.radius $radius \
TRAINER.LAMPA.align $align \
TRAINER.LAMPA.tradeoff $tradeoff  \
DATALOADER.TRAIN_X.BATCH_SIZE 36

DIR=output/${DATASET}/${TRAINER}/${CFG}/real_world/radius=$radius/align=$align/tradeoff=$tradeoff/seed_${SEED}/
echo "Run this job and save the output to ${DIR}"
mkdir -p ${DIR}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--source-domains art clipart product \
--target-domains  real_world \
--output-dir ${DIR} \
DATALOADER.TRAIN_X.SAMPLER RandomDomainSampler \
TRAINER.LAMPA.radius $radius \
TRAINER.LAMPA.align $align \
TRAINER.LAMPA.tradeoff $tradeoff  \
DATALOADER.TRAIN_X.BATCH_SIZE 36


# TRAINER=DAPL
# for source in art clipart product real_world
# do
#     for target in art clipart product real_world
#     do
#         DIR=output/${DATASET}/${TRAINER}/${CFG}/${source}2${target}/seed_${SEED}
#         echo "Run this job and save the output to ${DIR}"
#         mkdir -p ${DIR}
#         python train.py \
#         --root ${DATA} \
#         --seed ${SEED} \
#         --trainer ${TRAINER} \
#         --dataset-config-file configs/datasets/${DATASET}.yaml \
#         --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#         --source-domains $source \
#         --target-domains $target \
#         --output-dir ${DIR} 
#     done
# done