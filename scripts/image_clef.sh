#!/bin/bash

# cd ..

# custom config
DATA=/vast/hvp2011/data/    # you may change your path to dataset here
TRAINER=DAMP

DATASET=image_clef # name of the dataset
CFG=damp  # config file
TAU=0.6 # pseudo label threshold
U=1.0 # coefficient for loss_u
SEED=1

NAME=p
DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains b c i --target-domains p --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}
