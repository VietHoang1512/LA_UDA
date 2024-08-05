import os
import time
import subprocess

import numpy as np
import pandas as pd

slurm_template = """#!/bin/bash -e
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_output}/slurm_%A.out
#SBATCH --error={slurm_output}/slurm_%A.err
#SBATCH --gpus={num_gpus}
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=36G
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.haopt12@vinai.io
#SBATCH --ntasks=1

# conda activate ../../envs/mamba_final/

export MASTER_PORT={master_port}
export WORLD_SIZE=1

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH=$(pwd):$PYTHONPATH

export MODEL_TYPE={model_type}
export EPOCH_ID={epoch}
export EXP={exp}
export OUTPUT_LOG={output_log}

echo "----------------------------"
echo $MODEL_TYPE $EPOCH_ID $EXP {method} {num_steps}
echo "----------------------------"

CUDA_VISIBLE_DEVICES={device} torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node={num_gpus} vim/sample_sit_ddp.py {sampler} \
    --model $MODEL_TYPE \
    --per-proc-batch-size {batch_size} \
    --image-size {image_size} \
    --ckpt {ckpt_root}/{epoch:07d}.pt \
    --path-type GVP \
    --num-classes {num_cls} \
    --num-fid-samples {n_sample} \
    --sampling-method {method} \
    --num-sampling-steps {num_steps} \
    --diffusion-form {diff_form} \
    --sample-dir samples/{exp} \
    --block-type combined \
    --bimamba-type {bimamba_type} \
    --eval-refdir {eval_refdir} \
    --eval-metric {eval_metric} \
    --cond-mamba \
    --rms-norm \
    --fused-add-norm \
    --learnable-pe \
    --use-attn-every-k-layers -1 \
    --mask-ratio 0.3 \
    --decoder-layer 4 \
    # --label-dropout 0.15 \
    # --cfg-scale 1.50 \
    # --use-final-norm \
    # --enable-fourier-layers \
    # --scanning-continuity \

"""

###### ARGS
n_sample = 10000
num_cls = 1
model_type = "MDT-L/2"
image_size = 256
batch_size = 50
bimamba_type = "none" # 'v2', 'none', 'zigma_8', 'sweep_8', 'jpeg_8', 'sweep_4'
exp = "DiT" # YOUR EXP HERE
ckpt_root = f"./results/{exp}/checkpoints/" # YOUR PATH TO EXP HERE
real_data = "/research/cbim/vast/qd66/workspace/real_samples/celeba_256/" # YOUR REAL DATA HERE
eval_metric = "fid{num_samples}k_full".format(num_samples=int(n_sample//1000))
BASE_PORT = 18078
num_gpus = 4
device = "0,1,2,3" # "4,5,6,7"
epochs = [100,150,200,250,300,350,400,450,500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
config = pd.DataFrame({
    "epochs": epochs,
    "num_steps": [250]*len(epochs),
    "methods": ['dopri5']*len(epochs),
    "cfg_scale": [1.]*len(epochs),
    "diff_form": ["none"]*len(epochs),
    "sampler": ['ODE']*len(epochs),
})
print(config)

###################################
slurm_file_path = f"/research/cbim/vast/qd66/workspace/MambaDiff/slurm/run_{exp}.sh"
slurm_output = f"/research/cbim/medical/qd66/mamba_exp/{exp}/"
output_log = f"{slurm_output}/log"
os.makedirs(slurm_output, exist_ok=True)
job_name = "test"

for idx, row in config.iterrows():
    # device = str(idx % 2)
    # slurm_file_path = f"/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurm_scripts/{exp}/run{device}.sh"
    slurm_command = slurm_template.format(
        job_name=job_name,
        model_type=model_type,
        exp=exp,
        epoch=row.epochs,
        master_port=str(BASE_PORT+idx),
        slurm_output=slurm_output,
        num_gpus=num_gpus,
        output_log=output_log,
        method=row.methods,
        num_steps=row.num_steps,
        device=device,
        cfg_scale=row.cfg_scale,
        diff_form=row.diff_form,
        sampler=row.sampler,
        real_data=real_data,
        ckpt_root=ckpt_root,
        eval_refdir=real_data,
        eval_metric=eval_metric,
        n_sample=n_sample,
        image_size=image_size,
        batch_size = batch_size,
        num_cls = num_cls,
        bimamba_type = bimamba_type \
    )
    mode = "w" if idx == 0 else "a"
    # mode = "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])