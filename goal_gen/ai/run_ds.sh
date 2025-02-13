#!/bin/bash

cd $HOME/DeepSpeedExamples/training/HelloDeepSpeed

MASTER_ADDR=$(getent hosts $(scontrol show hostname $SLURM_NODELIST | head -n 1) | awk '{ print $1 }')
MASTER_PORT=29500
NODE_RANK=$SLURM_NODEID

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Node Rank: $NODE_RANK"

deepspeed --hostfile="$HOME/nccl_goal_generator/example/myhostfile" \
    --no_ssh --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train_bert_ds.py --num_iterations=1 --checkpoint_dir "$HOME/DeepSpeedExamples/training/HelloDeepSpeed/experiment_deepspeed"

    