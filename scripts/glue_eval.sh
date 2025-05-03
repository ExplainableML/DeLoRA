#!/bin/bash

# Arguments
TASK=$1  # mnli, sst2, qnli, qqp, rte, stsb, mrpc, cola
SEED=$2  # 42 43 44 45 46

echo "Running task $TASK with seed $SEED"

# Set parameters based on task
if [ "$TASK" = "mnli" ]; then
    delora_lambda=12
    lr=1e-3
    batch_size=32
    num_epochs=30
    dropout=0
    delora_lambda_lr=3e-3
elif [ "$TASK" = "sst2" ]; then
    delora_lambda=12
    lr=1e-3
    batch_size=32
    num_epochs=30
    dropout=0.1
    delora_lambda_lr=3e-3
elif [ "$TASK" = "qnli" ]; then
    delora_lambda=12
    lr=3e-3
    batch_size=32
    num_epochs=25
    dropout=0.25
    delora_lambda_lr=3e-3
elif [ "$TASK" = "qqp" ]; then
    delora_lambda=12
    lr=3e-3
    batch_size=32
    num_epochs=25
    dropout=0.1
    delora_lambda_lr=3e-3
elif [ "$TASK" = "cola" ]; then
    delora_lambda=4
    lr=1e-2
    batch_size=8
    num_epochs=80
    dropout=0.2
    delora_lambda_lr=1e-2
elif [ "$TASK" = "rte" ]; then
    delora_lambda=4
    lr=1e-2
    batch_size=32
    num_epochs=80
    dropout=0
    delora_lambda_lr=1e-2
elif [ "$TASK" = "stsb" ]; then
    delora_lambda=12
    lr=1e-2
    batch_size=8
    num_epochs=40
    dropout=0.2
    delora_lambda_lr=1e-2
else
    echo "Invalid task: $TASK"
    echo "Valid tasks are: mnli, sst2, qnli, qqp, cola, rte, stsb"
    exit 1
fi

python glue_delora.py \
--model_name_or_path FacebookAI/roberta-base \
--lr $lr \
--batch_size $batch_size \
--num_epochs $num_epochs \
--delora_lambda $delora_lambda \
--dropout $dropout \
--task $TASK \
--adapter_type delora \
--delora_lambda_lr $delora_lambda_lr \
--r 8 \
--micro_batch_size 32 \
--seed $SEED
