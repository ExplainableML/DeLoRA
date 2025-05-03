import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, DataCollatorWithPadding,AutoModelForSequenceClassification, get_linear_schedule_with_warmup, set_seed
from peft.src.peft import (
    get_peft_model,
    LoraConfig,
    DeloraConfig,
    PeftType,
)
import evaluate
from datasets import load_dataset, load_from_disk

from data_utils import load_glue_data_final


# - args
parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument("--micro_batch_size", type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument("--task", type=str, default="cola")
parser.add_argument("--model_name_or_path", type=str, default="roberta-large")
parser.add_argument("--adapter_type", type=str, default="delora")
parser.add_argument("--r", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--delora_lambda", type=float, default=1.)
parser.add_argument("--delora_lambda_lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug", type=bool, default=False)

args = parser.parse_args()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(args)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# - setup
set_seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

model_name_or_path = args.model_name_or_path
device = "cuda"
lr = float(args.lr)
task_key = args.task if args.task != "mnli-mm" else "mnli"

try:
    dataset = load_from_disk("./data/" + task_key)
except:
    dataset = load_dataset("glue", task_key)
    dataset.save_to_disk("./data/" + task_key)
print(dataset)
metric = evaluate.load("glue", task_key)

# check task computing correct metric
fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
print(metric.compute(predictions=fake_preds, references=fake_labels))

# target modules to finetune
if model_name_or_path.split("/")[-1] in ["roberta-base", "roberta-large"]:
    target_modules = ["query", "value"]
    str_target_modules = "".join([m[0] for m in target_modules]) # get first letter of each module name
else:
    raise ValueError("Add here name of target modules for not supported model")

# setup repos
logdir = Path(".logs")
model_logdir = logdir / model_name_or_path.split("/")[-1]
task_model_logdir = model_logdir / args.task

os.makedirs(logdir, exist_ok=True, mode=0o777)
os.makedirs(model_logdir, exist_ok=True, mode=0o777)
os.makedirs(task_model_logdir, exist_ok=True, mode=0o777)

if args.adapter_type == "lora":
    args.lora_alpha = args.delora_lambda
    settings = f"{args.adapter_type}_r{args.r}_{str_target_modules}_a{args.lora_alpha}_d{args.dropout}_lr{lr}_bs{args.batch_size}_ep{args.num_epochs}_ml{args.max_length}_seed{args.seed}"
elif args.adapter_type == "delora":
    settings = f"{args.adapter_type}_r{args.r}_{str_target_modules}_a{args.delora_lambda}_d{args.dropout}_lr{lr}_bs{args.batch_size}_ep{args.num_epochs}_ml{args.max_length}_dllr{args.delora_lambda_lr}_seed{args.seed}"

if os.path.exists(task_model_logdir / f"{settings}.txt") and not args.debug:
    print(f"~~~~ Already exists: {task_model_logdir}/{settings}.txt ~~~~~")
    exit()


# - peft config
if args.adapter_type == "lora":
    peft_type = PeftType.LORA
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
    )

elif args.adapter_type == "delora":
    peft_type = PeftType.DELORA
    peft_config = DeloraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=args.r,
        delora_lambda=args.delora_lambda,
        delora_dropout=args.dropout,
        target_modules=target_modules,
    )


# - preprocess data
dataset_name = args.task
model_type = "roberta_base"

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_dataset, eval_dataset, test_dataset, num_labels = load_glue_data_final(
    tokenizer = tokenizer, dataset_name = dataset_name, model_type = model_type, seed=args.seed
    )

print(f'train nums: {str(len(train_dataset))}, eval nums: {str(len(eval_dataset))}, test nums: {str(len(test_dataset))}')
print(f'train example: {train_dataset[0]["input_ids"]}')

if args.micro_batch_size > args.batch_size:
    print(f"WARNING: micro_batch_size ({args.micro_batch_size}) is greater than batch_size ({args.batch_size}). Setting micro_batch_size to batch_size.")
    args.micro_batch_size = args.batch_size

data_collator = DataCollatorWithPadding(tokenizer)
train_dataloader = DataLoader(
                dataset = train_dataset, 
                shuffle = True,
                batch_size = args.micro_batch_size,
                collate_fn = data_collator,
               )

eval_dataloader = DataLoader(
                dataset = eval_dataset, 
                batch_size = args.micro_batch_size,
                collate_fn = data_collator,
               )

test_dataloader = DataLoader(
                dataset = test_dataset, 
                batch_size = args.micro_batch_size,
                collate_fn = data_collator,
               )


# - model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, return_dict=True, num_labels=num_labels
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model

if args.adapter_type == "delora":
    from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad and "delora_lambda" in n)
            ],
            "lr": args.delora_lambda_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad and "delora_lambda" not in n)
            ],
            "lr": lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
else:
    optimizer = AdamW(params=model.parameters(), lr=lr)

# instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)

# gradient accumulation
accumulation_steps = args.batch_size // args.micro_batch_size

best_eval_metric = 0
best_epoch = 0
model.to(device)
for epoch in range(args.num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # perform optimizer step for remaining gradients if any
    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        if args.task != "stsb":
            predictions = outputs.logits.argmax(dim=-1)
        else:
            predictions = outputs.logits
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"VAL: epoch {epoch}:", eval_metric)

    # write in txt
    with open(task_model_logdir / f"{settings}.txt", "a") as f:
        f.write(f"epoch {epoch}: {eval_metric}\n")
    try:
        os.chmod(task_model_logdir / f"{settings}.txt", 0o770)
    except:
        pass

    # if metric better than best_eval_metric
    if float(float(list(eval_metric.values())[0])) > float(best_eval_metric):
        best_eval_metric = float(list(eval_metric.values())[0])
        best_epoch = epoch
        # get test predictions
        model.eval()
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            if args.task != "stsb":
                predictions = outputs.logits.argmax(dim=-1)
            else:
                predictions = outputs.logits
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        test_metric = metric.compute()
        print(f"TEST: epoch {epoch}:", test_metric)

        # write in txt
        with open(task_model_logdir / f"{settings}_TEST.txt", "a") as f:
            f.write(f"epoch {epoch}: {test_metric}\n")
        try:
            os.chmod(task_model_logdir / f"{settings}_TEST.txt", 0o770)
        except:
            pass

# end with separator
with open(task_model_logdir / f"{settings}.txt", "a") as f:
    f.write(f"~~~~~~~~~~~~~~~\n")

with open(task_model_logdir / f"{settings}_TEST.txt", "a") as f:
    f.write(f"~~~~~~~~~~~~~~~\n")
