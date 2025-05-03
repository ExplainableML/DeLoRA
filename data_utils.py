# Script from https://github.com/mlwu22/RED

import os
import numpy as np
from torch.utils.data import Subset
from datasets import load_from_disk
from datasets import disable_caching
disable_caching()


glue_data_keys_map = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2")
}

glue_data_num_labels_map = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "stsb": 1,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2
}

roberta_base_len_map={
    "mnli":256,
    "sst2":256,
    "mrpc":256,
    "cola":256,
    "qnli":256,
    "qqp":256,
    "rte":256,
    "stsb":256
}

roberta_large_len_map={
    "mnli":256,
    "sst2":256,
    "mrpc":256,
    "cola":256,
    "qnli":256,
    "qqp":256,
    "rte":256,
    "stsb":256
}



# load dataset and tokenization
def load_glue_data_final(tokenizer, dataset_name: str, max_seq_length: int = 256, seed=42, max_label_seq_length=5, model_type="roberta_base"):
        main_dir = os.path.dirname(os.path.abspath(__file__))
        if(model_type=="roberta_base"):
            dataset = load_from_disk(
                os.path.join(
                    main_dir,
                    "data",
                    dataset_name
                ))
            max_seq_length = roberta_base_len_map[dataset_name]
        elif(model_type == "roberta_large"):
            dataset = load_from_disk(
                os.path.join(
                    main_dir,
                    "data",
                    dataset_name
                ))
            max_seq_length = roberta_large_len_map[dataset_name]

        sentence1_key, sentence2_key = glue_data_keys_map[dataset_name]

        dataset = dataset.map(lambda examples: tokenization(
                                    examples=examples,
                                    tokenizer=tokenizer,
                                    max_seq_length=max_seq_length,
                                    max_label_seq_length=max_label_seq_length,
                                    sentence1_key=sentence1_key,
                                    sentence2_key=sentence2_key,
                                    model_type = model_type,
                                    dataset_name = dataset_name
                                    ),
                              batched=True,
                              )
        if(sentence2_key):
            dataset = dataset.remove_columns([sentence1_key, sentence2_key, "idx", "label"])
        else:
            dataset = dataset.remove_columns([sentence1_key, "idx", "label"])

        train_dataset = dataset["train"]
        eval_dataset = dataset["validation_matched"] if dataset_name == "mnli" else dataset["validation"]

        permuted_indices = np.random.RandomState(seed=seed).permutation(len(eval_dataset)).tolist()

        if(dataset_name in ["mnli", "qnli", "qqp"]):
            num_eval_data = 1000
        elif(dataset_name in ["sst2", "cola", "stsb", "mrpc", "rte"]):
            num_eval_data = int(len(eval_dataset)/2)  

        if(dataset_name in ["cola"]):
            test_dataset = Subset(dataset=eval_dataset, indices=permuted_indices[:num_eval_data])
            eval_dataset = Subset(dataset=eval_dataset, indices=permuted_indices[num_eval_data:])
        else:   
            test_dataset = Subset(dataset=eval_dataset, indices=permuted_indices[num_eval_data:])
            eval_dataset = Subset(dataset=eval_dataset, indices=permuted_indices[:num_eval_data])

        num_labels = glue_data_num_labels_map[dataset_name]
        
        return train_dataset, eval_dataset, test_dataset, num_labels


def load_glue_data_t5(tokenizer, dataset_name: str, max_seq_length: int = 256, seed=42, max_label_seq_length=5, model_type="enc_dec"):
        main_dir = os.path.dirname(os.path.abspath(__file__))
        print(main_dir)
        if(model_type=="t5-base"):
            dataset = load_from_disk(
                os.path.join(
                    main_dir,
                    "data/glue_t5_base",
                    dataset_name
                ))
            max_seq_length=256

        sentence1_key, sentence2_key = glue_data_keys_map[dataset_name]
        dataset = dataset.map(lambda examples: tokenization(
                                    examples=examples,
                                    tokenizer=tokenizer,
                                    max_seq_length=max_seq_length,
                                    max_label_seq_length=max_label_seq_length,
                                    sentence1_key=sentence1_key,
                                    sentence2_key=sentence2_key,
                                    model_type = model_type,
                                    dataset_name = dataset_name
                                    ),
                              batched=True,
                              )
        if(sentence2_key):
            dataset = dataset.remove_columns([sentence1_key, sentence2_key, "idx", "label"])
        else:
            dataset = dataset.remove_columns([sentence1_key, "idx", "label"])

        train_dataset = dataset["train"]
        permuted_indices = np.random.RandomState(seed=seed).permutation(len(train_dataset)).tolist()
        eval_dataset = dataset["validation_matched"] if dataset_name == "mnli" else dataset["validation"]

        if(dataset_name in ["mnli", "qnli", "qqp", "sst2"]):
            num_eval_data = 1000
            test_dataset = eval_dataset
            eval_dataset = Subset(dataset=train_dataset, indices=permuted_indices[:num_eval_data])
            train_dataset = Subset(dataset=train_dataset, indices=permuted_indices[num_eval_data:])
            
        elif(dataset_name in ["cola", "stsb", "mrpc", "rte"]):
            permuted_indices = np.random.RandomState(seed=seed).permutation(len(eval_dataset)).tolist()
            num_eval_data = int(len(eval_dataset)/2)    
            test_dataset = Subset(dataset=eval_dataset, indices=permuted_indices[num_eval_data:])
            eval_dataset = Subset(dataset=eval_dataset, indices=permuted_indices[:num_eval_data])

        num_labels = glue_data_num_labels_map[dataset_name]
        
        return train_dataset, eval_dataset, test_dataset, num_labels

# detailed function
def tokenization(examples, tokenizer, max_seq_length, sentence1_key, sentence2_key, max_label_seq_length, model_type, dataset_name):
    output = tokenizer(
        text=examples[sentence1_key],
        text_pair=examples[sentence2_key] if sentence2_key else None,
        max_length=max_seq_length, 
        truncation=True
        )
    input_ids = output.input_ids
    attention_mask = output.attention_mask
    labels = examples["label"]
    if(dataset_name=="stsb" and model_type in ["t5-base", "dec"]):
         labels = [round(label, 1) for label in labels]      
    if(model_type in ["t5-base", "dec"]):
        labels = [str(label) for label in labels] 
        labels = tokenizer(
                    text=labels,
                    max_length=max_label_seq_length,
                    padding=True,
                    truncation=True
                    ).input_ids 
             
    return {
         "input_ids" : input_ids,
         "attention_mask" : attention_mask,
         "labels" : labels
    }

def load_e2e_data(tokenizer, max_seq_length: int = 64, max_label_seq_length=100):
    main_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = load_from_disk(
        os.path.join(
            main_dir,
            "data/e2e_nlg"
            )
        )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    
    train_dataset = train_dataset.map(lambda examples: tokenization_generation(
                                    examples=examples,
                                    tokenizer=tokenizer,
                                    max_seq_length=max_seq_length,
                                    max_label_seq_length=max_label_seq_length
                                    ),
                              batched=True,
                              )

    train_dataset = train_dataset.remove_columns(["meaning_representation", "human_reference"])
    eval_dataset = eval_dataset.map(lambda examples: tokenization_generation(
                                    examples=examples,
                                    tokenizer=tokenizer,
                                    max_seq_length=max_seq_length,
                                    max_label_seq_length=max_label_seq_length
                                    ),
                              batched=True,
                              )
    eval_dataset = eval_dataset.remove_columns(["meaning_representation", "human_reference"])

    test_dataset = test_dataset.map(lambda examples: tokenization_generation_test(
                                    examples=examples,
                                    tokenizer=tokenizer,
                                    max_seq_length=max_seq_length,
                                    max_label_seq_length=max_label_seq_length
                                    ),
                              batched=True,
                              )
    test_dataset = test_dataset.remove_columns(["meaning_representation", "human_reference"])
   
    return train_dataset, eval_dataset, test_dataset


def tokenization_generation(examples, tokenizer, max_seq_length, max_label_seq_length):
    max_length = max_seq_length+max_label_seq_length        
    bs = len(examples["meaning_representation"])

    text = examples["meaning_representation"]
    text = [t + tokenizer.eos_token for t in text]                   
    test_ids = tokenizer(text = text, max_length=max_seq_length, truncation=True).input_ids                  
    
    pad_id = tokenizer.pad_token_id
    labels = examples["human_reference"]
    labels = [l + tokenizer.eos_token for l in labels]
    target_ids = tokenizer(
        text = labels, 
        max_length=max_label_seq_length, 
        truncation=True
        ).input_ids
    input_ids = [test_ids[i] + target_ids[i] for i in range(bs)]                                     

    target_ids = [[pad_id for _ in range(len(test_ids[i]))]+ target_ids[i] for i in range(bs)]              
    target_ids  = [target_ids[i] + [pad_id for _ in range(max_length-len(input_ids[i]))] for i in range(bs)]
    input_ids = [input_ids[i] + [pad_id for _ in range(max_length-len(input_ids[i]))] for i in range(bs)]
             
    return {
         "input_ids" : input_ids,
         "labels" : target_ids
    }


def tokenization_generation_test(examples, tokenizer, max_seq_length, max_label_seq_length):
    text = examples["meaning_representation"]
    text = [t + tokenizer.eos_token for t in text]                    
    input_ids = tokenizer(text = text, max_length=max_seq_length, truncation=True).input_ids        
            
    labels = examples["human_reference"]
    target_ids = tokenizer(
        text = labels, 
        max_length=max_label_seq_length, 
        truncation=True
        ).input_ids
    
    return {
         "input_ids" : input_ids,
         "labels" : target_ids
    }
