from datasets import load_from_disk, load_dataset
from datasets import disable_caching, Dataset
from transformers import AutoTokenizer, GPTNeoXTokenizerFast
from typing import List
import numpy as np
from copy import deepcopy
import os
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Subset
import json

disable_caching()

TASK_PROMPT_TEMPLATE_MAP= \
{
    "commonsense": 
        "%s\n",

    "math": 
        """Below is an instruction that describes a task. Write a response that appropriately completes the request.
        ### Instruction: %s

        ### Response:
<<<<<<< HEAD
        """,
    
    "instruction_following":
        "Human: %s",

    "code":
        "Question: %s"

=======
        """
>>>>>>> d0975e6c6c00ac2818bc254467907ac0b12a25c7
}

TRIGGER_TOKEN_MAP= \
{
    "commonsense": 
        "the correct answer is ",

    "math": 
<<<<<<< HEAD
        "### Response:",
    
    "instruction_following":
        "\n\nAssistant: ",

    "code":
        "\n\nAnswer: ",

=======
        "### Response:"
>>>>>>> d0975e6c6c00ac2818bc254467907ac0b12a25c7
}




# commonsense input(prompt+response) >= 384 的只有 238 条数据
# math input(prompt+response) >= 256 的只有两条
<<<<<<< HEAD
# instruction_following input(prompt+response) >= 2048 的只有237条
# instruction_following input(prompt+response) >= 2048 的只有16条
MAX_INPUT_LENGTH_MAP = {
    "commonsense": 384,
    "math": 256,
    "instruction_following":2048,
    "code":2048,
=======
MAX_INPUT_LENGTH_MAP = {
    "commonsense": 384,
    "math": 256
>>>>>>> d0975e6c6c00ac2818bc254467907ac0b12a25c7
}

IGNORE_INDEX = -100


"""拆分验证集并过滤掉长度不符的token"""
def split_eval_data(tokenizer, seed=42, num_eval_data=1000, task_type="commonsense"):
    if(task_type=="commonsense"):
<<<<<<< HEAD
        data_path = "/home/lwh/code/SuperRED/data/dataset/train/commonsense_170k/train.json"
    elif(task_type=="math"):
        data_path = "/home/lwh/code/SuperRED/data/dataset/train/math_10k/train.json"
    elif(task_type=="instruction_following"):
        data_path = "/home/lwh/code/SuperRED/data/dataset/train/instruction_following_60k/train.json"
    elif(task_type=="code"):
        data_path = "/home/lwh/code/SuperRED/data/dataset/train/code_12k/train.json"
=======
        data_path = "/data/wml/SuperRED/data/dataset/train/commonsense_170k/train.json"
    elif(task_type=="math"):
        data_path = "/data/wml/SuperRED/data/dataset/train/math_10k/train.json"
>>>>>>> d0975e6c6c00ac2818bc254467907ac0b12a25c7
    train_dataset = load_dataset("json", data_files = data_path, split = "train")

    train_dataset = train_dataset.map(
        lambda example: comput_length(
        example = example,
        tokenizer = tokenizer,
        task_type=task_type
        )
    )
    print("Number of examples before filtering: ", len(train_dataset))
    train_dataset = train_dataset.filter(lambda x: x["input_length"] <= MAX_INPUT_LENGTH_MAP[task_type])
    print("Number of examples after filtering: ", len(train_dataset))

    train_dataset = train_dataset.map(
        lambda example: tokenize(
        example = example,
        tokenizer = tokenizer,
        split = "train"
        )
    )
    train_dataset=train_dataset.remove_columns(["input_ids", "labels"])

    permuted_indices = np.random.RandomState(seed=seed).permutation(len(train_dataset)).tolist()
    eval_indices, train_indices = permuted_indices[:num_eval_data], permuted_indices[num_eval_data:]
    eval_dataset_list, train_dataset_list = [], []

    for i in eval_indices:
        eval_example = train_dataset[i]
        eval_example["instance_id"] = i
        eval_dataset_list.append(eval_example)

    for i in train_indices:
        train_example = train_dataset[i]
        train_example["instance_id"] = i
        train_dataset_list.append(train_example)

    print("Number of training examples: ", len(train_dataset_list))
    print("Number of evaluation examples: ", len(eval_dataset_list))

    output_path = "/".join(data_path.split("/")[:-1])
    with open(os.path.join(output_path, "train_split.json"), "w", encoding="utf-8")as f:
        json.dump(train_dataset_list, f, indent=4)
    with open(os.path.join(output_path, "eval_split.json"), "w", encoding="utf-8")as f:
        json.dump(eval_dataset_list, f, indent=4)
    
"""计算input的长度"""
def comput_length(example, tokenizer, task_type):
    task_prompt_template = TASK_PROMPT_TEMPLATE_MAP[task_type]
    trigger_tokens = TRIGGER_TOKEN_MAP[task_type]
    base_prompt = task_prompt_template % (example['instruction'])
    base_input = base_prompt + trigger_tokens + example["answer"] + tokenizer.eos_token
    example["prompt_str"] = base_prompt
    example["input_str"] = base_input
    example["input_length"] = len(tokenizer(base_input, return_tensors="pt")["input_ids"][0])
    return example

"""tokenization"""
def tokenize(example, tokenizer, split = "train"):
    prompt_ids = tokenizer(example["prompt_str"], return_tensors="pt")["input_ids"][0]
    input_ids = tokenizer(example["input_str"], return_tensors="pt")["input_ids"][0]
    prompt_length = len(prompt_ids)

    if(split=="train"):
        output_ids = deepcopy(input_ids)
        output_ids[:prompt_length] = IGNORE_INDEX
        example["input_ids"] = input_ids
        example["labels"] = output_ids
    else:
        example["input_ids"] = prompt_ids
    return example

"""获取train,eval"""
def get_train_and_eval(tokenizer, task_type="commonsense"):
    if(task_type=="commonsense"):
<<<<<<< HEAD
        meta_path = "/home/lwh/code/SuperRED/data/dataset/train/commonsense_170k"
    elif(task_type=="math"):
        meta_path = "/home/lwh/code/SuperRED/data/dataset/train/math_10k"
    elif(task_type=="instruction_following"):
        meta_path = "/home/lwh/code/SuperRED/data/dataset/train/instruction_following_60k"
    elif(task_type=="code"):
        meta_path = "/home/lwh/code/SuperRED/data/dataset/train/code_12k"
=======
        meta_path = "/data/wml/SuperRED/data/dataset/train/commonsense_170k"
    elif(task_type=="math"):
        meta_path = "/data/wml/SuperRED/data/dataset/train/math_10k"
>>>>>>> d0975e6c6c00ac2818bc254467907ac0b12a25c7
    train_dataset = load_dataset("json", data_files = os.path.join(meta_path, "train_split.json"), split = "train")
    eval_dataset = load_dataset("json", data_files = os.path.join(meta_path, "eval_split.json"), split = "train")

    train_dataset = train_dataset.map(
        lambda example:tokenize(
            example = example,
            tokenizer = tokenizer,
            split = "train"
<<<<<<< HEAD
        ),
        num_proc=8,
=======
        )
>>>>>>> d0975e6c6c00ac2818bc254467907ac0b12a25c7
    )
    train_dataset = train_dataset.remove_columns(["instance_id", "input", "prompt_str", "answer", "input_str", "output", "input_length", "instruction"])

    # eval_dataset= eval_dataset.select(range(12))
    eval_dataset = eval_dataset.map(
        lambda example:tokenize(
            example = example,
            tokenizer = tokenizer,
            split = "eval"
<<<<<<< HEAD
        ),
        num_proc=8,
=======
        )
>>>>>>> d0975e6c6c00ac2818bc254467907ac0b12a25c7
    )
    # eval_dataset = eval_dataset.remove_columns(["instance_id", "input", "prompt_str", "input_str", "output", "input_length", "instruction"])
    eval_dataset = eval_dataset.remove_columns(["input", "prompt_str", "input_str", "output", "input_length", "instruction"])

    return train_dataset, eval_dataset

"""获取test"""
def get_test(tokenizer, task_type="commonsense"):
    if(task_type=="commonsense"):
<<<<<<< HEAD
        meta_path = "/home/lwh/code/SuperRED/data/dataset/train/commonsense_170k"
    elif(task_type=="math"):
        meta_path = "/home/lwh/code/SuperRED/data/dataset/train/math_10k"
=======
        meta_path = "/data/wml/SuperRED/data/dataset/train/commonsense_170k"
    elif(task_type=="math"):
        meta_path = "/data/wml/SuperRED/data/dataset/train/math_10k"
>>>>>>> d0975e6c6c00ac2818bc254467907ac0b12a25c7
    test_dataset = load_dataset("json", data_files = os.path.join(meta_path, "test.json"), split = "train")
    
    test_dataset = test_dataset.map(
        lambda example:tokenize(
            example = example,
            tokenizer = tokenizer,
            split = "test"
        )
    )
    test_dataset = test_dataset.remove_columns(["instance_id", "input", "prompt_str", "input_str", "output", "input_length", "instruction"])

    return test_dataset
<<<<<<< HEAD
# tokenizer = AutoTokenizer.from_pretrained("/data/lwh/models/llama2/7B/7B")

# train_dataset, eval_dataset = split_eval_data(tokenizer=tokenizer, task_type="math")
=======
# tokenizer = AutoTokenizer.from_pretrained("/data/wml/cache/model/llama2-7B")

# train_dataset, eval_dataset = get_train_and_eval(tokenizer=tokenizer, task_type="math")
>>>>>>> d0975e6c6c00ac2818bc254467907ac0b12a25c7
# print(train_dataset[0])
# print(train_dataset[0].keys())
# print(eval_dataset[0])
# print(eval_dataset[0].keys())
# split_eval_data(tokenizer=tokenizer, seed=42, num_eval_data=500, task_type="math")