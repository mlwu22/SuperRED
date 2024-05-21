import os
import logging
from transformers import AutoTokenizer
import torch

TASK_TYPE_TOMODEL_PATH = {
    "llama2-7B": "/data/wml/cache/model/llama2-7B"
}

STRING_TO_WEIGHT_TYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}

"""设置log"""
def set_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    final_path = f"{log_dir}/log"
    logger = logging.getLogger()
    logger.setLevel('INFO')
    control = logging.StreamHandler() 
    control.setLevel('INFO')
    fhlr = logging.FileHandler(final_path)
    logger.addHandler(fhlr)
    logger.addHandler(control)
    return logger

"""设置log_dir,save_dir,record_dir"""
def set_dir(meta_dir, args):
    output_dir = f"{meta_dir}/output/[task_type]/[model_type]/lr_[lr]_epoch_[epoch]_weightType_[weightType]/{str(args.seed)}/[Substitute_meta]"
    output_dir = output_dir.replace("[model_type]", args.model_type)
    output_dir = output_dir.replace("[lr]", str(args.lr))
    output_dir = output_dir.replace("[epoch]", str(args.num_train_epochs))
    output_dir = output_dir.replace("[weightType]", str(args.weight_type))
    output_dir = output_dir.replace("[task_type]", str(args.task_type))
    log_dir = output_dir.replace("[Substitute_meta]", "logdir")
    save_dir = output_dir.replace("[Substitute_meta]", "save_models")
    record_dir = output_dir.replace("[Substitute_meta]", "record")
    return log_dir, save_dir, record_dir

"""设置pad_token"""
def add_special_token(tokenizer: AutoTokenizer, model):
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    if need_resize:
        model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model