import os
import logging
from transformers import AutoTokenizer
import torch
from transformers import Trainer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import re
from transformers.trainer_utils import EvalLoopOutput

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

"""数学任务计算metric"""
def compute_custom_metric_for_math(generation_list, answer_list, tokenizer: AutoTokenizer, trigger_token:str):
    count = 0
    length = len(generation_list)
    generation_list = tokenizer.batch_decode(generation_list, skip_special_tokens=True)
    generation_list = [g.split(trigger_token)[-1] for g in generation_list]
    generation_list = [extract_answer_letter(g) for g in generation_list]
    generation_list = [g.strip() for g in generation_list]
    answer_list = [a.strip() for a in answer_list]
    for i in range(length):
        if generation_list[i]==answer_list[i]:
            count+=1
    print(generation_list[:10], answer_list[:10])
    return {'eval_acc': count/length}

"""抽取答案"""
def extract_answer_letter(sentence: str) -> str:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py

    Note that it becomes ambiguous whether to extract the
    first letter or the last letter. Either way may lead
    to inaccurately assess the model performance. 

    We choose to follow the LLM-Adaptor repo, but leave this note
    for future research to explore the impact of this.
    """
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''

class CustomTrainer(Trainer):
    def __init__(self, generation_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger = generation_args.pop("trigger")
        self.generation_args = generation_args

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix=None):

        args = self.args

        output = {}
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        # batch_size = self.args.eval_batch_size
        model.eval()
        # eval_dataset = getattr(dataloader, "dataset", None)


        generation_list = []
        answer_list = []
        
        for step, inputs in tqdm(enumerate(dataloader)):
            generation_ids = model.generate(inputs["input_ids"].to(args.device), **self.generation_args)
            for a in inputs["answers"]:
                answer_list.append(a)
            for g in generation_ids:
                generation_list.append(g.cpu())
        
        metric_results = self.compute_metrics(generation_list, answer_list, self.tokenizer, self.trigger)
        
        return EvalLoopOutput(
            predictions = generation_list,
            label_ids = answer_list,
            metrics = metric_results,
            num_samples = len(generation_list)
        )
    

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        generation_ids = model.generate(inputs["input_ids"], **self.generation_args)
        return (generation_ids, inputs["answers"])
            
    
    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        return DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, collate_fn=self.custom_collate_fn)
    
    def custom_collate_fn(self, batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        labels = [item["answer"] for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "answers": labels
            }