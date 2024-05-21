from data.data_process import get_train_and_eval
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import Trainer, TrainingArguments
import argparse
from utils import TASK_TYPE_TOMODEL_PATH
from utils import set_log, set_dir, add_special_token
from peft import REDConfig, get_peft_model
import os
from transformers import DataCollatorForSeq2Seq


parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=5e-3)
parser.add_argument("--weight_decay", default=0.0)
parser.add_argument("--model_type", default="llama2-7B")
parser.add_argument("--warmup_ratio", default=0)
parser.add_argument("--seed", default=42)
parser.add_argument("--task_type", default="math")
parser.add_argument("--per_device_train_batch_size", default=16)
parser.add_argument("--per_device_eval_batch_size", default=16)
parser.add_argument("--gradient_accumulation_steps", default=1)
parser.add_argument("--num_train_epochs", default=12)
parser.add_argument("--meta_dir", default=f"./test")
parser.add_argument("--weight_type", default="float32")
parser.add_argument("--layer_type", default="all")
# parser.add_argument("--edit_position", default="FFN")
# parser.add_argument("--trainable_layers", default="All")
args = parser.parse_args()

cur_path = os.path.dirname(os.path.abspath(__file__))
log_dir, save_dir, record_dir = set_dir(cur_path, args)


model = LlamaForCausalLM.from_pretrained(TASK_TYPE_TOMODEL_PATH[args.model_type], device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(TASK_TYPE_TOMODEL_PATH[args.model_type])
tokenizer, model = add_special_token(tokenizer, model)

train_dataset, eval_dataset = get_train_and_eval(
    tokenizer=tokenizer,
    task_type="math"
)

peft_config = REDConfig(
    inference_mode=False,
    layer_type="bias"
)

model = get_peft_model(model=model, peft_config=peft_config)
logger = set_log(log_dir)

logger.info(
    f"Args: \n"
    f"lr: {str(args.lr)} \n"
    f"weight_decay: {str(args.weight_decay)} \n"
    f"model_type: {str(args.model_type)} \n"
    f"warmup_ratio: {str(args.warmup_ratio)} \n"
    f"seed: {str(args.seed)} \n"
    f"per_device_train_batch_size: {str(args.per_device_train_batch_size)} \n"
    f"per_device_eval_batch_size: {str(args.per_device_eval_batch_size)} \n"
    f"gradient_accumulation_steps: {str(args.gradient_accumulation_steps)} \n"
    f"num_train_epochs: {str(args.num_train_epochs)} \n"
    f"save_dir: {str(save_dir)} \n"
    f"layer_type: {str(args.layer_type)} \n"
)

training_args = TrainingArguments(
    output_dir = save_dir,
    num_train_epochs = int(args.num_train_epochs),
    per_device_train_batch_size = int(args.per_device_train_batch_size),
    per_device_eval_batch_size = int(args.per_device_eval_batch_size),
    gradient_accumulation_steps = int(args.gradient_accumulation_steps),
    evaluation_strategy = "steps",
    eval_steps = 1000,
    save_steps = 1000,
    save_strategy = "steps",
    load_best_model_at_end = True,
    metric_for_best_model = "acc",
    logging_strategy="steps",
    logging_steps = 1,
    learning_rate = float(args.lr),
    warmup_ratio =  float(args.warmup_ratio),
    weight_decay = float(args.weight_decay),
    seed = int(args.seed),
    report_to="none"
)

data_collator_fn = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding="longest"
)

trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator_fn,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = training_args
)

trainer.train()