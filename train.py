from data.data_process import get_train_and_eval
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import Trainer, TrainingArguments
import argparse
from utils import TASK_TYPE_TOMODEL_PATH, STRING_TO_WEIGHT_TYPE
from utils import set_log, set_dir, add_special_token
from utils import compute_custom_metric_for_math
from utils import CustomTrainer
from peft import REDConfig, get_peft_model
import os
from transformers import DataCollatorForSeq2Seq


parser = argparse.ArgumentParser()
# training args
parser.add_argument("--lr", default=1e-3)
parser.add_argument("--weight_decay", default=0.0)
parser.add_argument("--model_type", default="llama2-7B")
parser.add_argument("--warmup_ratio", default=0)
parser.add_argument("--per_device_train_batch_size", default=1)
parser.add_argument("--per_device_eval_batch_size", default=1)
parser.add_argument("--gradient_accumulation_steps", default=1)
parser.add_argument("--num_train_epochs", default=12)
parser.add_argument("--save_strategy",default="steps")
parser.add_argument("--save_steps",default=1000)

parser.add_argument("--seed", default=42)
parser.add_argument("--task_type", default="math")

# parser.add_argument("--meta_dir", default=f"./test")
parser.add_argument("--weight_type", default="bfloat16")
parser.add_argument("--layer_type", default="all")

args = parser.parse_args()

cur_path = os.path.dirname(os.path.abspath(__file__))
log_dir, save_dir, record_dir = set_dir(cur_path, args)


model = LlamaForCausalLM.from_pretrained(TASK_TYPE_TOMODEL_PATH[args.model_type], device_map="auto", torch_dtype=STRING_TO_WEIGHT_TYPE[args.weight_type])
tokenizer = AutoTokenizer.from_pretrained(TASK_TYPE_TOMODEL_PATH[args.model_type])
tokenizer, model = add_special_token(tokenizer, model)

train_dataset, eval_dataset = get_train_and_eval(
    tokenizer=tokenizer,
    task_type=args.task_type
)
print(train_dataset[0])
peft_config = REDConfig(
    inference_mode=False,
    target_modules = ["down_proj"],
    feedforward_modules = ["down_proj"],
    layer_type=args.layer_type
)

model = get_peft_model(model=model, peft_config=peft_config)
print(model)
logger = set_log(log_dir)

logger.info(
    f"Args: \n"
    "\Training Args\n"
    f"lr: {str(args.lr)} \n"
    f"weight_decay: {str(args.weight_decay)} \n"
    f"model_type: {str(args.model_type)} \n"
    f"warmup_ratio: {str(args.warmup_ratio)} \n"
    f"seed: {str(args.seed)} \n"
    f"per_device_train_batch_size: {str(args.per_device_train_batch_size)} \n"
    f"per_device_eval_batch_size: {str(args.per_device_eval_batch_size)} \n"
    f"gradient_accumulation_steps: {str(args.gradient_accumulation_steps)} \n"
    f"save_strategy: {str(args.save_strategy)} \n"
    f"save_steps: {str(args.save_steps)} \n"
    f"num_train_epochs: {str(args.num_train_epochs)} \n"
    f"save_dir: {str(save_dir)} \n"

    "\nModel Args\n"
    f"layer_type: {str(args.layer_type)} \n"
    f"task_type: {str(args.task_type)} \n"
    f"weight_type: {str(args.weight_type)} \n"
)

training_args = TrainingArguments(
    output_dir = save_dir,
    num_train_epochs = int(args.num_train_epochs),
    per_device_train_batch_size = int(args.per_device_train_batch_size),
    per_device_eval_batch_size = int(args.per_device_eval_batch_size),
    gradient_accumulation_steps = int(args.gradient_accumulation_steps),
    evaluation_strategy = "steps" if args.task_type == "math" else None,
    eval_steps = 1000 if args.task_type == "math" else None,
    save_strategy = args.save_strategy,
    save_steps = int(args.save_steps) if args.save_strategy == "steps" else None,
    load_best_model_at_end = True if args.task_type == "math" else False,
    metric_for_best_model = "eval_acc" if args.task_type == "math" else None,
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

if(args.task_type=="math"):
    generation_args = {
        "max_new_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.75,
        "top_k": 40,
        "num_beams": 4,
        "do_sample": True,
        "trigger": "### Response:"
    }
else:
    generation_args = {
        "max_new_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.75,
        "top_k": 40,
        "num_beams": 4,
        "do_sample": True,
        "trigger": "\n\nAssistant: "
    }

trainer = CustomTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator_fn,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = training_args,
    compute_metrics=compute_custom_metric_for_math,
    generation_args = generation_args
)

trainer.train()