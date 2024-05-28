from transformers import AutoModel, AutoTokenizer


model_path = "/data/lwh/models/llama2/7B/7B"
model = AutoModel.from_pretrained(model_path)
tokenizer  = AutoTokenizer.from_pretrained(model_path)
input_text = "I love you"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model(input_ids)
print(output.last_hidden_state[-1][-1][:10])

from peft import  get_peft_model, IA3Config, LoraConfig, VeraConfig, REDConfig, PeftModel

# peft_config = IA3Config(
#     inference_mode=False
# )

peft_config = REDConfig(
    inference_mode=False,
    layer_type="bias"
)

# peft_config = LoraConfig(
#     inference_mode=False, 
#     r=8, 
#     lora_alpha=32, 
#     lora_dropout=0.1
# )

peft_model = get_peft_model(model=model, peft_config=peft_config)
print(peft_model)



output = peft_model(input_ids)
print(output.last_hidden_state[-1][-1][:10])
peft_model.save_pretrained("./test")

model = AutoModel.from_pretrained(model_path)
peft_model = peft_model.from_pretrained(model, "./test")
# print(peft_model)

output = peft_model(input_ids)
print(output.last_hidden_state[-1][-1][:10])