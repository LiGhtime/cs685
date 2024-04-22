import json

import torch
from unsloth import FastLanguageModel

max_seq_length = 4096 # 8192 | Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
] # More models at https://huggingface.co/unsloth

model, tokenizer= FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2b-it-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# add lora to model
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# read huggingface dataset from local
from datasets import load_from_disk

dataset_train = load_from_disk('./data/gemma_train')
dataset_eval = load_from_disk('./data/gemma_eval')

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for input, output in zip(inputs, outputs):
        text = "### Input:\n{inputs_holder}\n\n### Response:{outputs_holder}".format(inputs_holder= input, outputs_holder= output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

train_dataset = dataset_train.map(formatting_prompts_func, batched = True,)
# test_dataset = dataset_eval.map(formatting_prompts_func, batched = True,)

# training
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    # eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,
        warmup_steps = 15,
        num_train_epochs = 1,
        # max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 1,
        # eval_accumulation_steps = 1,
        # evaluation_strategy = "steps", # epoch
        # eval_steps = 100,
    ),
)

trainer_stats = trainer.train()

# saving model and loss history
# get current datetime
from datetime import datetime
now = datetime.now()

# name the model
model_name = "model_" + now.strftime("%m%d%Y_%H%M%S")
model_path = "outputs/" + model_name
model.save_pretrained(model_path) # Local saving
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving

# save hyperparameters as a json dict to model_path
hyper_params = {
    "max_seq_length": max_seq_length,
    "dtype": dtype,
    "load_in_4bit": load_in_4bit,
    "model_name": "unsloth/gemma-2b-it-bnb-4bit",
    "r": 8,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 15,
    "num_train_epochs": 1,
    "learning_rate": 1e-4,
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    }

with open(model_path + "/hyperparameters.json", "w") as file:
    json.dump(hyper_params, file, indent=4)
        
# save trainer.state.log_history to model_path
with open(model_path + "/trainer_state_log_history.json", "w") as file:
    json.dump(trainer.state.log_history, file, indent=4)
