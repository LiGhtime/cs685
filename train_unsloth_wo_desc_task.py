import json
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel

hyper_params = {
    # Model hyperparameters
    "max_seq_length": 4096, # 8192 | Choose any! We auto support RoPE Scaling internally!
    "dtype": None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    "load_in_4bit": True, # Use 4bit quantization to reduce memory usage. Can be False.,
    "model_name": "unsloth/gemma-2b-it-bnb-4bit",
    "r": 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # Add more to target more modules
    "lora_alpha": 16,
    "lora_dropout": 0, # Supports any, but = 0 is optimized
    "lora_bias": "none", # Supports any, but = "none" is optimized
    "lora_use_gradient_checkpointing": "unsloth", # True or "unsloth" for very long context
    "lora_random_state": 3407,
    "lora_use_rslora": False, # We support rank stabilized LoRA
    "lora_loftq_config": None, # And LoftQ
    # Training hyperparameters
    "dataset_train_path": "./data/gemma_chat_train_wo_desc_task_fixed_empty_string_filter",
    "dataset_eval_path": "./data/gemma_chat_eval_wo_desc_task_fixed_empty_string_filter",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 25, # will replace num_warmup_steps in lr_scheduler_kwargs
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine_with_restarts",
    "lr_scheduler_kwargs": {"num_cycles": 3}, # "num_warmup_steps" and "num_training_steps" will be added automatically
    "seed": 3407,
}

# load model and tokenizer
# More models at https://huggingface.co/unsloth
model, tokenizer= FastLanguageModel.from_pretrained(
    model_name = hyper_params["model_name"], # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = hyper_params["max_seq_length"],
    dtype = hyper_params["dtype"],
    load_in_4bit = hyper_params["load_in_4bit"],
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# add lora to model
model = FastLanguageModel.get_peft_model(
    model,
    r = hyper_params['r'],
    target_modules = hyper_params['target_modules'],
    lora_alpha = hyper_params['lora_alpha'],
    lora_dropout = hyper_params['lora_dropout'],
    bias = hyper_params['lora_bias'],
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = hyper_params['lora_use_gradient_checkpointing'],
    random_state = hyper_params['lora_random_state'],
    use_rslora = hyper_params['lora_use_rslora'], # We support rank stabilized LoRA
    loftq_config = hyper_params['lora_loftq_config'], # And LoftQ
)

# read huggingface dataset from local
dataset_train = load_from_disk(hyper_params['dataset_train_path'])
# dataset_eval = load_from_disk(hyper_params['dataset_eval_path'])

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

prompt_template = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n{}<end_of_turn>\n"

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for input, output in zip(inputs, outputs):
        # text = "### Input:\n{inputs_holder}\n\n### Response:{outputs_holder}".format(inputs_holder= input, outputs_holder= output) + EOS_TOKEN
        # gemma chat template:
        text = prompt_template.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

train_dataset = dataset_train.map(formatting_prompts_func, batched = True,)
# test_dataset = dataset_eval.map(formatting_prompts_func, batched = True,)

# # take the first 100 of the dataset for a quick testing
# train_dataset = train_dataset.select(range(10))

# training
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    # eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = hyper_params['max_seq_length'],
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = hyper_params['per_device_train_batch_size'],
        gradient_accumulation_steps = hyper_params['gradient_accumulation_steps'],
        warmup_steps = hyper_params['warmup_steps'],
        num_train_epochs = hyper_params['num_train_epochs'],
        # max_steps = 100,
        learning_rate = hyper_params['learning_rate'],
        fp16 = hyper_params['fp16'],
        bf16 = hyper_params['bf16'],
        logging_steps = hyper_params['logging_steps'],
        optim = hyper_params['optim'],
        weight_decay = hyper_params['weight_decay'],
        lr_scheduler_type = hyper_params['lr_scheduler_type'],
        lr_scheduler_kwargs = hyper_params['lr_scheduler_kwargs'],
        seed = hyper_params['seed'],
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
now = datetime.now()

# name the model
model_name = "model_" + now.strftime("%m%d%Y_%H%M%S")
model_path = "outputs/" + model_name
model.save_pretrained(model_path) # Local saving
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving

# save hyperparameters as a json dict to model_path
with open(model_path + "/hyperparameters.json", "w") as file:
    json.dump(hyper_params, file, indent=4)
        
# save trainer.state.log_history to model_path
with open(model_path + "/trainer_state_log_history.json", "w") as file:
    json.dump(trainer.state.log_history, file, indent=4)

# plot trainer.state.log_history, plot training loss and evaluation loss
# Extracting training and eval loss data
train_losses = []
eval_losses = []
steps = []

for entry in trainer.state.log_history:
    if 'loss' in entry:
        train_losses.append((entry['step'], entry['loss']))
    if 'eval_loss' in entry:
        eval_losses.append((entry['step'], entry['eval_loss']))
    if 'train_loss' in entry:
        train_losses.append((entry['step'], entry['train_loss']))

# Sorting values by steps to align the data points correctly
train_losses.sort(key=lambda x: x[0])
eval_losses.sort(key=lambda x: x[0])

# Plotting
plt.figure(figsize=(10, 6))
# Unzipping for plotting
if eval_losses != []:
    train_steps, train_loss_values = zip(*train_losses)
    eval_steps, eval_loss_values = zip(*eval_losses)
    plt.plot(train_steps, train_loss_values, label='Training Loss', marker='o')
    plt.plot(eval_steps, eval_loss_values, label='Evaluation Loss', marker='x')
else:
    train_steps, train_loss_values = zip(*train_losses)
    plt.plot(train_steps, train_loss_values, label='Training Loss', marker='o')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss Over Steps')
plt.legend()
plt.grid(True)
# plt.show()

# save the plot
plt.savefig(model_path + "/loss_plot.png")
