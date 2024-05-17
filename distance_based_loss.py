import json
from datetime import datetime
import os
import logging

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, zeros
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import get_cosine_schedule_with_warmup

from datasets import load_from_disk

from unsloth import FastLanguageModel

# read huggingface dataset from local
# read the tokenized data
tokenized_train_dataset = load_from_disk('./data/gemma_chat_train_predict_emb_task_fixed_empty_string_filter_tokenized')
tokenized_eval_dataset = load_from_disk('./data/gemma_chat_eval_predict_emb_task_fixed_empty_string_filter_tokenized')
# tokenized_test_dataset = load_from_disk('./data/gemma_chat_test_predict_emb_task_fixed_empty_string_filter_tokenized')

# take first 10 of train, first 5 of eval, first 5 of test for quick testing
tokenized_train_dataset = tokenized_train_dataset.select(range(12))
tokenized_eval_dataset = tokenized_eval_dataset.select(range(4))

hyper_params = {
    # Model hyperparameters
    "max_seq_length": 4096, # 8192 | Choose any! We auto support RoPE Scaling internally!
    "dtype": None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    "load_in_4bit": True, # Use 4bit quantization to reduce memory usage. Can be False.,
    # "model_name": "unsloth/gemma-2b-it-bnb-4bit",
    "model_name": "outputs/model_05152024_105402_merged_16bit",
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
    "dataset_train_path": "./data/gemma_chat_train_no_user_intention_fixed_empty_string_filter",
    "dataset_eval_path": "./data/gemma_chat_eval_no_user_intention_fixed_empty_string_filter",
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
model, tokenizer= FastLanguageModel.from_pretrained(
    model_name = hyper_params["model_name"], # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = hyper_params["max_seq_length"],
    dtype = hyper_params["dtype"],
    load_in_4bit = hyper_params["load_in_4bit"],
)

# Assume `model` is your pre-trained model
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Define a custom module with multi-head attention
class MultiheadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, internal_dim, output_dim, num_heads):
        super(MultiheadAttentionLayer, self).__init__()
        self.input_linear = nn.Linear(embed_dim, internal_dim)
        self.gelu_1 = nn.GELU()
        self.multihead_attn = nn.MultiheadAttention(internal_dim, num_heads)
        self.layer_norm = nn.LayerNorm(internal_dim)
        # self.dropout = nn.Dropout(0.1)
        self.output_linear = nn.Linear(internal_dim, output_dim)
        # self.gelu = nn.GELU()
        # self.out_proj = nn.Linear(internal_dim, output_dim)

    def forward(self, x):
        # Assuming `x` shape is (seq_len, batch_size, embed_dim)
        x = self.input_linear(x)
        x = self.gelu_1(x)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.layer_norm(attn_output + x)
        # attn_output = self.dropout(attn_output)

        x = self.output_linear(attn_output)

        return x

# Replace the last layer with the new multi-head attention layer
embed_dim = 2048
internal_dim = 1024
output_dim = 768
num_heads = 4  # Choose the number of attention heads
model.lm_head = MultiheadAttentionLayer(embed_dim, internal_dim, output_dim, num_heads).to(model.device)

# Ensure the new layer's parameters are trainable
for param in model.lm_head.parameters():
    param.requires_grad = True
    
# define customized loss function
# 1. embedding distance loss with L2 regularization
def embedding_distance_loss_with_l2(predicted_emb, ground_truth_emb, model, l2_lambda=0.01):
    # cosine similarity loss
    cosine_sim = F.cosine_similarity(predicted_emb, ground_truth_emb, dim=-1)
    # converting similarity to a loss (minimizing negative similarity)
    cosine_sim_loss = 1 - cosine_sim.mean()
    # L2 regularization
    l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
    # total loss
    loss = cosine_sim_loss + l2_lambda * l2_reg
    
    return loss

# 2. triplet loss
def triplet_loss(anchor, positive, negative, margin=1.0):
    # compute distances
    pos_dist = F.cosine_embedding_loss(anchor, positive, torch.tensor(1.0))
    neg_dist = F.cosine_embedding_loss(anchor, negative, torch.tensor(-1.0))
    # compute triplet loss
    loss = F.relu(pos_dist - neg_dist + margin)
    
    return loss.mean()

def collate_fn(batch):
    inputs = [item['input'][0] for item in batch]
    ground_truth_embs = [item['output'] for item in batch]

    # Pad sequences to the same length
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_ground_truth_embs = pad_sequence(ground_truth_embs, batch_first=True, padding_value=0)
    # len_to_be_padded = hyper_params["max_seq_length"] - len(inputs[0])
    # padded_inputs = torch.nn.functional.pad(tensor, (0, len_to_be_padded), "constant", 0)

    return padded_inputs, padded_ground_truth_embs

train_loader = DataLoader(tokenized_train_dataset.with_format("torch"), batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(tokenized_eval_dataset.with_format("torch"), batch_size=4, shuffle=False, collate_fn=collate_fn)

device = model.device # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Scheduler with warmup
num_train_steps_per_epoch = len(train_loader)
num_epochs = 1
total_train_steps = num_train_steps_per_epoch * num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=25, num_training_steps=total_train_steps, num_cycles=2)

l2_lambda = 0.01  # L2 regularization weight

# Evaluation function
def evaluate(model, val_loader, device, l2_lambda):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, ground_truth_embs in tqdm(val_loader, desc="Evaluating", leave=False):
            input_tensors = inputs.to(device)
            ground_truth_embs = ground_truth_embs.to(device)
            
            predicted_embs = model(input_tensors)
            loss = embedding_distance_loss_with_l2(predicted_embs.logits, ground_truth_embs, model, l2_lambda)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

# Training loop
num_eval_log_steps = 100

# Log history dictionary
log_history = {
    "train_loss": [],
    "eval_loss": []
}

# # List to keep track of saved checkpoints
# saved_checkpoints = []
# max_checkpoints = 5

# Single checkpoint path
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_folder = f"outputs" + f"/distance_based_loss_checkpoint_{current_time}"
# make the directory if it doesn't exist
os.makedirs(model_save_folder, exist_ok=True)
model_save_path = model_save_folder + f"/best_model_checkpoint.pth"

# Set up logging
log_filename = model_save_folder + "/training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Remove this line if you don't want to print to console
    ]
)

# Early stopping parameters
patience = 10  # Number of evaluation steps to wait for improvement
best_val_loss = float('inf')
patience_counter = 0

global_step = 0
for epoch in range(num_epochs):
    model.train()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    for step, (inputs, ground_truth_embs) in progress_bar:  # each batch contains a tuple (inputs, ground_truth_emb)
        input_tensors = inputs.to(device)
        ground_truth_emb = ground_truth_embs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predicted_emb = model(input_tensors)
        
        # Compute loss
        loss = embedding_distance_loss_with_l2(predicted_emb.logits, ground_truth_emb, model, l2_lambda)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        log_history["train_loss"].append((global_step, loss.item()))
        logging.info(f'Step [{global_step}/{total_train_steps}], Train Loss: {loss.item():.4f}')
        progress_bar.set_postfix(train_loss=loss.item())
        
        # Log every num_eval_log_steps steps
        if global_step % num_eval_log_steps == 0:
            avg_val_loss = evaluate(model, val_loader, device, l2_lambda)
            log_history["eval_loss"].append((global_step, avg_val_loss))
            logging.info(f'Step [{global_step}/{total_train_steps}], Val Loss: {avg_val_loss:.4f}')

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # # Save the model and log history
                # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                # model_save_path = f"outputs/best_model_checkpoint_epoch_{epoch}_step_{global_step}_{current_time}.pth"
                # torch.save({
                #     'model_state_dict': model.state_dict(),
                #     'log_history': log_history
                # }, model_save_path)
                # logging.info(f"Model and log history saved at step {global_step} (epoch {epoch}) with validation loss {avg_val_loss:.4f}")

                # # Track saved checkpoints and remove the oldest if necessary
                # saved_checkpoints.append(model_save_path)
                # if len(saved_checkpoints) > max_checkpoints:
                #     os.remove(saved_checkpoints.pop(0))
                
                # Save the model and log history, overiding the previous checkpoint
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'log_history': log_history
                }, model_save_path)
                logging.info(f"Model and log history saved at step {global_step} (epoch {epoch}) with validation loss {avg_val_loss:.4f}")                
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break
            
        global_step += 1
    
    # Step the scheduler
    scheduler.step()
    
    avg_train_loss = sum(loss for _, loss in log_history["train_loss"][-num_train_steps_per_epoch:]) / num_train_steps_per_epoch
    avg_val_loss = evaluate(model, val_loader, device, l2_lambda)
    
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# save the final model
torch.save({
    'model_state_dict': model.state_dict(),
    'log_history': log_history
}, model_save_folder + f"/final_model_checkpoint.pth")

logging.info("Training completed.")


# -------------------------------------------------
# model loading example:
# # Path to the saved checkpoint
# model_save_path = "best_model_checkpoint.pth"

# # Load the checkpoint
# checkpoint = torch.load(model_save_path)

# # Retrieve the saved log history
# log_history = checkpoint['log_history']

# # Restore the model state
# model.load_state_dict(checkpoint['model_state_dict'])

# logging.info("Model and log history loaded successfully.")

# # Now you can access the log history
# train_loss_history = log_history['train_loss']
# eval_loss_history = log_history['eval_loss']

# logging.info("Train loss history:", train_loss_history)
# logging.info("Eval loss history:", eval_loss_history)