# read huggingface dataset from local
from datasets import load_from_disk

# read the tokenized data
tokenized_train_dataset = load_from_disk('./data/gemma_chat_train_predict_emb_task_fixed_empty_string_filter_tokenized')
tokenized_eval_dataset = load_from_disk('./data/gemma_chat_eval_predict_emb_task_fixed_empty_string_filter_tokenized')
tokenized_test_dataset = load_from_disk('./data/gemma_chat_test_predict_emb_task_fixed_empty_string_filter_tokenized')

import json
from datetime import datetime

import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel

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

import torch
import torch.nn as nn

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
import torch
import torch.nn.functional as F

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

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs = [item['input'][0] for item in batch]
    ground_truth_embs = [item['output'] for item in batch]

    # Pad sequences to the same length
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_ground_truth_embs = pad_sequence(ground_truth_embs, batch_first=True, padding_value=0)
    # len_to_be_padded = hyper_params["max_seq_length"] - len(inputs[0])
    # padded_inputs = torch.nn.functional.pad(tensor, (0, len_to_be_padded), "constant", 0)

    return padded_inputs, padded_ground_truth_embs

import torch
from torch import nn, optim, zeros
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

train_loader = DataLoader(tokenized_train_dataset.with_format("torch"), batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(tokenized_eval_dataset.with_format("torch"), batch_size=4, shuffle=False, collate_fn=collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        for inputs, ground_truth_embs in val_loader:
            input_tensors = inputs.to(device)
            ground_truth_embs = ground_truth_embs.to(device)
            
            predicted_embs = model(input_tensors)
            loss = embedding_distance_loss_with_l2(predicted_embs, ground_truth_embs, model, l2_lambda)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, ground_truth_embs in train_loader:  # each batch contains a tuple (inputs, ground_truth_emb)
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
        
        total_loss += loss.item()
    
    # Step the scheduler
    scheduler.step()
    
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = evaluate(model, val_loader, device, l2_lambda)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

print("Training completed.")