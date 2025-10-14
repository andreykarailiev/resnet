from neptune_scale import Run
from unsloth import FastLanguageModel, prepare_model_for_training

from dataclasses import dataclass
from pathlib import Path
from jaxtyping import Float, Int

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Placeholder simple MLP stack
class MemoryMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int = 512):
        super().__init__()
        self.MLPStack = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, hidden_size)
        )

    def forward(self, x):
        return x + self.MLPStack(x)

# Load pre-trained small LLM model
model, tokenizer = FastLanguageModel.from_pretrained(
        model_name =  "unsloth/SmolLM-1.7B",
        max_seq_length = 2048,
        dtype = None,        # automatic dtype selection (e.g. bf16/FP16)
        load_in_4bit = True, # quantized loading if supported
    )

# Attach it directly to the Unsloth model
hidden_size = model.model.embed_tokens.embedding_dim  # or model.config.hidden_size
model.memory_mlp = MemoryMLP(hidden_size).to(model.device)

# Modify forward() to use it (simple patch)
old_forward = model.forward

def new_forward(**kwargs):
    outputs = old_forward(**kwargs)
    hidden_states = outputs.last_hidden_state
    # Apply your MLP on the final hidden states
    hidden_states = model.memory_mlp(hidden_states)
    # Replace output (preserve logits computation)
    outputs.last_hidden_state = hidden_states
    return outputs

model.forward = new_forward



if __name__ == "__main__":
    print(hidden_size)