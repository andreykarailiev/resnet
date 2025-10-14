import einops
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import unsloth
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm
import neptune_scale as Neptune

from transformers import TrainerCallback
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset

from dataclasses import dataclass
from pathlib import Path
from jaxtyping import Float, Int

# Placeholder simple MLP stack
class MemoryMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int = 4096):
        """
        A simple MLP with with an expanding MLP dimensions

        Args:
            hidden_size:        The size of the input and output features.
            mlp_dim:            The size of the hidden layer in the MLP.
        """
        super().__init__()
    
        self.MLPStack = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, hidden_size)
        )

    def forward(self, x):
        """
        Forward function with residual connection.
        """
        return x + self.MLPStack(x)

# Load pre-trained small LLM model
model, tokenizer = FastLanguageModel.from_pretrained(
        model_name =  "unsloth/Qwen3-14B",
        max_seq_length = 2048,
        load_in_4bit = True,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
    )

# Load small datasets for training
reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

# Function for generating correct format of data
def generate_conversation(examples):
    problems  = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    return { "conversations": conversations, }

reasoning_data = reasoning_dataset.map(generate_conversation, batched=True)
reasoning_conversations = [
    tokenizer.apply_chat_template(conv, tokenize=False)
    for conv in reasoning_data["conversations"]
]

dataset = standardize_sharegpt(non_reasoning_dataset)
non_reasoning_conversations = [
    tokenizer.apply_chat_template(conv, tokenize=False)
    for conv in dataset["conversations"]
]

# Combine datasets (with percentage of chat vs reasoning set by chat_percentage)
chat_percentage = 0.25
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations)*(chat_percentage/(1 - chat_percentage))),
    random_state = 2407,
)

data = pd.concat([
    pd.Series(reasoning_conversations),
    pd.Series(non_reasoning_subset)
])
data.name = "text"

combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)

# Attach it directly to the Unsloth model
hidden_size = model.config.hidden_size
model.MemoryMLPLayer = MemoryMLP(hidden_size).to(model.device)

# Save old forward function
old_forward = model.forward

# New forward function using the MLP layer
def new_forward(**kwargs):
    """
    A new forward function that applies the MLP layer to the final hidden state.
    """
    # Extract the last hidden state from the old forward function
    outputs = old_forward(**kwargs)
    hidden_states = outputs.last_hidden_state

    # Apply your MLP on the final hidden states
    hidden_states = model.MemoryMLPLayer(hidden_states)

    # Replace output (preserve logits computation)
    outputs.last_hidden_state = hidden_states
    return outputs

# Replace the LLM forward function with the new function
model.forward = new_forward

# Freeze all model parameters except the MLP layer
for p in model.parameters():
    p.requires_grad = False

for p in model.MemoryMLPLayer.parameters():
    p.requires_grad = True

@dataclass
class MLPTrainingArgs:
    """Contains params for use in neptune.Run"""
    neptune_project: str | None = "andreykarailiev/resnet-tests"
    neptune_name: str | None = "nebius-test-unsloth"

class MLPTrainer:
    def __init__(self, args: MLPTrainingArgs):
        self.args = args

    def pre_training_setup(self):
        """Initializes the neptune run using `neptune.Run` and `neptune.log`."""
        self.run = neptune.init_run(project=self.args.neptune_project, experiment_name=self.args.neptune_name, api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vc2NhbGUubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3NjYWxlLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYmI0Y2IxZTYtNDM2Ny00MzA1LWFlOWYtYjlkODQ0YjQwN2M2In0=")
        self.neptune_callback = NeptuneCallback(run=self.run)

        self.trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = combined_dataset,
            eval_dataset = None, # Can set up evaluation!
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4, # Use GA to mimic batch size!
                warmup_steps = 5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = 30,
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "None", 
            ),
            callbacks=[self.neptune_callback]
        )

    def train(self) -> None:
        """Neptune Integration with Unsloth."""
        self.pre_training_setup()

        self.trainer.train()

        self.run.stop()

if __name__ == "__main__":
    args = MLPTrainingArgs()
    trainer = MLPTrainer(args)
    trainer.train()