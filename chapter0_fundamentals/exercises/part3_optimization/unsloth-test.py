import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import unsloth
import neptune_scale as Neptune

from transformers import TrainerCallback
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from peft import PeftConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft import mapping
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from dataclasses import dataclass

# Load pre-trained small LLM model
model, tokenizer = FastLanguageModel.from_pretrained(
        model_name =  "unsloth/Qwen3-14B",
        max_seq_length = 2048,
        load_in_4bit = True
    )

def get_training_dataset() -> Dataset:
    """Loads and combines the reasoning and non-reasoning datasets."""
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
    
    return combined_dataset

class MemoryMLPConfig(PeftConfig):
    def __init__(self, memory_dim=1024, target_modules=["transformer.h"], **kwargs):
        super().__init__(**kwargs)
        self.memory_dim = memory_dim
        self.target_modules = target_modules
        self.peft_type = "MEMORY_MLP"

class MemoryMLPLayer(BaseTunerLayer, nn.Module):
    def __init__(self, base_layer: nn.Module, config: MemoryMLPConfig):
        super().__init__()
        self.base_layer = base_layer
        hidden_dim = getattr(base_layer, "in_features", None)

        self.memory = nn.Sequential(
            nn.Linear(hidden_dim, config.memory_dim),
            nn.ReLU(),
            nn.Linear(config.memory_dim, hidden_dim),
        )

        for p in self.memory.parameters():
            p.requires_grad = True

    def forward(self, x, *args, **kwargs):
        base_out = self.base_layer(x, *args, **kwargs)
        mem_out = self.memory(x)
        return base_out + mem_out

class MemoryMLPTuner(BaseTuner):
    def __init__(self, model, peft_config, adapter_name):
        super().__init__(model, peft_config, adapter_name)

    def _check_target_module_exists(self, peft_config, key):
        """Return True if this module name matches a target."""
        match = any(t in key for t in peft_config.target_modules)
        # if match:
            # print(f"✅ Target match found: {key}")
        return match
    
    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent_module, **kwargs):
        """Replace target module with a MemoryMLPLayer."""
        old_module = target
        new_module = MemoryMLPLayer(old_module, peft_config)
        return new_module
    
    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent_module, **kwargs):
        """Replace target module with a MemoryMLPLayer."""
        new_module = MemoryMLPLayer(target, peft_config)
        # manually assign the replacement into the parent container
        if isinstance(parent_module, t.nn.ModuleList):
            idx = int(target_name.split(".")[-1])
            parent_module[idx] = new_module
        else:
            setattr(parent_module, target_name.split(".")[-1], new_module)
        return new_module
    
    def _mark_only_adapters_as_trainable(self, model):
        """Freeze all except memory MLP parameters."""
        for n, p in model.named_parameters():
            if "memory" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _prepare_adapter_config(self, peft_config, model_config):
        """Optional: modify config before injection (we'll just return it)."""
        return peft_config

    def set_adapter(self, adapter_name: str):
        """Set the active adapter. For MemoryMLP, there's usually only one."""
        self.active_adapter = adapter_name

    def enable_adapter_layers(self):
        """No-op for now."""
        pass

    def disable_adapter_layers(self):
        """No-op for now."""
        pass

# Register the tuner
mapping.PEFT_TYPE_TO_TUNER_MAPPING["MEMORY_MLP"] = MemoryMLPTuner

# Usage
config = MemoryMLPConfig(
    memory_dim=1024,
    target_modules=["model.layers.39.self_attn.q_proj"],
)

model = get_peft_model(model, config)

print(model)

for name, module in model.named_modules():
    if "Memory" in module.__class__.__name__:
        print("✅ Injected:", name)

@dataclass
class MLPTrainingArgs:
    """Contains params for use in neptune.Run"""
    neptune_project: str | None = "andreykarailiev/resnet-tests"
    neptune_name: str | None = "nebius-test-unsloth-2"

class NeptuneScaleCallback(TrainerCallback):
    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k, v in logs.items():
                self.run[f"train/{k}"].append(v)

class MLPTrainer:
    def __init__(self, args: MLPTrainingArgs):
        self.args = args

    def pre_training_setup(self):
        """Initializes the neptune run using `neptune.Run` and `neptune.log`."""
        self.run = Neptune.Run(project=self.args.neptune_project, experiment_name=self.args.neptune_name, api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vc2NhbGUubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3NjYWxlLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYmI0Y2IxZTYtNDM2Ny00MzA1LWFlOWYtYjlkODQ0YjQwN2M2In0=")
        self.neptune_callback = NeptuneScaleCallback(self.run)

        self.training_dataset = get_training_dataset()

        self.trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = self.training_dataset,
            eval_dataset = None,
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

        self.run.close()

if __name__ == "__main__":
    args = MLPTrainingArgs()
    # trainer = MLPTrainer(args)
    # trainer.train()