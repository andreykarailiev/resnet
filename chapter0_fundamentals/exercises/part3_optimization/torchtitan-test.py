from neptune_scale import Run
from transformers import AutoTokenizer, Qwen3ForCausalLM, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from dataclasses import dataclass
from jaxtyping import Float, Int

from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange
from functools import partial
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, stack
from torch.utils.data import DataLoader
from titans_pytorch import MemoryAsContextTransformer
from titans_pytorch.mac_transformer import pad_and_segment_with_inverse, pack_with_inverse, default, exists, create_mac_block_mask, flex_attention
from datasets import load_dataset, Dataset
from tqdm import tqdm

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

class MemoryMLP(nn.Module):
    def __init__(self, in_features: int, mlp_dim: int = 512):
        super().__init__()
        self.MLPLayers = nn.Sequential(
            nn.Linear(in_features, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, in_features)
        )

    def forward(self, x):
        return self.MLPLayers(x)

class MemoryAsContextLayer(MemoryAsContextTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        disable_flex_attn = False,
        cache = None,
        factorized_pos_emb = None
    ):
        batch, seq_len, dim = x.shape
        neural_mem_segment_len = self.neural_memory_segment_len
        segment_len = self.segment_len
        num_longterm_mem_tokens = self.num_longterm_mem_tokens
        attn_window_size = self.attn_window_size
        seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)

        # intersperse longterm memory
        x, inverse_segment = pad_and_segment_with_inverse(x, segment_len, inverse_remove_pad = False)

        mems = repeat(self.longterm_mems, 'n d -> b n d', b = x.shape[0])
        x, inverse_pack_mems = pack_with_inverse((x, mems), 'b * d')

        x = inverse_segment(x)

        # splice out unneeded tokens from padding for longterm mems
        x = x[:, :seq_len_with_mem]

        # apply axial positional embedding
        # so intra and inter segment can be more easily discerned by the network
        pos_emb = self.axial_pos_emb.forward_with_seq_len(seq_len_with_mem, (neural_mem_segment_len,), factorized = factorized_pos_emb)

        x = x + pos_emb

        # prep flex attention
        use_flex_attn = x.is_cuda and self.use_flex_attn and not disable_flex_attn

        flex_attn_fn = None

        if use_flex_attn:
            block_mask = create_mac_block_mask(seq_len_with_mem, self.attn_window_size, self.num_persist_mem_tokens, self.sliding_window_attn)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # kv caching
        is_inferencing = exists(cache)

        if not exists(cache):
            cache = (seq_len_with_mem - 1, None, None)

        inference_seq_index, kv_caches, neural_mem_caches = cache

        kv_caches = iter(default(kv_caches, []))
        neural_mem_caches = iter(default(neural_mem_caches, []))

        next_kv_caches = []
        next_neural_mem_caches = []

        # value residual
        value_residual = None

        # neural mem weight residual
        mem_weight_residual = None

        # layers for the neural mem to select the qkv inputs from
        mem_input_layers = []

        # when inferencing, only do one token at a time
        if is_inferencing:
            ind = inference_seq_index
            x = x[:, ind:(ind + 1)]

        # expand and reduce streams for hyper connections
        x = self.expand_streams(x)

        for mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, attn, ff in self.layers:

            retrieved = None
            attn_out_gates = None
            next_neural_mem_cache = None

            # maybe neural memory

            if exists(mem):

                mem_input, add_residual = mem_hyper_conn(x)

                if not exists(mem_qkv_layer_selector):
                    qkv_mem_input = stack((mem_input, mem_input, mem_input))
                else:
                    layers_to_choose_from = stack((mem_input, *mem_input_layers))

                    # let the current `mem_input` select the 3 layers for qkv

                    selected = mem_qkv_layer_selector(mem_input)

                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')

                retrieved, next_neural_mem_cache = mem.forward(
                    qkv_mem_input,
                    state = next(neural_mem_caches, None),
                    prev_weights = mem_weight_residual
                )

                if self.neural_mem_weight_residual:
                    mem_weight_residual = next_neural_mem_cache.updates

                if self.gate_attn_output:
                    attn_out_gates = retrieved.sigmoid()
                else:
                    x = add_residual(retrieved)

            # attention

            attn_in, add_residual = attn_hyper_conn(x)

            mem_input_layers.append(attn_in)

            attn_out, (values, next_kv_cache) = attn(
                attn_in,
                value_residual = value_residual,
                disable_flex_attn = disable_flex_attn,
                flex_attn_fn = flex_attn_fn,
                output_gating = attn_out_gates,
                cache = next(kv_caches, None)
            )

            mem_input_layers.append(attn_out)

            value_residual = default(value_residual, values)

            x = add_residual(attn_out)

            # caches

            next_kv_caches.append(next_kv_cache)
            next_neural_mem_caches.append(next_neural_mem_cache)

            # feedforward

            ff_in, add_ff_residual = ff_hyper_conn(x)

            mem_input_layers.append(ff_in)

            ff_out = ff(ff_in)

            mem_input_layers.append(ff_out)

            x = add_ff_residual(ff_out)

        # hyper connection reducing of streams
        x = self.reduce_streams(x)

        # excise out the memories
        if not is_inferencing:

            x, inverse_segment = pad_and_segment_with_inverse(x, attn_window_size, inverse_remove_pad = False)

            x, _ = inverse_pack_mems(x)

            x = inverse_segment(x)

            x = x[:, :seq_len]

        # to logits
        x = self.norm(x)

        return x

class MemoryAugmentModel(Qwen3ForCausalLM):
    def __init__(self, config, memory_dim=1024):
        super().__init__(config)

        # save lm_head temporarily
        old_lm_head = self.lm_head
        del self.lm_head

        # Extract lm_head input dimension
        hidden_size = old_lm_head.in_features

        # Add memory MLP layer
        self.InjectedMemory = MemoryAsContextLayer(num_tokens=hidden_size, 
                                    dim=hidden_size, 
                                    depth=4,
                                    segment_len=256,
                                    num_longterm_mem_tokens=64,
                                    num_persist_mem_tokens=32)

        # Re-attach lm_head
        self.lm_head = old_lm_head

    def forward(self, *args, **kwargs):
        # Get outputs from the base model and extract the hidden state
        outputs = self.model(*args, **kwargs)
        hidden_state= outputs.last_hidden_state

        # Pass the hidden state through the memory MLP
        memory_state = self.InjectedMemory(hidden_state)

        # Compute new logits
        logits = self.lm_head(hidden_state)

        # Output with expected structure
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def get_wikitext_data(tokenizer: AutoTokenizer) -> tuple[Dataset, Dataset]:
    def chunk_texts(examples):
        text = " ".join(examples["text"])
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        seq_len = 256
        chunks = [tokens[i:i+seq_len] for i in range(0, len(tokens) - seq_len, seq_len)]
        return {"input_ids": chunks}
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    tokenized = dataset.map(chunk_texts, batched=True, remove_columns=["text"], load_from_cache_file=True)

    train_set, test_set = tokenized["train"], tokenized["test"] 

    return train_set, test_set

def get_LLM_for_finetuning() -> MemoryAugmentModel:
    # Load your base model
    model = MemoryAugmentModel.from_pretrained("Qwen/Qwen3-0.6B")

    # 1️⃣ Freeze everything
    model.requires_grad_(False)

    # 2️⃣ Unfreeze only your custom MLP
    if hasattr(model, "InjectedMemory"):
        model.InjectedMemory.requires_grad_(True)
    else:
        raise ValueError("Model has no attribute 'InjectedMemory' — check naming.")

    return model

@dataclass
class MemoryFinetuningArgs:
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

class MemoryFinetuner:
    def __init__(self, args: MemoryFinetuningArgs):
        self.args = args

    def pre_training_setup(self):
        self.model = get_LLM_for_finetuning()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.model.to(device)


        self.optimizer = t.optim.AdamW(
            self.model.InjectedMemory.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        ) 

        self.trainset, self.testset = get_wikitext_data(self.tokenizer)

        def collate(batch):
            input_ids = t.stack([t.tensor(b["input_ids"], dtype=t.long) for b in batch])
            labels = input_ids.clone()
            return input_ids, labels

        self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate)
        self.test_loader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False, collate_fn=collate)

        self.logged_variables = {"loss": [], "accuracy": []}
        self.examples_seen = 0

    def training_step(
        self,
        input_ids: Int[t.Tensor, "batch seq"],
        labels: Int[t.Tensor, "batch seq"]
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data."""
        input_ids, labels = input_ids.to(device), labels.to(device)

        # Forward pass (logits shape: [batch, seq_len, vocab_size])
        outputs = self.model(input_ids)
        logits = outputs.logits

        # Shift so that tokens <n> predict token <n+1>
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1),
                               ignore_index=self.tokenizer.pad_token_id or -100)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += input_ids.size(0)
        self.logged_variables["loss"].append(loss.item())
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """Evaluate the model on the test set and return the accuracy."""
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        accuracy = total_correct / total_samples
        self.logged_variables["accuracy"].append(accuracy)
        return accuracy

    def train(self) -> dict[str, list[float]]:
        self.pre_training_setup()

        # accuracy = self.evaluate()
        accuracy = 0

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for input_ids, labels in pbar:
                loss = self.training_step(input_ids, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen:06}")

            # accuracy = self.evaluate()
            accuracy = 0
            pbar.set_postfix(loss=f"{loss:.3f}", accuracy=f"{accuracy:.2f}", ex_seen=f"{self.examples_seen:06}")

        return self.logged_variables

@dataclass
class NeptuneMemoryFinetuningArgs(MemoryFinetuningArgs):
    """Contains new params for use in neptune.Run, as well as all the ResNetFinetuningArgs params."""
    neptune_project: str | None = "andreykarailiev/resnet-tests"
    neptune_name: str | None = "llm-memory-test-MAC-2"

class NeptuneMemoryFinetuner(MemoryFinetuner):
    args: NeptuneMemoryFinetuningArgs  # adding this line helps with typechecker!
    examples_seen: int = 0  # tracking examples seen (used as step for Neptune)

    def pre_training_setup(self):
        """Initializes the neptune run using `neptune.Run` and `neptune.log`."""
        super().pre_training_setup()

        self.run = Run(project=self.args.neptune_project, experiment_name=self.args.neptune_name, api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vc2NhbGUubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3NjYWxlLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYmI0Y2IxZTYtNDM2Ny00MzA1LWFlOWYtYjlkODQ0YjQwN2M2In0=")
        self.run.log_configs(vars(self.args))

        self.examples_seen = 0

    def training_step(
        self,
        input_ids: Int[t.Tensor, "batch seq"],
        labels: Int[t.Tensor, "batch seq"]
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data."""
        input_ids, labels = input_ids.to(device), labels.to(device)

        # Forward pass (logits shape: [batch, seq_len, vocab_size])
        outputs = self.model(input_ids)
        logits = outputs.logits

        # Shift so that tokens <n> predict token <n+1>
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1),
                               ignore_index=self.tokenizer.pad_token_id or -100)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += input_ids.size(0)
        self.run.log_metrics({"loss": loss.item()}, step=self.examples_seen)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """Equivalent to ResNetFinetuner.evaluate, but logging the accuracy to neptune."""
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        accuracy = total_correct / total_samples
        self.run.log_metrics({"accuracy": accuracy}, step=self.examples_seen)
        return accuracy

    def train(self) -> None:
        """Equivalent to ResNetFinetuner.train, but with neptune integration."""
        self.pre_training_setup()
        #accuracy = self.evaluate()
        accuracy = 0

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for input_ids, labels in pbar:
                loss = self.training_step(input_ids, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen:06}")

            # accuracy = self.evaluate()
            accuracy = 0
            pbar.set_postfix(
                loss=f"{loss:.3f}", accuracy=f"{accuracy:.2f}", ex_seen=f"{self.examples_seen=:06}"
            )

        self.run.close()

if __name__ == "__main__":
    args = NeptuneMemoryFinetuningArgs()
    finetuner = NeptuneMemoryFinetuner(args)
    finetuner.train()