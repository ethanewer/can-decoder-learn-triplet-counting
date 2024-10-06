import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import Block, Linear, TransformerConfig


class Transformer(nn.Module):
    """Transformer model without lm-head, with optional causal encoder."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.in_vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.apply(self.__init_weights)
        for p_name, p in self.named_parameters():
            if p_name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def __init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.wte(input_ids)

        T = x.shape[1]
        assert T <= self.config.n_positions

        if self.config.use_wpe:
            position_ids = torch.arange(T, device=x.device)
            x += self.wpe(position_ids)

        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class TransformerLMHead(nn.Module):
    """Transformer language model, with optional causal encoder."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_positions = config.n_positions
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.out_vocab_size, bias=False)
        if config.in_vocab_size == config.out_vocab_size:
            self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.transformer(input_ids)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        decoder: bool = True,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.n_positions:
                input_ids = input_ids[:, -self.n_positions :]

            logits = self(input_ids)

            if deterministic:
                next_id = torch.argmax(logits[:, -1:], dim=2)
            else:
                distribution = F.softmax(logits[:, -1] / temperature, dim=1)
                next_id = torch.multinomial(distribution, num_samples=1)

            input_ids = torch.cat((input_ids, next_id), dim=1)

        return input_ids
