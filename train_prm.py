import os
import sys
from pprint import pprint
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils import data

import wandb
from nano_transformer import (
    TransformerConfig,
    TransformerLMHead,
    configure_optimizer,
    flat_cross_entropy,
)
from util import Config, Environment, LRSchedule


class PRMDataset(data.Dataset):
    def __init__(self, config: Config, split: Literal["train", "val", "test"]) -> None:
        data_path = f"{config.data_dir}/{split}.npz"
        data = np.load(data_path)
        self.x = torch.from_numpy(data["x"] + 1).long()  # type: ignore
        self.y = torch.from_numpy(data["y"]).long()  # type: ignore
        self.mask = torch.any(self.y != -1, dim=0)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.x[i], self.y[i]

    @property
    def vocab_size(self) -> int:
        return int(self.x.max().item()) + 1


@torch.no_grad()
def evaluate_model(
    model: TransformerLMHead,
    dataset: PRMDataset,
    step: int,
    config: Config,
    env: Environment,
    max_iters=25,
) -> float:
    model.eval()
    data_loader = data.DataLoader(
        dataset,
        batch_size=config.test_batch_size,
        shuffle=True,
        pin_memory=env.pin_memory,
        pin_memory_device=env.pin_memory_device,
    )

    token_acc = []
    sequence_acc = []
    losses = []

    for i, (x, y) in enumerate(data_loader):
        if i >= max_iters:
            break

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            logits = model(x)
            loss = flat_cross_entropy(logits[:, dataset.mask], y[:, dataset.mask])

        y = y[:, dataset.mask]
        y_pred = torch.argmax(logits, dim=2)[:, dataset.mask]
        token_acc.append(torch.mean((y == y_pred).float()).item())
        sequence_acc.append(torch.mean(torch.all(y == y_pred, dim=1).float()).item())
        losses.append(loss.item())

    wandb.log(
        {
            "val_token_accuracy": np.mean(token_acc),
            "val_sequence_accuracy": np.mean(sequence_acc),
            "val_loss": np.mean(losses),
        },
        step=step,
    )

    return float(np.mean(losses))


def train(config: Config, env: Environment) -> None:
    """
    Trains model using config parameters. Assumes data is in `config.data_dir`.
    Saves model in `config.model_dir`.
    """

    print(f"{env.device=}, env.context={str(type(env.context))[8 : -2]}", end=", ")
    print(f"{env.pin_memory=}, {env.pin_memory_device=}, {env.compile_blocks=}")
    pprint(config.to_dict())

    run = wandb.init(
        dir=config.model_dir,
        project="can-decoder-triplet-count",
        config=config.to_dict(),
        name=config.name,
        resume=config.resume,
    )

    env.seed_everything(config.seed)

    train_dataset = PRMDataset(config, split="train")
    val_dataset = PRMDataset(config, split="val")

    model_config = TransformerConfig(
        n_positions=config.block_size,
        in_vocab_size=train_dataset.vocab_size,
        out_vocab_size=2,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        use_wpe=config.use_wpe,
    )

    model = TransformerLMHead(model_config).to(env.device)

    if env.compile_blocks:
        model.compile()

    optimizer = configure_optimizer(
        model,
        lr=config.min_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        custom_optim_groups=config.custom_optim_groups,
        device=env.device,
    )

    lr_schedule = LRSchedule(config)
    i = 0
    best_val_loss = float("inf")
    n_evals_without_improving = 0

    if config.resume:
        load_path = os.path.join(config.model_dir, config.checkpoint_name)
        checkpoint = torch.load(load_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        i = checkpoint["i"]
        best_val_loss = checkpoint["best_val_loss"]

    while True:
        train_data_loader = data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=env.pin_memory,
            pin_memory_device=env.pin_memory_device,
        )

        for x, y in train_data_loader:
            i += 1

            model.train()

            lr = lr_schedule(i)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x = x.to(env.device)
            y = y.to(env.device)

            with env.context:
                logits = model(x)
                loss = flat_cross_entropy(
                    logits[:, train_dataset.mask],
                    y[:, train_dataset.mask],
                )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            wandb.log({"train_loss": loss.item()}, step=i)

            if i % config.eval_interval == 0:
                val_loss = evaluate_model(model, val_dataset, i, config, env)

                if val_loss < best_val_loss:
                    n_evals_without_improving = 0
                    print(f"saved checkpoint    {f'{i=}':8}  {val_loss=:.3f}")
                    best_val_loss = val_loss
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "i": i,
                        "best_val_loss": best_val_loss,
                    }
                    save_path = os.path.join(config.model_dir, config.checkpoint_name)
                    torch.save(checkpoint, save_path)
                else:
                    n_evals_without_improving += 1

            if i >= config.max_iters or (
                n_evals_without_improving >= config.max_evals_without_improving
                and best_val_loss < config.max_loss_for_early_stopping
            ):
                run.finish()
                return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python train_prm.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    env = Environment()

    train(config, env)
