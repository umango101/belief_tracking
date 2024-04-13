from numpy import dtype
import torch
import argparse
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import os
from train_utils import load_model_tokenizer, DataModule, TrainingModule


def train(args):
    model, tokenizer = load_model_tokenizer(args.model_name)
    torch.set_float32_matmul_precision("medium")
    wandb_logger = WandbLogger(
        log_model="all",
        name=f"{args.model_name}_local",
        project="tom",
    )

    datamodule = DataModule(
        tokenizer,
        args.data_path,
        args.batch_size,
        model.config.n_positions,
        args.seed,
    )
    model = TrainingModule(
        model,
        tokenizer,
        seed=args.seed,
        output_dir=args.output_dir,
        lr=args.lr,
        eval_metric=datamodule.eval_metric(),
    )

    trainer = Trainer(
        default_root_dir=args.output_dir,
        devices=args.devices,
        max_epochs=args.epochs,
        precision="32-true",
        gradient_clip_val=1.0,
        deterministic=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator="gpu",
        enable_checkpointing=False,
        val_check_interval=0.01,
        strategy="ddp",
        log_every_n_steps=1,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule.test_dataloader())
    trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")

    parser.add_argument(
        "--data_path",
        type=str,
        default="training_data.jsonl",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./weights")
    parser.add_argument("--devices", type=list, default=[0])
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(args)


if __name__ == "__main__":
    main()
