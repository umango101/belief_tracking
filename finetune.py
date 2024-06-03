import os
import sys
from numpy import dtype
import torch
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from train_utils import (
    load_model_tokenizer,
    DataModule,
    TrainingModule,
    print_trainable_parameters,
)
import warnings

warnings.filterwarnings("ignore")

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

current_dir = os.path.dirname(os.path.realpath(__file__))
if "mind" not in current_dir:
    current_dir = f"{current_dir}/mind"
print(f"Current directory: {current_dir}")


def train(args):
    torch.set_float32_matmul_precision("medium")

    model, tokenizer = load_model_tokenizer(args.model_name)

    wandb_logger = WandbLogger(
        log_model="all",
        name=f"{args.model_name}_local",
        project="mind",
    )
    datamodule = DataModule(
        tokenizer,
        args.data_path,
        args.batch_size,
        args.max_context_len,
        args.seed,
        model.config,
        current_dir,
    )

    # For LoRA training
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    model = TrainingModule(
        model,
        tokenizer,
        seed=args.seed,
        output_dir=args.output_dir,
        lr=args.lr,
        eval_metric=datamodule.eval_metric(),
        train_log_step=(args.accumulate_grad_batches // args.batch_size),
    )

    trainer = Trainer(
        default_root_dir=args.output_dir,
        devices=args.devices,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        precision="16-mixed",
        gradient_clip_val=1.0,
        deterministic=True,
        accumulate_grad_batches=(args.accumulate_grad_batches // args.batch_size),
        accelerator="gpu",
        enable_checkpointing=False,
        val_check_interval=0.1,
        strategy="ddp",
        log_every_n_steps=args.train_log_step,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule.test_dataloader())
    trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")

    parser.add_argument(
        "--data_path",
        type=str,
        default="generated_tomi.json",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./finetuning_output")
    parser.add_argument("--devices", type=list, default=[0])
    parser.add_argument("--accumulate_grad_batches", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--train_log_step", type=int, default=1)
    parser.add_argument("--max_context_len", type=int, default=1024)

    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=list,
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    args = parser.parse_args()
    # args.output_dir = os.path.join(args.output_dir, args.model_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(f"{args.output_dir}/checkpoints")

    train(args)


if __name__ == "__main__":
    main()
