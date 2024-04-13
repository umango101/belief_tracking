from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.utilities.data import DataLoader
import torch
import os
from torchmetrics import Metric
import json
from datasets import load_dataset, Dataset
from torch.utils.data import random_split


class LabelClassification(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            if pred.strip() == target.strip():
                self.correct += 1
            self.total += 1

    def compute(self):
        return {"accuracy": self.correct.float() / self.total.float()}


class DataModule(LightningDataModule):

    def __init__(
        self,
        tokenizer,
        data_path=None,
        batch_size=None,
        cutoff_len=None,
        seed=None,
        val_size=0,
        test_size=0,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.cutoff_len = cutoff_len
        self.prepare_data_per_node = False

        self.train_set = load_dataset("json", data_files=self.data_path)["train"]

        with open("data/unexpected_contents.jsonl") as f:
            unexpected_contents = [json.loads(line) for line in f]
        with open("data/unexpected_transfer.jsonl") as f:
            unexpected_transfer = [json.loads(line) for line in f]

        tom_data = []
        for data in unexpected_contents + unexpected_transfer:
            for i in range(3):
                inp = {"input": data["prompts"][i], "label": data[f"target_{i+1}"]}
                tom_data.append(inp)

        tom_dataset = Dataset.from_list(tom_data)
        self.val_set = tom_dataset

    def tokenize(self, prompts):
        results = self.tokenizer(
            prompts,
            truncation=False,
            padding="longest",
            return_tensors=None,
        )
        return results

    def test_collate_fn(self, batch):
        batch_size = len(batch)
        self.tokenizer.padding_side = "left"

        prompts = [batch[idx]["input"] for idx in range(batch_size)]
        tokenized_prompts = self.tokenize(prompts)
        tokenized_prompts["label"] = []

        for idx in range(batch_size):
            tokenized_prompts["label"].append(
                self.tokenizer.encode(" " + batch[idx]["label"])[0]
            )

        for key in tokenized_prompts.keys():
            tokenized_prompts[key] = torch.tensor(tokenized_prompts[key])

        return tokenized_prompts

    def train_collate_fn(self, batch):
        batch_size = len(batch)
        self.tokenizer.padding_side = "right"

        full_prompts = [
            batch[idx]["input"] + batch[idx]["output"] for idx in range(batch_size)
        ]
        tokenized_full_prompts = self.tokenize(full_prompts)
        tokenized_full_prompts["labels"] = tokenized_full_prompts["input_ids"].copy()

        # Masking the input tokens
        for idx in range(batch_size):
            input_len = len(self.tokenize(batch[idx]["input"])["input_ids"])
            tokenized_full_prompts["labels"][idx] = [
                -100
            ] * input_len + tokenized_full_prompts["labels"][idx][input_len:]

        # Truncate the input tokens if it exceeds the cutoff length
        for idx in range(batch_size):
            if len(tokenized_full_prompts["input_ids"][idx]) > self.cutoff_len:
                diff = len(tokenized_full_prompts["input_ids"][idx]) - self.cutoff_len
                tokenized_full_prompts["input_ids"][idx] = tokenized_full_prompts[
                    "input_ids"
                ][idx][diff:]
                tokenized_full_prompts["attention_mask"][idx] = tokenized_full_prompts[
                    "attention_mask"
                ][idx][diff:]
                tokenized_full_prompts["labels"][idx] = tokenized_full_prompts[
                    "labels"
                ][idx][diff:]

        for key in tokenized_full_prompts.keys():
            tokenized_full_prompts[key] = torch.tensor(tokenized_full_prompts[key])

        return tokenized_full_prompts

    def eval_metric(self):
        return LabelClassification()

    def train_dataloader(self):
        if self.train_set is None:
            return None
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                collate_fn=self.train_collate_fn,
            )

    def val_dataloader(self):
        if self.val_set is None:
            return None
        else:
            return DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                collate_fn=self.test_collate_fn,
            )


class TrainingModule(LightningModule):

    def __init__(
        self,
        model,
        tokenizer,
        seed,
        output_dir,
        eval_metric=None,
        log_valid_loss=False,
        ckpt_epochs=None,
        **kwargs,
    ):
        super(TrainingModule, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.log_valid_loss = log_valid_loss
        self.metric = eval_metric
        self.hyperparams = kwargs
        self.ckpt_epochs = ckpt_epochs

    def save_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def on_train_start(self) -> None:
        seed_everything(self.seed)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        try:
            local_rank = torch.distributed.get_rank()
        except RuntimeError:
            local_rank = -1
        if local_rank in [-1, 0]:
            self.model.save_pretrained(os.path.join(self.output_dir, "final_ckpt"))

    def on_validation_start(self):
        self.metric.reset()
        self.all_outputs = []
        self.all_targets = []
        if self.log_valid_loss:
            self.valid_losses = []
        return super().on_validation_start()

    def validation_step(self, batch, batch_idx):
        input_ids, labels = (
            batch["input_ids"],
            batch["label"],
        )
        outputs = self.model(input_ids)
        logits = outputs.logits[:, -1]
        pred_ids = torch.argmax(logits, dim=-1)
        pred_tokens = [self.tokenizer.decode(pred_id) for pred_id in pred_ids]
        labels = [self.tokenizer.decode(label) for label in labels]

        self.all_outputs.append(pred_tokens)
        self.all_targets.append(labels)

        metric_scores = self.metric(pred_tokens, labels)

        if self.log_valid_loss:
            loss = self.model(**batch).loss
            metric_scores["valid_loss"] = loss
            self.valid_losses.append(loss.item())

        self.log_dict(metric_scores, prog_bar=True, sync_dist=True)

    def on_validation_end(self) -> None:
        results = self.metric.compute()
        try:
            local_rank = torch.distributed.get_rank()
        except RuntimeError:
            local_rank = 0

        if local_rank in [-1, 0]:
            results = {k: v.item() for k, v in results.items()}
            if self.log_valid_loss:
                results["valid_loss"] = sum(self.valid_losses) / len(self.valid_losses)

            with open(os.path.join(self.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            f.close()

        save_path = os.path.join(self.output_dir, f"preds_{local_rank}.txt")

        with open(save_path, "w", encoding="utf-8") as f:
            for output, target in zip(self.all_outputs, self.all_targets):
                f.write("Output: {}\n".format(output))
                f.write("Target: {}\n\n".format(target))
        f.close()

        return super().on_validation_end()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), **self.hyperparams)


def load_model_tokenizer(model_name: str):
    """
    Loads the model and tokenizer for the probing task.

    Args:
        model_name (str): Name of the Transformer model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if "llama" in model_name:
    #     hf_name = "baffo32/decapoda-research-llama-7B-hf"
    #     model = AutoModelForCausalLM.from_pretrained(hf_name)
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         "hf-internal-testing/llama-tokenizer", padding_side="left"
    #     )

    # elif "gpt" in model_name:
    #     model = AutoModelForCausalLM.from_pretrained(model_name)
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # if "pythia" in model_name:
    #     assert model_name in [
    #         "pythia-2.8b",
    #         "pythia-1.4b",
    #         "pythia-1b",
    #         "pythia-410m",
    #         "pythia-160m",
    #         "pythia-70m",
    #     ], "Invalid model name."
    #     hf_name = f"EleutherAI/{model_name}"
    #     model = AutoModelForCausalLM.from_pretrained(hf_name)
    #     tokenizer = AutoTokenizer.from_pretrained(hf_name, padding_side="left")

    # else:
    #     raise ValueError(f"Model {model_name} not found.")

    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer
