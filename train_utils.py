from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.utilities.data import DataLoader
import torch
import os
from torchmetrics import Metric
import json
from datasets import load_dataset, Dataset
from torch.utils.data import random_split
from accelerate import Accelerator

device_index = Accelerator().process_index
device_map = {"": device_index}


class LabelClassification(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        self.correct += torch.sum(preds == targets).item()
        self.total += len(targets)

    def compute(self):
        return {"accuracy": self.correct.float() / self.total.float()}


class DataModule(LightningDataModule):

    def __init__(
        self,
        tokenizer,
        data_path=None,
        current_dir=None,
        batch_size=None,
        cutoff_len=None,
        seed=None,
        config=None,
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
        self.config = config
        self.prepare_data_per_node = False
        self.train_set = load_dataset("json", data_files=self.data_path)["train"]
        self.current_dir = current_dir

    def preprocess_valid_data(self, data):
        prev_sent_idx = 0
        processed_data = []
        example = []

        with open(f"{self.current_dir}/priming_examples.txt", "r") as f:
            priming_exps = f.read()

        for sentence in data:
            sent_idx = int(sentence.split(" ")[0])
            sentence = sentence[2:]

            if sent_idx > prev_sent_idx:
                example.append(sentence)
            else:
                context = "".join(example[:-1]).strip()
                question = example[-1].split("\t")[0].strip()
                answer = example[-1].split("\t")[1].strip()
                processed_data.append(
                    {
                        "input": f"{priming_exps}Context: {context}Question: {question}\nAnswer:",
                        "target": " " + answer,
                    }
                )

                example = [sentence]

            prev_sent_idx = sent_idx

        return processed_data

    def get_valid_data(self):
        try:
            with open(
                f"{self.current_dir}/data/SymbolicToM Datasets/Linguistic Diversity Dataset/test.txt",
                "r",
            ) as f:
                valid_data = f.readlines()
            processed_data = self.preprocess_valid_data(valid_data)
            dataset = Dataset.from_list(processed_data)
            return dataset
        except FileNotFoundError:
            return None

    def tokenize(self, prompts):
        results = self.tokenizer(
            prompts,
            truncation=False,
            padding="longest",
            return_tensors=None,
        )
        return results

    def valid_collate_fn(self, examples):
        inputs = self.tokenizer(
            [ex["input"] for ex in examples],
            return_tensors="pt",
            padding=True,
        )
        if (
            (
                self.config.architectures[0] == "LlamaForCausalLM"
                and "Llama-3" not in self.config._name_or_path
            )
            or self.config.architectures[0] == "LlaMAForCausalLM"
            or self.config.architectures[0] == "GemmaForCausalLM"
            or self.config.architectures[0] == "MistralForCausalLM"
            or self.config.architectures[0] == "MixtralForCausalLM"
        ):
            inputs["target"] = torch.tensor(
                [self.tokenizer.encode(ex["target"])[2] for ex in examples]
            )

        else:
            inputs["target"] = torch.tensor(
                [self.tokenizer.encode(ex["target"])[0] for ex in examples]
            )
        return inputs

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
            # masking the input tokens
            tokenized_full_prompts["labels"][idx] = [
                -100
            ] * input_len + tokenized_full_prompts["labels"][idx][input_len:]

            # masking the padded tokens
            content_len = sum(tokenized_full_prompts["attention_mask"][idx])
            tokenized_full_prompts["labels"][idx] = tokenized_full_prompts["labels"][
                idx
            ][:content_len] + [-100] * (
                len(tokenized_full_prompts["input_ids"][idx]) - content_len
            )

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
        self.val_set = self.get_valid_data()
        if self.val_set is None:
            return None
        else:
            return DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                collate_fn=self.valid_collate_fn,
            )


class TrainingModule(LightningModule):

    def __init__(
        self,
        model,
        tokenizer,
        seed,
        output_dir,
        train_log_step=10,
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
        self.train_log_step = train_log_step

    def save_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def on_train_start(self) -> None:
        seed_everything(self.seed)
        self.losses = []
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.losses.append(loss.item())

        if batch_idx % self.train_log_step == 0:
            self.log(
                "train_loss",
                sum(self.losses) / len(self.losses),
                logger=True,
                prog_bar=True,
                on_step=True,
            )
            self.losses = []

        # self.log("train_loss", loss, logger=True, prog_bar=True, on_step=True)
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
        input_ids, targets = (
            batch["input_ids"],
            batch["target"],
        )
        outputs = self.model(input_ids)
        logits = outputs.logits[:, -1]
        pred_token_ids = torch.argmax(logits, dim=-1)

        self.all_outputs.append(pred_token_ids)
        self.all_targets.append(targets)

        metric_scores = self.metric(pred_token_ids, targets)

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
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.hyperparams)
        return optimizer


def load_model_tokenizer(model_name: str):
    """
    Loads the model and tokenizer for the probing task.

    Args:
        model_name (str): Name of the Transformer model.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
