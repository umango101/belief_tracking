import os
import json
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from utils import get_dataset, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
if "mind" not in current_dir:
    current_dir = f"{current_dir}/mind"
print(f"Current directory: {current_dir}")

with open(f"{current_dir}/models.json", "r") as f:
    models = json.load(f)

with open(f"{current_dir}/results.json", "r") as f:
    evaluated_models = json.load(f)

# Remove already evaluated models from models list using evaluated_models dict
for model_name in evaluated_models.keys():
    models = [model for model in models if model_name not in model["model_name"]]


def main():
    results = {}
    for model_details in models:
        if "model" in locals():
            del model
            torch.cuda.empty_cache()
        if "tokenizer" in locals():
            del tokenizer

        model_name = model_details["model_name"]
        precision = model_details["precision"]

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token="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"
        )
        tokenizer.pad_token = tokenizer.eos_token

        if precision == "fp32":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, token="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"
            ).to(device)
        elif precision == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                token="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",
            ).to(device)
        elif precision == "int8":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16,
                token="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",
            )
        elif precision == "int4":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
                token="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",
            )

        dataset = get_dataset(
            datafiles=[
                f"{current_dir}/data/unexpected_contents.jsonl",
                f"{current_dir}/data/unexpected_transfer.jsonl",
            ]
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda x: collate_fn(model, tokenizer, x),
        )

        correct, total = 0, 0
        with torch.no_grad():
            for inp in tqdm(dataloader, total=len(dataloader)):
                inp["input_ids"] = inp["input_ids"].to(device)
                inp["attention_mask"] = inp["attention_mask"].to(device)

                outputs = model(inp["input_ids"])
                logits = outputs.logits
                predicted_ids = torch.argmax(logits[:, -1], dim=-1)
                predicted_token = tokenizer.decode(predicted_ids)

                if predicted_token.strip() == inp["target"][0].strip():
                    with open(
                        f"{current_dir}/{model_name.split('/')[-1]}_log.txt", "a"
                    ) as f:
                        f.write(f"{predicted_token} == {inp['target'][0].strip()}\n")
                    correct += 1
                else:
                    with open(
                        f"{current_dir}/{model_name.split('/')[-1]}_log.txt", "a"
                    ) as f:
                        f.write(f"{predicted_token} != {inp['target'][0].strip()}\n")
                total += 1

                del inp, outputs, logits, predicted_ids
                torch.cuda.empty_cache()

        print(
            f"Model Name: {model_name.split('/')[-1]} | Accuracy: {correct/total:.2f}"
        )
        results[model_name.split("/")[-1]] = round(correct / total, 2)

        with open(f"{current_dir}/results.json", "a") as f:
            json.dump(results, f, indent=4)

        home_dir = str(Path.home())
        shutil.rmtree(
            f"{home_dir}/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"
        )


if __name__ == "__main__":
    main()
