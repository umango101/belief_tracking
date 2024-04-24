import os
import json
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from utils import load_model_and_tokenizer, load_tomi_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
if "mind" not in current_dir:
    current_dir = f"{current_dir}/mind"
print(f"Current directory: {current_dir}")

with open(f"{current_dir}/models.json", "r") as f:
    models = json.load(f)

# with open(f"{current_dir}/tomi_results.txt", "r") as f:
#     evaluated_models = f.read().split("\n")

# # Remove already evaluated models from models list using evaluated_models dict
# for model_name in evaluated_models:
#     models = [model for model in models if model_name not in model["model_name"]]


def main():
    for model_details in models:
        if "model" in locals():
            del model
            torch.cuda.empty_cache()
        if "tokenizer" in locals():
            del tokenizer

        model_name = model_details["model_name"]
        precision = model_details["precision"]
        batch_size = model_details["batch_size"]

        model, tokenizer = load_model_and_tokenizer(model_name, precision, device)
        print(f"{model_name} loaded successfully")
        dataloader = load_tomi_data(
            model.config, tokenizer, f"{current_dir}/data/ToMi", batch_size=batch_size
        )
        print(f"Data loaded successfully")

        correct, total = 0, 0
        with torch.no_grad():
            for inp in tqdm(dataloader, total=len(dataloader)):
                inp["input_ids"] = inp["input_ids"].to(device)
                inp["target"] = inp["target"].to(device)

                outputs = model(inp["input_ids"])
                logits = outputs.logits[:, -1]
                pred_token_ids = torch.argmax(logits, dim=-1)

                correct += torch.sum(pred_token_ids == inp["target"]).item()
                total += inp["target"].numel()

            del inp, outputs, logits, pred_token_ids
            torch.cuda.empty_cache()

        accuracy = round(correct / total, 2)
        print(f"Model Name: {model_name.split('/')[-1]} | Accuracy: {accuracy}")

        with open(f"{current_dir}/tom_results.txt", "a") as f:
            f.write(f"Model Name: {model_name} | Accuracy: {accuracy}\n")

        home_dir = str(Path.home())
        shutil.rmtree(
            f"{home_dir}/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"
        )


if __name__ == "__main__":
    main()
