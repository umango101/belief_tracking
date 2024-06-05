import os
import json
import shutil
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from nnsight import LanguageModel, CONFIG

from utils import load_model_and_tokenizer, load_tomi_data

CONFIG.set_default_api_key("6TnmrIokoS3Judkyez1F")
os.environ["HF_TOKEN"] = "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"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndif", type=bool, default=False)
    args = parser.parse_args()

    for model_details in models:
        if "model" in locals():
            del model
        if "tokenizer" in locals():
            del tokenizer

        model_name = model_details["model_name"]
        precision = model_details["precision"]
        batch_size = model_details["batch_size"]

        if not args.ndif:
            model, tokenizer = load_model_and_tokenizer(model_name, precision, device)
            dataloader = load_tomi_data(
                model.config, tokenizer, current_dir, batch_size=batch_size
            )
        else:
            model = LanguageModel(model_name)
            dataloader = load_tomi_data(
                model.config, model.tokenizer, current_dir, batch_size=batch_size
            )

        print(f"{model_name} and Data loaded successfully")

        correct, total = {}, {}
        with torch.no_grad():
            for batch_idx, inp in tqdm(enumerate(dataloader), total=len(dataloader)):
                if not args.ndif:
                    inp["input_ids"] = inp["input_ids"].to(device)
                    inp["target"] = inp["target"].to(device)
                    outputs = model(inp["input_ids"])
                    logits = outputs.logits[:, -1]
                    pred_token_ids = torch.argmax(logits, dim=-1)

                if args.ndif:
                    with model.trace(
                        inp["input_ids"], scan=False, validate=False, remote=True
                    ):
                        pred_token_ids = torch.argmax(
                            model.output["logits"][:, -1], dim=-1
                        ).save()

                for i in range(len(inp["category"])):
                    category = inp["category"][i]
                    if category in correct:
                        correct[category] += int(pred_token_ids[i] == inp["target"][i])
                        total[category] += 1
                    else:
                        correct[category] = int(pred_token_ids[i] == inp["target"][i])
                        total[category] = 1

                for idx in range(len(inp["target"])):
                    if not args.ndif:
                        target_text = tokenizer.decode(inp["target"][idx].tolist())
                        pred_text = tokenizer.decode(pred_token_ids[idx].tolist())
                    else:
                        target_text = model.tokenizer.decode(
                            inp["target"][idx].tolist()
                        )
                        pred_text = model.tokenizer.decode(pred_token_ids[idx].tolist())
                    category = inp["category"][idx]
                    with open(
                        f"{current_dir}/preds/same_shots/{model_name.split('/')[-1]}.jsonl",
                        "a",
                    ) as f:
                        f.write(
                            json.dumps(
                                {
                                    "category": category,
                                    "target": target_text,
                                    "pred": pred_text,
                                }
                            )
                            + "\n"
                        )

                del inp, pred_token_ids
                torch.cuda.empty_cache()

        all_corrects, all_totals = 0, 0
        for category in total:
            all_corrects += correct[category]
            all_totals += total[category]
        overall_accuracy = round(all_corrects / all_totals, 2)
        print(f"Model Name: {model_name}, Overall Accuracy: {overall_accuracy}")

        with open(f"{current_dir}/preds/same_shots/results.txt", "a") as f:
            f.write(
                f"Model Name: {model_name} | Overall Accuracy: {overall_accuracy}\n"
            )

        with open(f"{current_dir}/preds/same_shots/results.txt", "a") as f:
            for category in total:
                accuracy = round(correct[category] / total[category], 2)
                print(f"Category: {category}, Accuracy: {accuracy}")
                f.write(
                    f"Model Name: {model_name} | Category: {category} | Correct: {correct[category]} | Total: {total[category]}\n"
                )
            f.write("\n")

        del model, tokenizer
        torch.cuda.empty_cache()

        # home_dir = str(Path.home())
        # shutil.rmtree(
        #     f"{home_dir}/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"
        # )


if __name__ == "__main__":
    main()
