import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
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

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # try:
        if precision == "fp32":
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        elif precision == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(device)
        elif precision == "int8":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16,
            )
        elif precision == "int4":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
        # except:
        #     results[model_name.split("/")[-1]] = "Failed to load"
        #     print(f"Failed to load {model_name}. Skipping.")
        #     continue

        dataset = get_dataset(
            datafiles=[
                f"{current_dir}/data/unexpected_contents.jsonl",
                f"{current_dir}/data/unexpected_transfer.jsonl",
            ]
        )
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                collate_fn=lambda x: collate_fn(model, tokenizer, x),
            )
        except:
            results[model_name.split("/")[-1]] = "Failed to create dataloader"
            print(f"Failed to create dataloader for {model_name}. Skipping.")
            continue

        correct, total = 0, 0
        with torch.no_grad():
            for inp in tqdm(dataloader, total=len(dataloader)):
                inp["input_ids"] = inp["input_ids"].to(device)
                inp["attention_mask"] = inp["attention_mask"].to(device)

                outputs = model(inp["input_ids"])
                logits = outputs.logits
                predicted_ids = torch.argmax(logits[:, -1], dim=-1)

                for i, predicted_id in enumerate(predicted_ids):
                    if predicted_id == inp["target"][i]:
                        with open(
                            f"{current_dir}/{model_name.split('/')[-1]}_log.txt", "a"
                        ) as f:
                            f.write(
                                f"{tokenizer.decode(predicted_id)}({predicted_id}) == {tokenizer.decode(inp['target'][i])}({inp['target'][i]})\n"
                            )
                        correct += 1
                    else:
                        with open(
                            f"{current_dir}/{model_name.split('/')[-1]}_log.txt", "a"
                        ) as f:
                            f.write(
                                f"{tokenizer.decode(predicted_id)}({predicted_id}) != {tokenizer.decode(inp['target'][i])}({inp['target'][i]})\n"
                            )
                    total += 1

                del inp, outputs, logits, predicted_ids
                torch.cuda.empty_cache()

        print(
            f"Model Name: {model_name.split('/')[-1]} | Accuracy: {correct/total:.2f}"
        )
        results[model_name.split("/")[-1]] = round(correct / total, 2)

        with open(f"{current_dir}/results.json", "a") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
