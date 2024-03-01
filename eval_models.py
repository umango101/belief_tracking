import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from utils import get_dataset, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("models.json", "r") as f:
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
        half_precision = model_details["half_precision"]

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)
        if half_precision:
            model.half()
        try:
            model.to(device)
        except:
            results[model_name.split("/")[-1]] = "Failed to move to gpu"
            print(f"Failed to move {model_name}  to gpu. Skipping.")
            continue

        dataset = get_dataset(
            datafiles=[
                "data/unexpected_contents.jsonl",
                "data/unexpected_transfer.jsonl",
            ]
        )
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=8,
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
                        # print(f"{tokenizer.decode(inp['input_ids'][0])}: {tokenizer.decode(predicted_id)} == {tokenizer.decode(inp['target'][i])}")
                        correct += 1
                    else:
                        pass
                        # print(f"{tokenizer.decode(inp['input_ids'][0])}: {tokenizer.decode(predicted_id)} != {tokenizer.decode(inp['target'][i])}")
                    total += 1

                del inp, outputs, logits, predicted_ids
                torch.cuda.empty_cache()

        print(f"Model Name: {model_name.split('/')[-1]} | Accuracy: {correct/total:.2f}")
        results[model_name.split("/")[-1]] = round(correct / total, 2)


if __name__ == "__main__":
    main()
