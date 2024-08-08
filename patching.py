import json
import csv
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from nnsight import LanguageModel
from collections import defaultdict
from tqdm import tqdm
from utils import get_control_corrupt_data


def find_period_token_indices(input_tokens):
    # Find all indices of period tokens (627) except the first occurrence for each item in the batch
    period_mask = input_tokens == 627
    first_period_mask = period_mask.cumsum(dim=1) == 1
    period_mask = period_mask & ~first_period_mask

    # Get the indices for periods and their predecessors
    period_indices = period_mask.nonzero()
    predecessor_indices = period_indices.clone()
    predecessor_indices[:, 1] -= 1

    # Combine period and predecessor indices
    combined_indices = torch.cat([period_indices, predecessor_indices], dim=0)

    return combined_indices


model = LanguageModel(
    "meta-llama/Meta-Llama-3-70B-Instruct",
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    dispatch=True,
)
print("Model loaded...")

variable = "0_forward_belief"

with open(f"data/bigtom/{variable}_true_belief/stories.csv", "r") as f:
    reader = csv.reader(f, delimiter=";")
    tb_data = list(reader)

with open(f"data/bigtom/{variable}_false_belief/stories.csv", "r") as f:
    reader = csv.reader(f, delimiter=";")
    fb_data = list(reader)

with open(f"data/bigtom/{variable}_true_control/stories.csv", "r") as f:
    reader = csv.reader(f, delimiter=";")
    tb_control = list(reader)

with open(f"data/bigtom/{variable}_false_control/stories.csv", "r") as f:
    reader = csv.reader(f, delimiter=";")
    fb_control = list(reader)

n_samples = 48
batch_size = 16

accs = defaultdict(list)

for iter in range(10):
    samples = get_control_corrupt_data(tb_data, fb_control, n_samples)
    dataset = Dataset.from_list(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for layer_idx in range(0, 42, 2):
        correct, total = 0, 0
        for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            corrupt = batch["corrupt_prompt"]
            clean = batch["clean_prompt"]
            target = batch["corrupt_target"]

            with torch.no_grad():
                corrupt_tokens = model.tokenizer(
                    corrupt, return_tensors="pt", padding=True
                ).input_ids
                corrupt_imp_token_indices = find_period_token_indices(corrupt_tokens)
                with model.trace(corrupt, scan=False, validate=False):
                    layer_out = model.model.layers[layer_idx].output[0]
                    control_layer_out = layer_out[
                        corrupt_imp_token_indices[:, 0], corrupt_imp_token_indices[:, 1]
                    ].save()

                clean_tokens = model.tokenizer(
                    clean, return_tensors="pt", padding=True
                ).input_ids
                clean_imp_token_indices = find_period_token_indices(clean_tokens)
                with model.trace(clean, scan=False, validate=False):
                    layer_out = model.model.layers[layer_idx].output[0]
                    layer_out[
                        clean_imp_token_indices[:, 0], clean_imp_token_indices[:, 1]
                    ] = control_layer_out

                    preds = model.lm_head.output[:, -1].argmax(dim=-1).save()

                for i, pred in enumerate(preds):
                    # print(f"Prediction: {model.tokenizer.decode([pred])} | Target: {target[0]}")
                    if model.tokenizer.decode([pred]).strip() == target[i].strip():
                        correct += 1
                    total += 1

            # print(f"CUDA Memory: {torch.cuda.memory_summary()}")
            del preds
            torch.cuda.empty_cache()

        acc = round(correct / total, 2)
        accs[layer_idx].append(acc)
        print(f"Iteration: {iter} | Layer: {layer_idx} | Accuracy: {acc}")

        # Save accs in a json file
        with open(f"agent_perspective_variable_accs.json", "w") as f:
            json.dump(accs, f)
