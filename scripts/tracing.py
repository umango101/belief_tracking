import random
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nnsight import CONFIG, LanguageModel
from collections import defaultdict


sys.path.append("../")
from src.dataset import STORY_TEMPLATES
from src.utils import env_utils
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(10)

CONFIG.set_default_api_key("5da1d831c11c44e5a63f122fb06a4c18")
os.environ["HF_TOKEN"] = "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"

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
CONFIG.APP.REMOTE_LOGGING = False

all_states = {}
all_containers = {}
all_characters = json.load(
    open(os.path.join(env_utils.DEFAULT_DATA_DIR, "synthetic_entities", "characters.json"), "r")
)

for TYPE, DCT in {"states": all_states, "containers": all_containers}.items():
    ROOT = os.path.join(env_utils.DEFAULT_DATA_DIR, "synthetic_entities", TYPE)
    for file in os.listdir(ROOT):
        file_path = os.path.join(ROOT, file)
        with open(file_path, "r") as f:
            names = json.load(f)
        DCT[file.split(".")[0]] = names


model = LanguageModel("meta-llama/Meta-Llama-3.1-405B")
print("Model loaded successfully!")

n_samples = 10
batch_size = 1

dataset = get_obj_tracing_exps(
    STORY_TEMPLATES, all_characters, all_containers, all_states, n_samples
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print("Dataset created successfully!")

accs = {}
input_tokens = model.tokenizer(dataset[0]["clean_prompt"], return_tensors="pt")["input_ids"][0]
story_token_idx = [
    i for i, x in enumerate(input_tokens.tolist()) if x == model.tokenizer.encode("Story")[1]
][0]
input_tokens_len = len(
    model.tokenizer(dataset[0]["clean_prompt"], return_tensors="pt")["input_ids"][0]
)

if os.path.exists("../results/tracing_results.json"):
    old_results = json.load(open("../results/tracing_results.json", "r"))
    accs = old_results.copy()

for token_idx in range(input_tokens_len - 1, story_token_idx, -1):
    for layer_idx in range(31, 40, 1):

        print(f"Starting tracing for Layer: {layer_idx} | Token Idx: {token_idx}")
        if str(layer_idx) in accs and str(token_idx) in accs[str(layer_idx)]:
            continue
        elif str(layer_idx) not in accs:
            accs[str(layer_idx)] = {}

        correct, total = 0, 0
        for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            corrupt_prompt = batch["corrupt_prompt"][0]
            clean_prompt = batch["clean_prompt"][0]
            target = batch["target"][0]

            with model.trace(remote=True) as tracer:

                with tracer.invoke(corrupt_prompt):
                    corrupt_layer_out = model.model.layers[layer_idx].output[0][0, token_idx].save()

                with tracer.invoke(clean_prompt):
                    model.model.layers[layer_idx].output[0][0, token_idx] = corrupt_layer_out
                    pred = model.lm_head.output[0, -1].argmax(dim=-1).save()

            if model.tokenizer.decode([pred]).lower().strip() == target:
                correct += 1
            total += 1

            del corrupt_layer_out, pred
            torch.cuda.empty_cache()

        acc = round(correct / total, 2)
        accs[str(layer_idx)][token_idx] = acc
        print(f"Layer: {layer_idx} | Token Idx: {token_idx} | Accuracy: {acc}")

        with open("../results/tracing_results.json", "w") as f:
            json.dump(accs, f, indent=4)
