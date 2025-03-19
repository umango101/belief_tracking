import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

import json
import random
import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Any, List, Optional
import nnsight
from nnsight import CONFIG, LanguageModel
import numpy as np
from collections import defaultdict
from einops import einsum
import time
from einops import rearrange, reduce

sys.path.append("../")
from src.dataset import SampleV3, DatasetV3, STORY_TEMPLATES
from src.utils import env_utils
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(10)

CONFIG.set_default_api_key("d9e00ab7d4f74643b3176de0913f24a7")
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


model = LanguageModel(
    "meta-llama/Meta-Llama-3-70B-Instruct",
    cache_dir="/disk/u/nikhil/.cache/huggingface/hub/",
    device_map="auto",
    torch_dtype=torch.float16,
    dispatch=True,
)


n_samples = 100
batch_size = 10

dataset = get_obj_tracing_exps(
    STORY_TEMPLATES, all_characters, all_containers, all_states, n_samples
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

corrects = []
for bi, batch in enumerate(dataloader):
    corrupt_prompt = batch["corrupt_prompt"]
    clean_prompt = batch["clean_prompt"]
    corrupt_target = batch["corrupt_ans"]
    clean_target = batch["clean_ans"]

    with torch.no_grad():

        with model.trace() as tracer:

            with tracer.invoke(clean_prompt):
                clean_pred = model.lm_head.output[:, -1].argmax(dim=-1).save()

            with tracer.invoke(corrupt_prompt):
                corrupt_pred = model.lm_head.output[:, -1].argmax(dim=-1).save()

    for i in range(batch_size):
        clean_pred_token = model.tokenizer.decode([clean_pred[i]]).lower().strip()
        corrupt_pred_token = model.tokenizer.decode([corrupt_pred[i]]).lower().strip()
        clean_target_token = clean_target[i].lower().strip()
        corrupt_target_token = corrupt_target[i].lower().strip()
        index = bi * batch_size + i
        if clean_pred_token == clean_target_token and corrupt_pred_token == corrupt_target_token:
            corrects.append(index)

    del clean_pred, corrupt_pred
    torch.cuda.empty_cache()

    if len(corrects) >= 50:
        corrects = corrects[:50]
        break

batch_size = 50
correct_dataloader = DataLoader(
    [dataset[i] for i in corrects], batch_size=batch_size, shuffle=False
)
print("Correct dataset size:", len(correct_dataloader.dataset), "Batch size:", batch_size)


tracing_results = defaultdict(dict)
for t in tqdm(range(180, 128, -1)):
    for layer_idx in range(0, model.config.num_hidden_layers):
        correct, total = 0, 0

        for bi, batch in enumerate(correct_dataloader):
            corrupt_prompt = batch["corrupt_prompt"]
            clean_prompt = batch["clean_prompt"]
            target = batch["target"]
            batch_size = len(target)

            corrupt_layer_out = defaultdict(dict)
            with torch.no_grad():

                with model.trace() as tracer:

                    with tracer.invoke(corrupt_prompt):
                        corrupt_layer_out = model.model.layers[layer_idx].output[0][:, t].clone()

                    with tracer.invoke(clean_prompt):
                        model.model.layers[layer_idx].output[0][:, t] = corrupt_layer_out

                        pred = model.lm_head.output[:, -1].argmax(dim=-1).save()

            for i in range(batch_size):
                pred_token = model.tokenizer.decode([pred[i]]).lower().strip()
                target_token = target[i].lower().strip()
                if pred_token == target_token:
                    correct += 1
                total += 1

            del corrupt_layer_out, pred
            torch.cuda.empty_cache()

        acc = round(correct / total, 2)
        tracing_results[t][layer_idx] = acc
        print(f"Token: {t} | Layer: {layer_idx} | Accuracy: {acc}")

        if (layer_idx + 1) % 10 == 0:
            with open("../tracing_results/object.json", "w+") as f:
                json.dump(tracing_results, f, indent=4)
