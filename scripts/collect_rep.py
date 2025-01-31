import json
import random
import os
import math
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
all_containers= {}
all_characters = json.load(open(os.path.join(env_utils.DEFAULT_DATA_DIR, "synthetic_entities", "characters.json"), "r"))

for TYPE, DCT in {"states": all_states, "containers": all_containers}.items():
    ROOT = os.path.join(
        env_utils.DEFAULT_DATA_DIR, "synthetic_entities", TYPE
    )
    for file in os.listdir(ROOT):
        file_path = os.path.join(ROOT, file)
        with open(file_path, "r") as f:
            names = json.load(f)
        DCT[file.split(".")[0]] = names

model = LanguageModel("meta-llama/Meta-Llama-3-70B-Instruct", device_map="auto", load_in_4bit=True, torch_dtype=torch.float16, dispatch=True)

n_samples = 1000
batch_size = 1

configs_1, configs_2 = [], []
for _ in range(n_samples):
    template_1 = STORY_TEMPLATES['templates'][0]
    template_2 = STORY_TEMPLATES['templates'][1]
    characters = random.sample(all_characters, 2)
    containers = random.sample(all_containers[template_1["container_type"]], 2)
    states = random.sample(all_states[template_1["state_type"]], 2)
    event_idx = None
    event_noticed = False
    visibility = random.choice([True, False])

    sample = SampleV3(
        template=template_1 if not visibility else template_2,
        characters=characters,
        containers=containers,
        states=states,
        visibility=visibility,
        event_idx=event_idx,
        event_noticed=event_noticed,
    )
    configs_1.append(sample)

dataset = DatasetV3(configs_1)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

cached_acts = torch.zeros(n_samples, model.config.num_hidden_layers, 8, model.config.hidden_size)

for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    prompt = data['prompt'][0]
    
    with torch.no_grad():

        with model.trace() as tracer:

            with tracer.invoke(prompt):
                for l in range(model.config.num_hidden_layers):
                    for t_idx, t in enumerate(range(-8, 0)):
                        cached_acts[bi, l, t_idx] = model.model.layers[l].output[0][0, t].cpu().save()

    if bi % 500 == 0 and bi != 0:
        torch.save(cached_acts, "/media/sda/cached_acts.pt")
        print("Cache saved at", bi)