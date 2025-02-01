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
from utils import get_visibility_align_exps

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

n_samples = 500
batch_size = 1

dataset = get_visibility_align_exps(STORY_TEMPLATES,
                                    all_characters,
                                    all_containers,
                                    all_states,
                                    n_samples,
                                    question_type="belief_question",
                                    diff_visibility=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

cached_acts = torch.empty(n_samples, model.config.num_hidden_layers, 5, model.config.hidden_size)

for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    vis_prompt = batch['corrupt_prompt'][0]
    no_vis_prompt = batch['clean_prompt'][0]

    vis_acts, no_vis_acts = defaultdict(dict), defaultdict(dict)
    with torch.no_grad():

        with model.trace() as tracer:

            with tracer.invoke(vis_prompt):
                for l in range(model.config.num_hidden_layers):
                    for t_idx, token_idx in enumerate([-8, -7, -5, -3, -1]):
                        vis_acts[l][t_idx] = model.model.layers[l].output[0][0, token_idx].cpu().save()

            with tracer.invoke(no_vis_prompt):
                for l in range(model.config.num_hidden_layers):
                    for t_idx, token_idx in enumerate([-8, -7, -5, -3, -1]):
                        no_vis_acts[l][t_idx] = model.model.layers[l].output[0][0, token_idx].cpu().save()
    
    for l in range(model.config.num_hidden_layers):
        for t_idx, token_idx in enumerate([-8, -7, -5, -3, -1]):
            cached_acts[bi, l, t_idx] = vis_acts[l][t_idx] - no_vis_acts[l][t_idx]
    
    del vis_acts, no_vis_acts
    torch.cuda.empty_cache()

    if bi % 250 == 0 and bi != 0:
        torch.save(cached_acts, "/media/sda/visibility_diff_cache.pt")
        print("Cache saved at", bi)