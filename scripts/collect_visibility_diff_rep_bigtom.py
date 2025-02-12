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
import pandas as pd

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

model = LanguageModel("meta-llama/Meta-Llama-3-70B-Instruct", cache_dir="/disk/u/nikhil/.cache/huggingface/hub", device_map="auto", load_in_4bit=True, torch_dtype=torch.float16, dispatch=True)

# Helper functions
def get_ques_start_token_idx(tokenizer, prompt):
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze()
    corrolary_token = tokenizer.encode(":", return_tensors="pt").squeeze()[-1].item()
    ques_start_idx = (input_tokens == corrolary_token).nonzero()[2].item()

    return ques_start_idx-1


def get_prompt_token_len(tokenizer, prompt):
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze()
    return len(input_tokens)


# Loading BigToM data
## Read a csv file
df_false = pd.read_csv("../data/bigtom/0_forward_belief_false_belief/stories.csv", delimiter=";")
df_true = pd.read_csv("../data/bigtom/0_forward_belief_true_belief/stories.csv", delimiter=";")

## Create a dataset
batch_size = 1

true_stories, false_stories = [], []
for i in range(len(df_true)):
    story = df_true.iloc[i]['story']
    question = df_true.iloc[i]['question']
    answer = df_true.iloc[i]['answer']
    distractor = df_true.iloc[i]['distractor']
    true_stories.append({"story": story, "question": question, "answer": answer, "distractor": distractor})

for i in range(len(df_false)):
    story = df_false.iloc[i]['story']
    question = df_true.iloc[i]['question']
    answer = df_false.iloc[i]['answer']
    distractor = df_false.iloc[i]['distractor']
    false_stories.append({"story": story, "question": question, "answer": answer, "distractor": distractor})

dataset = []
# instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any beliefs about the container and its contents which they cannot observe. 4. To answer the question, predict only the final state of the queried object in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict container or character as the final output."
instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."


for i in range(min(len(true_stories), len(false_stories))):
    question = true_stories[i]['question']
    visible_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[i]['story']}\nQuestion: {question}\nAnswer:"

    question = false_stories[i]['question']
    invisible_prompt = f"Instructions: {instruction}\n\nStory: {false_stories[i]['story']}\nQuestion: {question}\nAnswer:"

    dataset.append({
        "visible_prompt": visible_prompt,
        "visible_ans": true_stories[i]['answer'],
        "invisible_prompt": invisible_prompt,
        "invisible_ans": false_stories[i]['answer'],
    })

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Find the prompt with most number of tokens between ques_idx and prompt_len
max_prompt_len = 0
for batch in dataloader:
    visible_prompt = batch['visible_prompt'][0]
    invisible_prompt = batch['invisible_prompt'][0]

    visible_ques_idx = get_ques_start_token_idx(model.tokenizer, visible_prompt)
    visible_prompt_len = get_prompt_token_len(model.tokenizer, visible_prompt)
    invisible_ques_idx = get_ques_start_token_idx(model.tokenizer, invisible_prompt)
    invisible_prompt_len = get_prompt_token_len(model.tokenizer, invisible_prompt)

    max_prompt_len = max(max_prompt_len, visible_prompt_len-visible_ques_idx, invisible_prompt_len-invisible_ques_idx)


cached_acts = torch.zeros(len(dataloader), model.config.num_hidden_layers, max_prompt_len, model.config.hidden_size)
token_lens = torch.zeros(len(dataloader))

for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    visible_prompt = batch['visible_prompt'][0]
    visible_ans = batch['visible_ans'][0]
    invisible_prompt = batch['invisible_prompt'][0]
    invisible_ans = batch['invisible_ans'][0]

    visible_ques_idx = get_ques_start_token_idx(model.tokenizer, visible_prompt)
    visible_prompt_len = get_prompt_token_len(model.tokenizer, visible_prompt)
    invisible_ques_idx = get_ques_start_token_idx(model.tokenizer, invisible_prompt)
    invisible_prompt_len = get_prompt_token_len(model.tokenizer, invisible_prompt)

    vis_acts, no_vis_acts = defaultdict(dict), defaultdict(dict)
    with torch.no_grad():

        with model.trace() as tracer:

            with tracer.invoke(visible_prompt):
                for l in range(model.config.num_hidden_layers):
                    for t_idx, token_idx in enumerate(range(visible_ques_idx, visible_prompt_len)):
                        vis_acts[l][t_idx] = model.model.layers[l].output[0][0, token_idx].cpu().save()

            with tracer.invoke(invisible_prompt):
                for l in range(model.config.num_hidden_layers):
                    for t_idx, token_idx in enumerate(range(invisible_ques_idx, invisible_prompt_len)):
                        no_vis_acts[l][t_idx] = model.model.layers[l].output[0][0, token_idx].cpu().save()

    for l in range(model.config.num_hidden_layers):
        for t_idx, token_idx in enumerate(range(visible_ques_idx, visible_prompt_len)):
            cached_acts[bi, l, t_idx] = vis_acts[l][t_idx] - no_vis_acts[l][t_idx]

    token_lens[bi] = visible_prompt_len-visible_ques_idx

    del vis_acts, no_vis_acts
    torch.cuda.empty_cache()

    if bi % 50 == 0 and bi != 0:
        torch.save(cached_acts, "/disk/u/nikhil/mind/caches/bigtom/visibility_diff_cache.pt")
        torch.save(token_lens, "/disk/u/nikhil/mind/caches/bigtom/token_lens.pt")
        print("\nCache saved at", bi)