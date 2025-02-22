import json
import random
import os
import math
import sys
import torch
import torch.nn as nn
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
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(10)

CONFIG.set_default_api_key("d9e00ab7d4f74643b3176de0913f24a7")
os.environ["HF_TOKEN"] = "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"

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
CONFIG.APP.REMOTE_LOGGING = False

# Define random seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# model = LanguageModel("meta-llama/Meta-Llama-3.1-405B")
model = LanguageModel("meta-llama/Meta-Llama-3-70B-Instruct", cache_dir="/disk/u/nikhil/.cache/huggingface/hub/", device_map="auto", load_in_4bit=True, torch_dtype=torch.float16, dispatch=True)

def get_ques_start_token_idx(batch_size, tokenizer, prompt, padding_side="right"):
    input_tokens = tokenizer(prompt, return_tensors="pt", padding=True, padding_side=padding_side).input_ids
    colon_token = tokenizer.encode(":", return_tensors="pt").squeeze()[-1].item()
    ques_start_idx = (input_tokens == colon_token).nonzero()[torch.arange(2, 4*batch_size, 4)][:, 1] - 1

    return ques_start_idx

def get_prompt_token_len(tokenizer, prompt, padding_side="right"):
    input_tokens = tokenizer(prompt, return_tensors="pt", padding=True, padding_side=padding_side)
    return input_tokens.attention_mask.sum(dim=-1)

def check_pred(pred, target):
    prompt = f"Instruction: Check if the following ground truth and prediction are same or different. If they are the same, then predict 'Yes', else 'No'.\nGround truth: {target}\nPrediction: {pred}\nAnswer:"

    with torch.no_grad():
        with model.generate(prompt, max_new_tokens=5, do_sample=False, num_return_sequences=1, pad_token_id=model.tokenizer.pad_token_id):
            out = model.generator.output.save()

    prompt_len = get_prompt_token_len(model.tokenizer, prompt)

    return out, prompt_len

# Read a csv file
df_false = pd.read_csv("../data/bigtom/0_forward_belief_false_belief/stories.csv", delimiter=";")
df_true = pd.read_csv("../data/bigtom/0_forward_belief_true_belief/stories.csv", delimiter=";")

# For each row in the dataframe extract story, answer, and distractor
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
instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

for i in range(min(len(true_stories), len(false_stories))):
    question = true_stories[i]['question']
    visible_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[i]['story']}\nQuestion: {question}\nAnswer:"

    question = false_stories[i]['question']
    invisible_prompt = f"Instructions: {instruction}\n\nStory: {false_stories[i]['story']}\nQuestion: {question}\nAnswer:"

    visible_ans = true_stories[i]['answer'].split()
    invisible_ans = false_stories[i]['answer'].split()

    # Find the index of first word which is different in both answers
    diff_idx = 0
    for idx, (v, j) in enumerate(zip(visible_ans, invisible_ans)):
        if v != j:
            diff_idx = idx
            break
    
    visible_ans = " ".join(visible_ans[diff_idx:])[:-1]
    invisible_ans = " ".join(invisible_ans[diff_idx:])[:-1]

    random_choice = random.choice([0, 1])

    dataset.append({
        "alt_story": true_stories[i]['story'] if random_choice == 0 else false_stories[i]['story'],
        "alt_question": true_stories[i]['question'] if random_choice == 0 else false_stories[i]['question'],
        "alt_prompt": visible_prompt if random_choice == 0 else invisible_prompt,
        "alt_ans": visible_ans if random_choice == 0 else invisible_ans,
        "org_story": false_stories[i]['story'] if random_choice == 0 else true_stories[i]['story'],
        "org_question": false_stories[i]['question'] if random_choice == 0 else true_stories[i]['question'],
        "org_prompt": invisible_prompt if random_choice == 0 else visible_prompt,
        "org_ans": invisible_ans if random_choice == 0 else visible_ans,
        "target": visible_ans if random_choice == 0 else invisible_ans,
    })


train_size = 80
valid_size = 40
batch_size = 4

train_dataset = dataset[:train_size]
valid_dataset = dataset[train_size:train_size+valid_size]

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

sing_vecs = defaultdict(dict)
for l in range(41):
    sing_vecs[l] = torch.load(f"../svd_results/bigtom/singular_vecs/{l}.pt").cpu()

valid_accs_dcm, rank_dcm, preds = {}, {}, {}

valid_accs = defaultdict(dict)
for layer_idx in [i for i in range(30, 34, 2)] + [i for i in range(0, 40, 10)] + [i for i in range(22, 30, 2)]:
    preds[layer_idx] = {"correct": [], "incorrect": []}
    model.tokenizer.padding_side = "right"

    modules = [i for i in range(sing_vecs[layer_idx].shape[0])]
    mask = torch.ones(len(modules), requires_grad=True, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.Adam([mask], lr=1e-1)
    n_epochs = 1
    lamb = 0.03

    print(f"Training layer: {layer_idx}")
    for epoch in range(n_epochs):
        epoch_loss = 0

        for bi, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            alt_prompt = batch["alt_prompt"]
            org_prompt = batch["org_prompt"]
            target = batch["target"]
            target_token = model.tokenizer(target, return_tensors="pt", padding=True, padding_side="right")
            target_input_ids = target_token.input_ids[:, 1:]
            target_attention_mask = target_token.attention_mask[:, 1:]
            batch_size = target_input_ids.size(0)

            alt_ques_idx = get_ques_start_token_idx(batch_size, model.tokenizer, alt_prompt, padding_side="right")
            alt_prompt_len = get_prompt_token_len(model.tokenizer, alt_prompt, padding_side="right")
            org_ques_idx = get_ques_start_token_idx(batch_size, model.tokenizer, org_prompt, padding_side="right")
            org_prompt_len = get_prompt_token_len(model.tokenizer, org_prompt, padding_side="right")

            optimizer.zero_grad()

            with model.trace() as tracer:

                alt_acts = defaultdict(dict)
                with tracer.invoke(alt_prompt):
                    for j in range(batch_size):
                        alt_acts[j] = model.model.layers[layer_idx].output[0][j, alt_ques_idx[j]:alt_prompt_len[j]].clone().save()

                with tracer.invoke(org_prompt):
                    sing_vec = sing_vecs[layer_idx].cuda()
                    masked_vec = sing_vec * mask.unsqueeze(-1)
                    proj_matrix = torch.matmul(masked_vec.t(), masked_vec).half()

                    for j in range(batch_size):
                        curr_output = model.model.layers[layer_idx].output[0][j, org_ques_idx[j]:org_prompt_len[j]].clone()

                        alt_proj = torch.matmul(alt_acts[j], proj_matrix)
                        org_proj = torch.matmul(curr_output, proj_matrix)

                        modified_out = curr_output - org_proj + alt_proj
                        model.model.layers[layer_idx].output[0][j, org_ques_idx[j]:org_prompt_len[j]] = modified_out

                    logits = model.lm_head.output[torch.arange(batch_size), org_prompt_len-1].save()

                    del sing_vec, proj_matrix, masked_vec
                    torch.cuda.empty_cache()

            target_logit = 0
            for j in range(batch_size):
                target_logit += logits[j, target_input_ids[j, target_attention_mask[j] == 1]].sum()
            task_loss = -(target_logit/batch_size)
            l1_loss = lamb * torch.norm(mask, p=1)
            loss = task_loss + l1_loss.to(task_loss.device)
            
            epoch_loss += loss.item()
            
            if bi % 4 == 0:
                mean_loss = epoch_loss / (bi + 1)
                print(f"Epoch: {epoch}, Batch: {bi}, Task Loss: {task_loss.item():.4f}, "
                    f"L1 Loss: {l1_loss.item():.4f}, Total Loss: {mean_loss:.4f}")
                with torch.no_grad():
                    mask.data.clamp_(0, 1)
                    rounded = torch.round(mask)
                    print(f"#Causal SVs: {(rounded == 1).sum().item()}")
            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mask.data.clamp_(0, 1)
    
    del alt_acts, alt_prompt, org_prompt, target, target_token, target_input_ids, target_attention_mask, logits, task_loss, l1_loss, loss
    torch.cuda.empty_cache()

    print(f"Training finished for layer: {layer_idx}, lambda: {lamb}")

    print(f"Validation started for layer: {layer_idx}")
    correct, total = 0, 0

    with torch.inference_mode():
        model.tokenizer.padding_side = "left"
        mask_data = mask.data.clone()
        mask_data.clamp_(0, 1)
        rounded = torch.round(mask)
        print(f"Rank: {(rounded == 1).sum().item()}")
        rank_dcm[layer_idx] = (rounded == 1).sum().item()

        # Save the mask
        torch.save(mask_data, f"../masks/bigtom/{layer_idx}.pt")

        for bi, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            alt_prompt = batch["alt_prompt"]
            org_prompt = batch["org_prompt"]
            alt_ans = batch["alt_ans"]
            batch_size = len(alt_ans)

            alt_ques_idx = get_ques_start_token_idx(batch_size, model.tokenizer, alt_prompt, padding_side="left")
            alt_prompt_len = get_prompt_token_len(model.tokenizer, alt_prompt, padding_side="left")
            org_ques_idx = get_ques_start_token_idx(batch_size, model.tokenizer, org_prompt, padding_side="left")
            org_prompt_len = get_prompt_token_len(model.tokenizer, org_prompt, padding_side="left")

            with model.session() as session:

                alt_acts = defaultdict(dict)
                with model.trace(alt_prompt):
                    for j in range(batch_size):
                        alt_acts[j] = model.model.layers[layer_idx].output[0][j, alt_ques_idx[j]:].save()

                with model.generate(org_prompt, max_new_tokens=8, do_sample=False, num_return_sequences=1, pad_token_id=model.tokenizer.pad_token_id, eos_token_id=model.tokenizer.eos_token_id):
                    sing_vec = sing_vecs[layer_idx].cuda()
                    masked_vec = sing_vec.to(rounded.device) * rounded.unsqueeze(-1)
                    proj_matrix = torch.matmul(masked_vec.t(), masked_vec).half()

                    for j in range(batch_size):
                        curr_output = model.model.layers[layer_idx].output[0][j, org_ques_idx[j]:].clone()

                        alt_proj = torch.matmul(alt_acts[j], proj_matrix)
                        org_proj = torch.matmul(curr_output, proj_matrix)

                        modified_out = curr_output - org_proj + alt_proj
                        model.model.layers[layer_idx].output[0][j, org_ques_idx[j]:] = modified_out

                    out = model.generator.output.save()

                    del sing_vec, proj_matrix
                    torch.cuda.empty_cache()

            for j in range(batch_size):
                check_out, prompt_len = check_pred(model.tokenizer.decode(out[j, max(org_prompt_len):]), alt_ans[j])
                check = model.tokenizer.decode(check_out[0, prompt_len:-1]).strip()

                if check == "Yes":
                    correct += 1
                    preds[layer_idx]['correct'].append({
                        'pred': model.tokenizer.decode(out[j]).replace("<|eot_id|>", "").replace("<|begin_of_text|>", "").strip(),
                        'ground_truth': alt_ans[j],
                    })
                else:
                    preds[layer_idx]['incorrect'].append({
                        'pred': model.tokenizer.decode(out[j]).replace("<|eot_id|>", "").replace("<|begin_of_text|>", "").strip(),
                        'ground_truth': alt_ans[j],
                    })
                total += 1

            del alt_acts, alt_prompt, org_prompt, alt_ans, out
            torch.cuda.empty_cache()

        print(f"Validation accuracy: {correct / total:.2f} | Correct: {correct} | Total: {total}\n")
        valid_accs_dcm[layer_idx] = round(correct / total, 2)
    
    with open(f"../predictions/bigtom/SVs/{layer_idx}.json", "w") as f:
        json.dump(preds[layer_idx], f, indent=4)
    
    # Save valid_accs and rank_dcm as json file
    with open(f"../valid_accs_dcm.json", "w") as f:
        json.dump(valid_accs_dcm, f, indent=4)
    with open(f"../rank_dcm.json", "w") as f:
        json.dump(rank_dcm, f, indent=4)