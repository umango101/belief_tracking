import os
import json
import random
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from nnsight import CONFIG, LanguageModel

sys.path.append("../")
from utils import *
from bigtom_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(10)

CONFIG.set_default_api_key("d9e00ab7d4f74643b3176de0913f24a7")
os.environ["HF_TOKEN"] = "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"

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
CONFIG.APP.REMOTE_LOGGING = True

all_states = {}
all_containers = {}
all_characters = json.load(
    open(os.path.join("../data", "synthetic_entities", "characters.json"), "r")
)

for TYPE, DCT in {"states": all_states, "containers": all_containers}.items():
    ROOT = os.path.join("../data", "synthetic_entities", TYPE)
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

batch_size = 1

df_false = pd.read_csv("../data/bigtom/0_forward_belief_false_belief/stories.csv", delimiter=";")
df_true = pd.read_csv("../data/bigtom/0_forward_belief_true_belief/stories.csv", delimiter=";")

fb_samples = get_bigtom_samples(df_false, df_true, len(df_false), "false_belief")
tb_samples = get_bigtom_samples(df_false, df_true, len(df_true), "true_belief")
dataloader = DataLoader(tb_samples + fb_samples, batch_size=1, shuffle=False)

# Compute the length of longest visibility sentence in dataloader
max_len = 0
for data in dataloader:
    prompt = data["prompt"][0]
    visibility_sent_start_idx = get_visitibility_sent_start_idx(model.tokenizer, prompt)
    ques_start_idx = get_ques_start_token_idx(model.tokenizer, prompt)
    max_len = max(max_len, ques_start_idx - visibility_sent_start_idx)

n_samples = len(dataloader.dataset)
last_token_acts = torch.zeros(n_samples, model.config.num_hidden_layers, model.config.hidden_size)
query_charac_acts = torch.zeros(
    n_samples, model.config.num_hidden_layers, 2, model.config.hidden_size
)
visibility_sent_acts = torch.zeros(
    n_samples, model.config.num_hidden_layers, max_len, model.config.hidden_size
)
visibility_sent_lens = torch.zeros(n_samples)


for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    prompt = data["prompt"][0]
    ques_start_idx = get_ques_start_token_idx(model.tokenizer, prompt)
    visibility_sent_start_idx = get_visitibility_sent_start_idx(model.tokenizer, prompt)
    visibility_sent_lens[bi] = ques_start_idx - visibility_sent_start_idx

    with torch.no_grad():

        with model.trace() as tracer:

            with tracer.invoke(prompt):
                for l in range(model.config.num_hidden_layers):
                    last_token_acts[bi, l] = model.model.layers[l].output[0][0, -1].cpu().save()

                    query_charac_acts[bi, l] = (
                        model.model.layers[l]
                        .output[0][0, ques_start_idx + 3 : ques_start_idx + 5]
                        .cpu()
                        .save()
                    )

                    for t_idx, t in enumerate(range(visibility_sent_start_idx, ques_start_idx)):
                        visibility_sent_acts[bi, l, t_idx] = (
                            model.model.layers[l].output[0][0, t].cpu().save()
                        )

torch.save(last_token_acts, "../caches/Llama-70B-Instruct/BigToM/last_token_acts.pt")
torch.save(query_charac_acts, "../caches/Llama-70B-Instruct/BigToM/query_charac_acts.pt")
torch.save(visibility_sent_acts, "../caches/Llama-70B-Instruct/BigToM/visibility_sent_acts.pt")
torch.save(visibility_sent_lens, "../caches/Llama-70B-Instruct/BigToM/visibility_sent_lens.pt")
print("Done!")
