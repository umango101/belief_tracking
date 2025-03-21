import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

import json
import random
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nnsight import CONFIG, LanguageModel

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

model = LanguageModel("meta-llama/Meta-Llama-3-70B-Instruct", cache_dir="/disk/u/nikhil/.cache/huggingface/hub/", device_map="auto", torch_dtype=torch.float16, dispatch=True)

n_samples = 500
batch_size = 1

first_visibility_sent = [i for i in range(169, 176)]
second_visibility_sent = [i for i in range(176, 183)]
charac_indices = [131, 133, 146, 147, 158, 159]
object_indices = [150, 151, 162, 163]
state_indices = [155, 156, 167, 168]
query_sent = [i for i in range(169, 181)]

configs = []
for _ in range(n_samples):
    template_idx = 0
    template = STORY_TEMPLATES["templates"][template_idx]
    characters = random.sample(all_characters, 2)
    containers = random.sample(all_containers[template["container_type"]], 2)
    states = random.sample(all_states[template["state_type"]], 2)

    sample = SampleV3(
        template_idx=template_idx,
        characters=characters,
        containers=containers,
        states=states,
    )
    configs.append(sample)

dataset = DatasetV3(configs)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

first_visibility_sent_acts = torch.zeros(n_samples, model.config.num_hidden_layers, len(first_visibility_sent), model.config.hidden_size)
second_visibility_sent_acts = torch.zeros(n_samples, model.config.num_hidden_layers, len(second_visibility_sent), model.config.hidden_size)

for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    prompt = data['prompt'][0]

    with torch.no_grad():

        with model.trace() as tracer:

            with tracer.invoke(prompt):
                for l in range(model.config.num_hidden_layers):
                    for t_idx, t in enumerate(first_visibility_sent):
                        first_visibility_sent_acts[bi, l, t_idx] = model.model.layers[l].output[0][0, t].cpu().save()
                    
                    for t_idx, t in enumerate(second_visibility_sent):
                        second_visibility_sent_acts[bi, l, t_idx] = model.model.layers[l].output[0][0, t].cpu().save()

torch.save(first_visibility_sent_acts, "../caches/belief_tracking/first_visibility_sent_acts.pt")
torch.save(second_visibility_sent_acts, "../caches/belief_tracking/second_visibility_sent_acts.pt")
print("Done!")