import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

import json
import random
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import nnsight
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
CONFIG.APP.REMOTE_LOGGING = True

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

model = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct")

n_samples = 500
batch_size = 1

first_visibility_sent = [i for i in range(169, 176)]
second_visibility_sent = [i for i in range(176, 183)]
charac_indices = [131, 133, 146, 147, 158, 159]
object_indices = [150, 151, 162, 163]
state_indices = [155, 156, 167, 168]
query_sent_with_vis = [i for i in range(183, 195)]
query_sent_no_vis = [i for i in range(169, 181)]

configs = []
for _ in range(n_samples):
    template_idx = 1
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


first_vis_acts = torch.zeros(n_samples, model.config.num_hidden_layers, len(first_visibility_sent), model.config.hidden_size).cpu()
second_vis_acts = torch.zeros(n_samples, model.config.num_hidden_layers, len(second_visibility_sent), model.config.hidden_size).cpu()
query_vis_acts = torch.zeros(n_samples, model.config.num_hidden_layers, len(query_sent_no_vis), model.config.hidden_size).cpu()

for i in tqdm(range(len(configs)//20)):
    n_samples = 20
    dataset = DatasetV3(configs[i*20:(i+1)*20])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with model.session(remote=True) as session:
        first_vis_tmp = torch.zeros(n_samples, model.config.num_hidden_layers, len(first_visibility_sent), model.config.hidden_size).cpu().save()
        second_vis_tmp = torch.zeros(n_samples, model.config.num_hidden_layers, len(second_visibility_sent), model.config.hidden_size).cpu().save()
        query_tmp = torch.zeros(n_samples, model.config.num_hidden_layers, len(query_sent_with_vis), model.config.hidden_size).cpu().save()

        bi = nnsight.list([0])
        with session.iter(dataloader) as data:
            prompt = data["prompt"]

            with model.trace(prompt) as tracer:
                for l in range(model.config.num_hidden_layers):
                    for t_idx, t in enumerate(first_visibility_sent):
                        first_vis_tmp[bi[-1], l, t_idx] = model.model.layers[l].output[0][0, t].cpu()

                    for t_idx, t in enumerate(second_visibility_sent):
                        second_vis_tmp[bi[-1], l, t_idx] = model.model.layers[l].output[0][0, t].cpu()

                    for t_idx, t in enumerate(query_sent_with_vis):
                        query_tmp[bi[-1], l, t_idx] = model.model.layers[l].output[0][0, t].cpu()

            bi.append(bi[-1] + 1)

    first_vis_acts[i*20:(i+1)*20] = first_vis_tmp.value
    second_vis_acts[i*20:(i+1)*20] = second_vis_tmp.value
    query_vis_acts[i*20:(i+1)*20] = query_tmp.value


    if i % 5 == 0:
        torch.save(first_vis_acts, "../caches/llama-405B-Instruct/first_vis_acts.pt")
        torch.save(second_vis_acts, "../caches/llama-405B-Instruct/second_vis_acts.pt")
        torch.save(query_vis_acts, "../caches/llama-405B-Instruct/query_vis_acts.pt")


torch.save(first_vis_acts, "../caches/llama-405B-Instruct/first_vis_acts.pt")
torch.save(second_vis_acts, "../caches/llama-405B-Instruct/second_vis_acts.pt")
torch.save(query_vis_acts, "../caches/llama-405B-Instruct/query_vis_acts.pt")
print("Done!")