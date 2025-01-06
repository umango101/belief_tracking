import random
import os
import json
import torch
import argparse
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from nnsight import LanguageModel, CONFIG
from src.dataset import SampleV3, DatasetV3, STORY_TEMPLATES
from src.utils import env_utils
# from utils import *

warnings.filterwarnings("ignore")

CONFIG.set_default_api_key("d9e00ab7d4f74643b3176de0913f24a7")
os.environ["HF_TOKEN"] = "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"

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# Get current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
if "mind" not in current_dir:
    current_dir = f"{current_dir}/mind"
print(f"Current directory: {current_dir}")

random.seed(10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-405B"
    )
    parser.add_argument("--precision", type=str, default="int4")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_iterations", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # Print arguments
    print(f"Model name: {args.model_name}")
    print(f"Precision: {args.precision}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Number of iterations: {args.n_iterations}")
    print(f"Batch size: {args.batch_size}")

    os.makedirs(
        f"{current_dir}/evals/",
        exist_ok=True,
    )
    
    all_states = {}
    all_containers = {}
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

    accs = {}
    with torch.no_grad():
        print(f"Loading {args.model_name} ...")
        model = LanguageModel(args.model_name)

        accs[args.model_name] = {
            'visibility': [],
            'no_visibility': []
        }

        for iteration in range(args.n_iterations):
            configs_1, configs_2 = [], []
            correct_vis, total = 0, 0

            for _ in range(args.n_samples):
                template_1 = STORY_TEMPLATES['templates'][0]
                template_2 = STORY_TEMPLATES['templates'][1]
                characters = random.sample(all_characters, 2)
                containers = random.sample(all_containers[template_1["container_type"]], 2)
                states = random.sample(all_states[template_1["state_type"]], 2)
                event_idx = None
                event_noticed = False

                sample = SampleV3(
                    template=template_2,
                    characters=characters,
                    containers=containers,
                    states=states,
                    visibility=True,
                    event_idx=event_idx,
                    event_noticed=event_noticed,
                )
                configs_1.append(sample)

                # sample = SampleV3(
                #     template=template_2,
                #     characters=characters,
                #     containers=containers,
                #     states=states,
                #     visibility=True,
                #     event_idx=event_idx,
                #     event_noticed=event_noticed,
                # )
                # configs_2.append(sample)

            dataset_1 = DatasetV3(configs_1)
            # dataset_2 = DatasetV3(configs_2)
            dataloader_1 = DataLoader(dataset_1, batch_size=1, shuffle=False)
            # dataloader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False)

            for i, data_1 in tqdm(enumerate(dataloader_1), total=len(dataloader_1)):
                prompt_1, target_1 = data_1['prompt'][0], data_1['target'][0]
                # print(prompt_1, target_1)
                # prompt_2, target_2 = data_2['prompt'][0], data_2['target'][0]

                with model.session(remote=True):

                    with model.trace() as tracer:

                        with tracer.invoke(prompt_1):
                            pred_1 = model.lm_head.output[0, -1].argmax(dim=-1).save()

                        # with tracer.invoke(prompt_2):
                        #     pred_2 = model.lm_head.output[0, -1].argmax(dim=-1).save()

                pred_1 = model.tokenizer.decode([pred_1]).lower().strip()
                # pred_2 = model.tokenizer.decode([pred_2]).lower().strip()

                print(f"Prediction: {pred_1} | Target: {target_1}")
                if pred_1 in target_1:
                    correct_vis += 1
                # if pred_2 in target_2:
                #     correct_vis += 1
                total += 1

                del pred_1
                torch.cuda.empty_cache()

            acc_vis = round(correct_vis/total, 2)
            # acc_no_vis = round(correct_no_vis/total, 2)
            print(f"Model name: {args.model_name} | Iteration: {iteration} | Vis acc: {acc_vis}")

            accs[args.model_name]['visibility'].append(acc_vis)
            # accs[args.model_name]['no_visibility'].append(acc_no_vis)

            # Save accs to a json file
            with open(f"{current_dir}/evals/{args.model_name.split('/')[-1]}_accs.json", 'w', encoding='utf-8') as file:
                json.dump(accs, file)

if __name__ == "__main__":
    main()
