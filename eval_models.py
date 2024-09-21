import os
import csv
import json
import torch
import argparse
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from nnsight import LanguageModel, CONFIG
from utils import load_model_and_tokenizer, get_new_template_exps

warnings.filterwarnings("ignore")

CONFIG.set_default_api_key("6TnmrIokoS3Judkyez1F")
os.environ["HF_TOKEN"] = "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"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
if "mind" not in current_dir:
    current_dir = f"{current_dir}/mind"
print(f"Current directory: {current_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct"
    )
    parser.add_argument("--precision", type=str, default="int4")
    parser.add_argument("--ndif", type=bool, default=False)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_iterations", type=int, default=10)
    parser.add_argument("--event_noticed", type=bool, default=False)
    parser.add_argument("--question_type", type=str, default="true_state")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # Print arguments
    print(f"Model name: {args.model_name}")
    print(f"Precision: {args.precision}")
    print(f"NDIF: {args.ndif}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Number of iterations: {args.n_iterations}")
    print(f"Event noticed: {args.event_noticed}")
    print(f"Question type: {args.question_type}")
    print(f"Batch size: {args.batch_size}")

    os.makedirs(
        f"{current_dir}/preds/new_bigtom/",
        exist_ok=True,
    )

    if not args.ndif:
        model = LanguageModel(args.model_name, device_map="auto", load_in_4bit=True, torch_dtype=torch.float16, dispatch=True)
    else:
        model = LanguageModel(args.model_name)

    if "tokenizer" not in locals():
        tokenizer = model.tokenizer
    
    data_path = os.path.join(current_dir, "data", "new_bigtom_formatted.csv")
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    
    characters_path = os.path.join(current_dir, "data", "synthetic_entities", "actor.json")
    characters = json.load(open(characters_path, 'r', encoding='utf-8'))

    accs = {}
    with torch.no_grad():
        for iteration in range(args.n_iterations):
            dataset, _ = get_new_template_exps(data=data, 
                                               characters=characters, 
                                               n_samples=args.n_samples, 
                                               event_noticed=args.event_noticed, 
                                               question_type=args.question_type)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            correct, total = 0, 0
            for batch in tqdm(dataloader, total=len(dataloader)):
                if not args.ndif:
                    prompt = batch['prompt'][0]
                    target = batch['target'][0]

                    with model.trace(prompt, scan=False, validate=False):
                        pred = model.lm_head.output[0, -1].argmax(dim=-1).item().save()

                pred_text = tokenizer.decode([pred]).lower().strip()
                if pred_text == target:
                    correct += 1
                total += 1

                del batch, pred
                torch.cuda.empty_cache()

            acc = round(correct/total, 2)
            print(f"Accuracy: {acc}")

            accs[iteration] = acc
            # Save accs to a json file
            with open(f"{current_dir}/preds/new_bigtom/{args.model_name.split('/')[-1]}_accs.json", 'w', encoding='utf-8') as file:
                json.dump(accs, file)

            print(f"Iteration {iteration} completed")


    # home_dir = str(Path.home())
    # shutil.rmtree(
    #     f"{home_dir}/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"
    # )


if __name__ == "__main__":
    main()
