import os
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
from utils import load_model_and_tokenizer, load_bigtom

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
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ndif", type=bool, default=False)
    parser.add_argument("--method_name", type=str, default="2shots")
    parser.add_argument("--variable", type=str, default="0_forward_belief")
    parser.add_argument("--condition", type=str, default="true_belief")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Print arguments
    print(f"Model name: {args.model_name}")
    print(f"Precision: {args.precision}")
    print(f"NDIF: {args.ndif}")
    print(f"Method name: {args.method_name}")
    print(f"Variable: {args.variable}")
    print(f"Condition: {args.condition}")
    print(f"Batch size: {args.batch_size}")

    os.makedirs(
        f"{current_dir}/preds/bigtom/{args.method_name}/{args.variable}_{args.condition}",
        exist_ok=True,
    )

    if not args.ndif:
        model, tokenizer = load_model_and_tokenizer(
            args.model_name, args.precision, device
        )
    else:
        model = LanguageModel(args.model_name)

    if "tokenizer" not in locals():
        tokenizer = model.tokenizer

    dataloader = load_bigtom(
        model.config,
        tokenizer,
        current_dir,
        batch_size=args.batch_size,
        method_name=args.method_name,
        variable=args.variable,
        condition=args.condition,
    )
    print(f"{args.model_name} and Data loaded successfully")

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, inp in tqdm(enumerate(dataloader), total=len(dataloader)):
            if not args.ndif:
                inp["input_ids"] = inp["input_ids"].to(device)
                inp["target"] = inp["target"].to(device)
                outputs = model(inp["input_ids"])
                logits = outputs.logits[:, -1]
                pred_token_ids = torch.argmax(logits, dim=-1)

            if args.ndif:
                with model.trace(
                    inp["input_ids"], scan=False, validate=False, remote=True
                ):
                    pred_token_ids = torch.argmax(
                        model.output["logits"][:, -1], dim=-1
                    ).save()

            for idx in range(len(inp["target"])):
                input_text = tokenizer.decode(inp["input_ids"][idx].tolist()).strip()
                target_text = tokenizer.decode(inp["target"][idx].tolist()).strip()
                pred_text = tokenizer.decode(pred_token_ids[idx].tolist()).strip()

                if pred_text == target_text:
                    correct += 1
                total += 1

                with open(
                    f"{current_dir}/preds/bigtom/{args.method_name}/{args.variable}_{args.condition}/{args.model_name.split('/')[-1]}.jsonl",
                    "a",
                ) as f:
                    f.write(
                        json.dumps(
                            {
                                "pred": pred_text,
                                "target": target_text,
                                "input": input_text,
                            }
                        )
                        + "\n"
                    )

            del inp, outputs, logits, pred_token_ids
            torch.cuda.empty_cache()

    print(f"Accuracy: {round(correct/total, 2)}")

    with open(
        f"{current_dir}/preds/bigtom/{args.method_name}/{args.variable}_{args.condition}/results.txt",
        "a",
    ) as f:
        f.write(f"Model: {args.model_name}\nAccuracy: {round(correct/total, 2)}\n\n")

    del model
    torch.cuda.empty_cache()

    # home_dir = str(Path.home())
    # shutil.rmtree(
    #     f"{home_dir}/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"
    # )


if __name__ == "__main__":
    main()
