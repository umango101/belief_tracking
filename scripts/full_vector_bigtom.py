import sys
import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from nnsight import LanguageModel

sys.path.append("../")
from utils import (
    get_bigtom_value_fetcher_exps,
    get_bigtom_answer_state_exps,
    get_bigtom_query_charac,
    get_bigtom_visibility_exps,
)

from bigtom_utils import (
    get_ques_start_token_idx,
    get_visitibility_sent_start_idx,
    get_prompt_token_len,
)


def run_patching_experiment(model, dataset, args):
    test_dataset = dataset[args.train_size + args.valid_size :]
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_accs = {}

    for layer_idx in args.patch_layers:
        correct, total = 0, 0

        for bi, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            org_prompt = batch["org_prompt"][0]
            org_ans = batch["org_ans"][0]
            alt_prompt = batch["alt_prompt"][0]
            alt_ans = batch["alt_ans"][0]
            target = batch["target"][0]

            org_prompt_len = get_prompt_token_len(model.tokenizer, org_prompt)
            alt_prompt_len = get_prompt_token_len(model.tokenizer, alt_prompt)

            if args.experiment_type == "answer":
                with torch.no_grad():
                    with model.session() as session:
                        alt_layer_out = defaultdict(dict)
                        with model.trace(alt_prompt):
                            alt_layer_out = model.model.layers[layer_idx].output[0][0, -1].save()

                        with model.generate(
                            org_prompt,
                            max_new_tokens=2,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=model.tokenizer.pad_token_id,
                            eos_token_id=model.tokenizer.eos_token_id,
                        ):
                            model.model.layers[layer_idx].output[0][0, -1] = alt_layer_out
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "answer_state":
                with torch.no_grad():
                    with model.session() as session:
                        alt_layer_out = defaultdict(dict)
                        with model.trace(alt_prompt):
                            alt_layer_out = model.model.layers[layer_idx].output[0][0, -1].save()

                        with model.generate(
                            org_prompt,
                            max_new_tokens=2,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=model.tokenizer.pad_token_id,
                            eos_token_id=model.tokenizer.eos_token_id,
                        ):
                            model.model.layers[layer_idx].output[0][0, -1] = alt_layer_out
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "query_character":
                org_ques_start_idx = get_ques_start_token_idx(model.tokenizer, org_prompt)
                alt_ques_start_idx = get_ques_start_token_idx(model.tokenizer, alt_prompt)

                with torch.no_grad():
                    with model.session() as session:
                        alt_layer_out = defaultdict(dict)
                        with model.trace(alt_prompt):
                            for t_idx, t in enumerate(
                                [i for i in range(alt_ques_start_idx + 3, alt_ques_start_idx + 5)]
                            ):
                                alt_layer_out[t_idx] = (
                                    model.model.layers[layer_idx].output[0][0, t].save()
                                )

                        with model.generate(
                            org_prompt,
                            max_new_tokens=2,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=model.tokenizer.pad_token_id,
                            eos_token_id=model.tokenizer.eos_token_id,
                        ):
                            for t_idx, t in enumerate(
                                [i for i in range(org_ques_start_idx + 3, org_ques_start_idx + 5)]
                            ):
                                model.model.layers[layer_idx].output[0][0, t] = alt_layer_out[t_idx]
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "visibility_sent":
                org_vis_sent_start_idx = get_visitibility_sent_start_idx(
                    model.tokenizer, org_prompt
                )
                alt_vis_sent_start_idx = get_visitibility_sent_start_idx(
                    model.tokenizer, alt_prompt
                )

                with torch.no_grad():
                    with model.session() as session:
                        alt_layer_out = defaultdict(dict)
                        org_ques_start_idx = get_ques_start_token_idx(model.tokenizer, org_prompt)
                        alt_ques_start_idx = get_ques_start_token_idx(model.tokenizer, alt_prompt)

                        with model.trace(alt_prompt):
                            for t_idx, t in enumerate(
                                [i for i in range(alt_vis_sent_start_idx, alt_ques_start_idx)]
                            ):
                                alt_layer_out[t_idx] = (
                                    model.model.layers[layer_idx].output[0][0, t].save()
                                )

                        with model.generate(
                            org_prompt,
                            max_new_tokens=2,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=model.tokenizer.pad_token_id,
                            eos_token_id=model.tokenizer.eos_token_id,
                        ):
                            for t_idx, t in enumerate(
                                [
                                    i
                                    for i in range(
                                        org_vis_sent_start_idx,
                                        org_vis_sent_start_idx + len(alt_layer_out),
                                    )
                                ]
                            ):
                                model.model.layers[layer_idx].output[0][0, t] = alt_layer_out[t_idx]
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "query_sent":
                org_ques_start_idx = get_ques_start_token_idx(model.tokenizer, org_prompt)
                alt_ques_start_idx = get_ques_start_token_idx(model.tokenizer, alt_prompt)

                with torch.no_grad():
                    with model.session() as session:
                        alt_layer_out = defaultdict(dict)
                        with model.trace(alt_prompt):
                            for t_idx, t in enumerate(
                                [i for i in range(alt_ques_start_idx, alt_prompt_len)]
                            ):
                                alt_layer_out[t_idx] = (
                                    model.model.layers[layer_idx].output[0][0, t].save()
                                )

                        with model.generate(
                            org_prompt,
                            max_new_tokens=2,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=model.tokenizer.pad_token_id,
                            eos_token_id=model.tokenizer.eos_token_id,
                        ):
                            for t_idx, t in enumerate(
                                [i for i in range(org_ques_start_idx, org_prompt_len)]
                            ):
                                model.model.layers[layer_idx].output[0][0, t] = alt_layer_out[t_idx]
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "vis_query_sent":
                org_vis_sent_start_idx = get_visitibility_sent_start_idx(
                    model.tokenizer, org_prompt
                )
                alt_vis_sent_start_idx = get_visitibility_sent_start_idx(
                    model.tokenizer, alt_prompt
                )
                org_ques_start_idx = get_ques_start_token_idx(model.tokenizer, org_prompt)
                alt_ques_start_idx = get_ques_start_token_idx(model.tokenizer, alt_prompt)

                with torch.no_grad():
                    with model.session() as session:
                        alt_vis_layer_out, alt_query_layer_out = defaultdict(dict), defaultdict(
                            dict
                        )
                        with model.trace(alt_prompt):
                            for t_idx, t in enumerate(
                                [i for i in range(alt_vis_sent_start_idx, alt_ques_start_idx)]
                            ):
                                alt_vis_layer_out[t_idx] = (
                                    model.model.layers[layer_idx].output[0][0, t].save()
                                )

                            for t_idx, t in enumerate(
                                [i for i in range(alt_ques_start_idx, alt_prompt_len)]
                            ):
                                alt_query_layer_out[t_idx] = (
                                    model.model.layers[layer_idx].output[0][0, t].save()
                                )

                        with model.generate(
                            org_prompt,
                            max_new_tokens=2,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=model.tokenizer.pad_token_id,
                            eos_token_id=model.tokenizer.eos_token_id,
                        ):
                            for t_idx, t in enumerate(
                                [
                                    i
                                    for i in range(
                                        org_vis_sent_start_idx,
                                        org_vis_sent_start_idx + len(alt_vis_layer_out),
                                    )
                                ]
                            ):
                                model.model.layers[layer_idx].output[0][0, t] = alt_vis_layer_out[
                                    t_idx
                                ]

                            for t_idx, t in enumerate(
                                [i for i in range(org_ques_start_idx, org_prompt_len)]
                            ):
                                model.model.layers[layer_idx].output[0][0, t] = alt_query_layer_out[
                                    t_idx
                                ]

                            out = model.generator.output.save()

                        del alt_vis_layer_out, alt_query_layer_out
                        torch.cuda.empty_cache()

            else:
                raise ValueError(f"Unknown experiment type: {args.experiment_type}")

            pred = model.tokenizer.decode(out[0][org_prompt_len:-1]).strip()
            if args.verbose:
                print(f"Pred: {pred} | Target: {target}")

            if pred in target:
                correct += 1
            total += 1

        acc = round(correct / total, 2)
        test_accs[layer_idx] = acc
        print(f"Layer: {layer_idx} | Accuracy: {acc}")

        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(f"{args.output_dir}/{args.experiment_type}.json", "w") as f:
                json.dump(test_accs, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Run BigToM experiments")
    parser.add_argument(
        "--data_dir", type=str, default="../data/bigtom", help="Directory containing BigToM data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        choices=[
            "answer",
            "answer_state",
            "query_character",
            "visibility_sent",
            "query_sent",
            "vis_query_sent",
        ],
        help="Type of patching experiment to run",
    )
    parser.add_argument("--train_size", type=int, default=80, help="Size of training set")
    parser.add_argument("--valid_size", type=int, default=40, help="Size of validation set")
    parser.add_argument("--test_size", type=int, default=80, help="Size of test set")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../bigtom_patching_results",
        help="Directory to save results",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/disk/u/nikhil/.cache/huggingface/hub/",
        help="HuggingFace cache directory",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model {args.model_name}...")
    model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16, "dispatch": True}
    if args.cache_dir:
        model_kwargs["cache_dir"] = args.cache_dir

    model = LanguageModel(args.model_name, **model_kwargs)

    # Ensure model configuration is loaded
    if not hasattr(model, "config") or model.config is None:
        raise ValueError(
            "Model configuration is not properly loaded. Check the model initialization."
        )
    print("Model loaded successfully!")

    # Load data
    print(f"Loading data from {args.data_dir}...")
    df_false = pd.read_csv(
        f"{args.data_dir}/0_forward_belief_false_belief/stories.csv", delimiter=";"
    )
    df_true = pd.read_csv(
        f"{args.data_dir}/0_forward_belief_true_belief/stories.csv", delimiter=";"
    )

    if not args.experiment_type:
        parser.error("--experiment_type is required for patching mode")

    print(f"Running {args.experiment_type} patching experiment...")

    if args.experiment_type == "answer":
        args.patch_layers = [i for i in range(0, 60, 10)] + [
            i for i in range(51, model.config.num_hidden_layers, 1)
        ]
        dataset = get_bigtom_value_fetcher_exps(
            df_false, df_true, args.train_size + args.valid_size + args.test_size
        )
    elif args.experiment_type == "answer_state":
        args.patch_layers = [i for i in range(0, 40, 10)] + [
            i for i in range(31, model.config.num_hidden_layers, 1)
        ]
        dataset = get_bigtom_answer_state_exps(
            df_false, df_true, args.train_size + args.valid_size + args.test_size
        )
    elif args.experiment_type == "query_character":
        args.patch_layers = [i for i in range(0, 41, 1)] + [
            i for i in range(50, model.config.num_hidden_layers, 10)
        ]
        dataset = get_bigtom_query_charac(
            df_false, df_true, args.train_size + args.valid_size + args.test_size
        )
    elif args.experiment_type == "visibility_sent":
        args.patch_layers = [i for i in range(0, model.config.num_hidden_layers, 1)]
        dataset = get_bigtom_visibility_exps(
            df_false, df_true, args.train_size + args.valid_size + args.test_size
        )
    elif args.experiment_type == "query_sent":
        args.patch_layers = [i for i in range(0, model.config.num_hidden_layers, 1)]
        dataset = get_bigtom_visibility_exps(
            df_false, df_true, args.train_size + args.valid_size + args.test_size
        )
    elif args.experiment_type == "vis_query_sent":
        args.patch_layers = [i for i in range(0, model.config.num_hidden_layers, 1)]
        dataset = get_bigtom_visibility_exps(
            df_false, df_true, args.train_size + args.valid_size + args.test_size
        )
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment_type}")

    run_patching_experiment(model, dataset, args)

    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()
