import json
import os
import sys
from collections import defaultdict

import fire
import pandas as pd
import torch
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the root directory to Python path
root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(root_dir)

from experiments.bigToM.utils import (
    get_answer_lookback_payload_exps,
    get_answer_lookback_pointer_exps,
    get_binding_lookback_pointer_exps,
    get_prompt_token_len,
    get_ques_start_token_idx,
    get_visibility_lookback_exps,
    get_visitibility_sent_start_idx,
)


def run_patching_experiment(model, dataloader, args):
    results = {}

    for layer_idx in args.patch_layers:
        correct, total = 0, 0

        for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            org_prompt = batch["org_prompt"][0]
            alt_prompt = batch["alt_prompt"][0]
            target = batch["target"][0]

            org_prompt_len = get_prompt_token_len(model.tokenizer, org_prompt)
            alt_prompt_len = get_prompt_token_len(model.tokenizer, alt_prompt)

            if args.experiment_type == "answer_lookback_payload":
                with torch.no_grad():
                    with model.session() as session:
                        alt_layer_out = defaultdict(dict)
                        with model.trace(alt_prompt):
                            alt_layer_out = (
                                model.model.layers[layer_idx].output[0][0, -1].save()
                            )

                        with model.generate(
                            org_prompt,
                            max_new_tokens=2,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=model.tokenizer.pad_token_id,
                            eos_token_id=model.tokenizer.eos_token_id,
                        ):
                            model.model.layers[layer_idx].output[0][0, -1] = (
                                alt_layer_out
                            )
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "answer_lookback_pointer":
                with torch.no_grad():
                    with model.session() as session:
                        alt_layer_out = defaultdict(dict)
                        with model.trace(alt_prompt):
                            alt_layer_out = (
                                model.model.layers[layer_idx].output[0][0, -1].save()
                            )

                        with model.generate(
                            org_prompt,
                            max_new_tokens=2,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=model.tokenizer.pad_token_id,
                            eos_token_id=model.tokenizer.eos_token_id,
                        ):
                            model.model.layers[layer_idx].output[0][0, -1] = (
                                alt_layer_out
                            )
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "binding_lookback_pointer":
                org_ques_start_idx = get_ques_start_token_idx(
                    model.tokenizer, org_prompt
                )
                alt_ques_start_idx = get_ques_start_token_idx(
                    model.tokenizer, alt_prompt
                )

                with torch.no_grad():
                    with model.session() as session:
                        alt_layer_out = defaultdict(dict)
                        with model.trace(alt_prompt):
                            for t_idx, t in enumerate(
                                [
                                    i
                                    for i in range(
                                        alt_ques_start_idx + 3, alt_ques_start_idx + 5
                                    )
                                ]
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
                                        org_ques_start_idx + 3, org_ques_start_idx + 5
                                    )
                                ]
                            ):
                                model.model.layers[layer_idx].output[0][0, t] = (
                                    alt_layer_out[t_idx]
                                )
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "visibility_lookback_source":
                org_vis_sent_start_idx = get_visitibility_sent_start_idx(
                    model.tokenizer, org_prompt
                )
                alt_vis_sent_start_idx = get_visitibility_sent_start_idx(
                    model.tokenizer, alt_prompt
                )

                with torch.no_grad():
                    with model.session() as session:
                        alt_layer_out = defaultdict(dict)
                        org_ques_start_idx = get_ques_start_token_idx(
                            model.tokenizer, org_prompt
                        )
                        alt_ques_start_idx = get_ques_start_token_idx(
                            model.tokenizer, alt_prompt
                        )

                        with model.trace(alt_prompt):
                            for t_idx, t in enumerate(
                                [
                                    i
                                    for i in range(
                                        alt_vis_sent_start_idx, alt_ques_start_idx
                                    )
                                ]
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
                                model.model.layers[layer_idx].output[0][0, t] = (
                                    alt_layer_out[t_idx]
                                )
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "visibility_lookback_payload":
                org_ques_start_idx = get_ques_start_token_idx(
                    model.tokenizer, org_prompt
                )
                alt_ques_start_idx = get_ques_start_token_idx(
                    model.tokenizer, alt_prompt
                )

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
                                model.model.layers[layer_idx].output[0][0, t] = (
                                    alt_layer_out[t_idx]
                                )
                            out = model.generator.output.save()

                        del alt_layer_out
                        torch.cuda.empty_cache()

            elif args.experiment_type == "visibility_lookback_address_pointer":
                org_vis_sent_start_idx = get_visitibility_sent_start_idx(
                    model.tokenizer, org_prompt
                )
                alt_vis_sent_start_idx = get_visitibility_sent_start_idx(
                    model.tokenizer, alt_prompt
                )
                org_ques_start_idx = get_ques_start_token_idx(
                    model.tokenizer, org_prompt
                )
                alt_ques_start_idx = get_ques_start_token_idx(
                    model.tokenizer, alt_prompt
                )

                with torch.no_grad():
                    with model.session() as session:
                        alt_vis_layer_out, alt_query_layer_out = (
                            defaultdict(dict),
                            defaultdict(dict),
                        )
                        with model.trace(alt_prompt):
                            for t_idx, t in enumerate(
                                [
                                    i
                                    for i in range(
                                        alt_vis_sent_start_idx, alt_ques_start_idx
                                    )
                                ]
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
                                model.model.layers[layer_idx].output[0][0, t] = (
                                    alt_vis_layer_out[t_idx]
                                )

                            for t_idx, t in enumerate(
                                [i for i in range(org_ques_start_idx, org_prompt_len)]
                            ):
                                model.model.layers[layer_idx].output[0][0, t] = (
                                    alt_query_layer_out[t_idx]
                                )

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
        results[layer_idx] = acc
        print(f"Layer: {layer_idx} | Accuracy: {acc}")

        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(
                os.path.join(
                    args.output_dir,
                    f"{args.experiment_type.split('_')[-1]}.json",
                ),
                "w",
            ) as f:
                json.dump(results, f, indent=4)


def main(
    data_dir: str = "data/bigtom",
    model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct",
    n_samples: int = 80,
    experiment_type: str = None,
    output_dir: str = "causal_model_2",
    verbose: bool = False,
    is_remote: bool = False,
):
    """
    Run BigToM experiments with residual vector patching.

    Args:
        data_dir: Directory containing BigToM data
        model_name: Model name to use
        experiment_type: Type of patching experiment to run (answer_lookback_payload, answer_lookback_pointer, binding_lookback_pointer, visibility_lookback_source, visibility_lookback_payload, visibility_lookback_address_pointer)
        n_samples: Number of samples to use
        output_dir: Directory to save results
        verbose: Print verbose output
        cache_dir: HuggingFace cache directory
    """
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(os.path.join(root_dir, data_dir))
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(
            os.path.join(root_dir, "experiments", "bigToM", "results", output_dir)
        )

    # Create output directory
    for lookback in ["answer_lookback", "binding_lookback", "visibility_lookback"]:
        lookback_output_dir = os.path.join(output_dir, lookback)
        os.makedirs(lookback_output_dir, exist_ok=True)

    # Load model
    print(f"Loading model {model_name}...")
    if is_remote:
        model = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct")
        model_name = "Llama-3.1-405B-Instruct"
    else:
        model = LanguageModel(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            dispatch=True,
        )
        model_name = "Llama-3-70B-Instruct"
    print("Model loaded successfully!")

    # Load data
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(os.path.join(root_dir, data_dir))

    print(f"Loading data from {data_dir}...")
    df_false = pd.read_csv(
        os.path.join(data_dir, "0_forward_belief_false_belief", "stories.csv"),
        delimiter=";",
    )
    df_true = pd.read_csv(
        os.path.join(data_dir, "0_forward_belief_true_belief", "stories.csv"),
        delimiter=";",
    )

    if not experiment_type:
        raise ValueError("experiment_type is required for patching mode")

    print(f"Running {experiment_type} patching experiment...")

    if experiment_type == "answer_lookback_payload":
        patch_layers = [i for i in range(0, 60, 10)] + [
            i for i in range(51, model.config.num_hidden_layers, 1)
        ]
        dataset = get_answer_lookback_payload_exps(df_false, df_true, n_samples)
        output_dir = os.path.join(output_dir, "answer_lookback")
    elif experiment_type == "answer_lookback_pointer":
        patch_layers = [i for i in range(0, 40, 10)] + [
            i for i in range(31, model.config.num_hidden_layers, 1)
        ]
        dataset = get_answer_lookback_pointer_exps(df_false, df_true, n_samples)
        output_dir = os.path.join(output_dir, "answer_lookback")
    elif experiment_type == "binding_lookback_pointer":
        patch_layers = [i for i in range(0, 41, 1)] + [
            i for i in range(50, model.config.num_hidden_layers, 10)
        ]
        dataset = get_binding_lookback_pointer_exps(df_false, df_true, n_samples)
        output_dir = os.path.join(output_dir, "binding_lookback")
    elif experiment_type == "visibility_lookback_source":
        patch_layers = [i for i in range(0, model.config.num_hidden_layers, 1)]
        dataset = get_visibility_lookback_exps(df_false, df_true, n_samples)
        output_dir = os.path.join(output_dir, "visibility_lookback")
    elif experiment_type == "visibility_lookback_payload":
        patch_layers = [i for i in range(0, model.config.num_hidden_layers, 1)]
        dataset = get_visibility_lookback_exps(df_false, df_true, n_samples)
        output_dir = os.path.join(output_dir, "visibility_lookback")
    elif experiment_type == "visibility_lookback_address_pointer":
        patch_layers = [i for i in range(0, model.config.num_hidden_layers, 1)]
        dataset = get_visibility_lookback_exps(df_false, df_true, n_samples)
        output_dir = os.path.join(output_dir, "visibility_lookback")
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    args = type(
        "Args",
        (),
        {
            "patch_layers": patch_layers,
            "experiment_type": experiment_type,
            "n_samples": n_samples,
            "output_dir": output_dir,
            "verbose": verbose,
        },
    )()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    run_patching_experiment(model, dataloader, args)

    print("Experiment completed successfully!")


if __name__ == "__main__":
    fire.Fire(main)
