import argparse
import json
import os
from typing import Literal

import torch
from nnsight import CONFIG, LanguageModel

CONFIG.API.HOST = "nagoya.research.khoury.northeastern.edu:5001"
CONFIG.API.SSL = False
CONFIG.API.APIKEY = "hi"  # API key needs to be non-empty due to NDIF bug

from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.run_exp_utils import (
    Accuracy,
    exp_to_intervention_positions,
    exp_to_vec_type,
    free_gpu_cache,
    load_basis_directions,
    prepare_dataset,
    set_seed,
)
from src.utils import env_utils

query_object_indices = [-5, -4]
query_character_indices = [-8, -7]


@torch.inference_mode()
def validate(
    exp_name: Literal[
        "position_transmitter",  # *
        "value_fetcher",  # *
        "query_object",  # *
        "query_character",  # *
        "state_position",
        "vis_2nd",
        "vis_ques",
        "vis_2nd_and_ques",
        "vis_2nd_to_1st_and_ques",  # X
        "pointer",
    ],
    lm: LanguageModel,
    layer_idx: int,
    validation_loader: DataLoader,
    projection: torch.Tensor | dict[str, torch.Tensor] | None = None,
    verbose: bool = False,
    save_outputs: bool = True,
    projection_type: Literal[
        "full_rank", "singular_vector", "principal_component"
    ] = "full_rank",
    remote: bool = False,
) -> float:
    # lm.tokenizer.padding_side = "left"
    # lm.model.eval()

    save_outputs = save_outputs and not remote

    if save_outputs:
        save_path = os.path.join(
            "results",
            "lm_pred_on_val_set",
            lm.config._name_or_path.split("/")[-1],
            # "test",
            exp_name,
            f"layer_{layer_idx}",
            projection_type,
        )
        os.makedirs(save_path, exist_ok=True)
        valid_data = validation_loader.dataset

    intervention_positions = exp_to_intervention_positions[exp_name]
    patch_to_cache_map = {
        k: v
        for k, v in zip(
            intervention_positions["patch"], intervention_positions["cache"]
        )
    }
    # print(patch_to_cache_map)

    correct, total = 0, 0
    for batch_idx, batch in tqdm(
        enumerate(validation_loader), total=len(validation_loader)
    ):
        alt_prompts = batch["corrupt_prompt"]
        org_prompts = batch["clean_prompt"]
        targets = batch["target"] if "target" in batch else batch["corrupt_target"]
        target_tokens = lm.tokenizer(targets, return_tensors="pt").input_ids[:, -1]
        batch_size = target_tokens.size(0)
        alt_acts, org_acts_state = defaultdict(dict), defaultdict(dict)

        def nnsight_request(return_logits: bool = False):
            with lm.trace(remote=remote) as tracer:
                # tracer.log("alt_prompts", alt_prompts)
                with tracer.invoke(alt_prompts):
                    for t in intervention_positions["cache"]:
                        alt_acts[t] = lm.model.layers[layer_idx].output[0][:, t].clone()

                # tracer.log("org_prompts", org_prompts)
                with tracer.invoke(org_prompts):
                    for t in intervention_positions["patch"]:
                        curr_output = lm.model.layers[layer_idx].output[0][:, t].clone()
                        if projection is not None:
                            if isinstance(projection, dict):
                                if t in query_object_indices:
                                    proj = projection["query_obj_ordering_id"]
                                elif t in query_character_indices:
                                    proj = projection["query_charac_ordering_id"]
                                else:
                                    raise ValueError("Invalid projection type")
                            else:
                                proj = projection
                            alt_proj = torch.matmul(
                                alt_acts[patch_to_cache_map[t]], proj
                            )
                            org_proj = torch.matmul(curr_output, proj)
                            patch = curr_output - org_proj + alt_proj

                            del alt_proj, org_proj
                            free_gpu_cache()
                        else:
                            patch = alt_acts[patch_to_cache_map[t]]

                        lm.model.layers[layer_idx].output[0][:, t] = patch

                    logits = lm.lm_head.output[:, -1]
                    logits = logits.save() if return_logits else logits
                    pred = torch.argmax(logits, dim=-1).save()

            return logits, pred

        # logits, pred = (
        #     nnsight_request(return_logits=not remote)
        #     if not remote
        #     else send_request_to_ndif(nnsight_request, timeout=1800, n_try=5)
        # )
        logits, pred = nnsight_request(return_logits=not remote)

        pred = pred.cpu()

        for i in range(batch_size):
            pred_token = lm.tokenizer.decode(pred[i])
            is_correct = pred_token.lower().strip() == targets[i].lower().strip()
            if verbose:
                print(
                    f"Predicted: {pred_token.lower().strip()}, Target: {targets[i].lower().strip()}"
                )
            correct += int(is_correct)

            if save_outputs:
                sample = valid_data[batch_idx * batch_size + i]
                with open(
                    os.path.join(save_path, f"sample_{total}.json"),
                    "w",
                ) as f:
                    json.dump(
                        {
                            "sample": sample,
                            "pred": {
                                "token": pred_token,
                                "token_id": pred[i].item(),
                                "logit": logits[i, pred[i]].item(),
                            },
                            "logit_distribution": logits[i].tolist(),
                            "is_correct": is_correct,
                        },
                        f,
                        indent=4,
                    )

            total += 1

        del alt_acts, alt_prompts, org_prompts, targets, target_tokens, logits, pred
        free_gpu_cache()

    return correct / total


def get_low_rank_projection(
    exp_name: Literal[
        "position_transmitter",
        "value_fetcher",
        "query_object",
        "query_character",
        "state_position",
        "vis_2nd",
        "vis_ques",
        "vis_2nd_and_ques",
        "vis_2nd_to_1st_and_ques",
        "pointer",
    ],
    lm: LanguageModel,
    layer_idx: int,
    train_loader: DataLoader,
    basis_directions: torch.Tensor | dict[str, torch.Tensor],
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.1,
    verbose: bool = False,
    remote: bool = False,
) -> tuple[torch.Tensor, dict]:
    if remote == True:
        raise NotImplementedError("Training not tested for remote yet")

    intervention_positions = exp_to_intervention_positions[exp_name]
    patch_to_cache_map = {
        k: v
        for k, v in zip(
            intervention_positions["patch"], intervention_positions["cache"]
        )
    }

    if isinstance(basis_directions, dict) == False:
        basis_indices = list(range(basis_directions.size(0)))
        mask = torch.ones(
            len(basis_indices), requires_grad=True, device="cuda", dtype=torch.bfloat16
        )
        basis_directions = basis_directions.to("cuda")
        optimizer = torch.optim.Adam([mask], lr=learning_rate)
    else:
        masks = {}
        for key in basis_directions:
            basis_indices = list(range(basis_directions[key].size(0)))
            masks[key] = torch.ones(
                len(basis_indices),
                requires_grad=True,
                device="cuda",
                dtype=torch.bfloat16,
            )
            basis_directions[key] = basis_directions[key].to("cuda")

        optimizer = torch.optim.Adam([{"params": masks.values()}], lr=learning_rate)

    # training loop
    for epoch in range(n_epochs):
        epoch_loss = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            alt_prompts = batch["corrupt_prompt"]
            org_prompts = batch["clean_prompt"]
            targets = batch["target"] if "target" in batch else batch["corrupt_target"]
            target_tokens = lm.tokenizer(targets, return_tensors="pt").input_ids[:, -1]
            batch_size = target_tokens.size(0)
            alt_acts, org_acts_state = defaultdict(dict), defaultdict(dict)

            if isinstance(basis_directions, dict) == False:
                masked_directions = basis_directions * mask.unsqueeze(-1)
                proj_matrix = torch.matmul(masked_directions.T, masked_directions).to(
                    lm.dtype
                )
            else:
                proj_matrix = {}
                for key in basis_directions:
                    masked_directions = basis_directions[key] * masks[key].unsqueeze(-1)
                    proj_matrix[key] = torch.matmul(
                        masked_directions.T, masked_directions
                    ).to(lm.dtype)

            with lm.trace() as tracer:
                with tracer.invoke(alt_prompts):
                    for t in intervention_positions["cache"]:
                        alt_acts[t] = lm.model.layers[layer_idx].output[0][:, t].clone()

                with tracer.invoke(org_prompts):
                    for t in intervention_positions["patch"]:
                        if isinstance(basis_directions, dict) == False:
                            proj = proj_matrix
                        else:
                            if t in query_object_indices:
                                proj = proj_matrix["query_obj_ordering_id"]
                            elif t in query_character_indices:
                                proj = proj_matrix["query_charac_ordering_id"]
                            else:
                                raise ValueError("Invalid projection type")

                        curr_output = lm.model.layers[layer_idx].output[0][:, t].clone()
                        alt_proj = torch.matmul(alt_acts[patch_to_cache_map[t]], proj)
                        org_proj = torch.matmul(curr_output, proj)
                        lm.model.layers[layer_idx].output[0][:, t] = (
                            curr_output - org_proj + alt_proj
                        )

                    logits = lm.lm_head.output[:, -1].save()

                    del alt_acts, org_proj
                    free_gpu_cache()

            target_logit = logits[torch.arange(batch_size), target_tokens]
            task_loss = -torch.mean(target_logit)
            if isinstance(basis_directions, dict) is False:
                l1_loss = lamb * torch.norm(mask, p=1)
            else:
                l1_loss = 0
                for key in masks:
                    l1_loss += lamb * torch.norm(masks[key], p=1)
            loss = task_loss + l1_loss.to(task_loss.device)

            if verbose:
                if isinstance(basis_directions, dict) is False:
                    mask_data = mask.data.clone().clamp(0, 1).round()
                    cur_rank = mask_data.sum().item()
                else:
                    cur_rank = {}
                    for key in masks:
                        mask_data = masks[key].data.clone().clamp(0, 1).round()
                        cur_rank[key] = mask_data.sum().item()

                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Rank: {cur_rank} ,Loss: {loss.item()} |>> l_task: {task_loss.item()}, l1: {l1_loss.item()}"
                )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clamp after optimizer step
            with torch.no_grad():
                if isinstance(basis_directions, dict) is False:
                    mask.data.clamp_(0, 1)
                else:
                    for key in masks:
                        masks[key].data.clamp_(0, 1)

            epoch_loss += loss.item()

            del logits, proj_matrix
            free_gpu_cache()

    # build the projection after training
    if isinstance(basis_directions, dict) is False:
        mask_data = mask.data.clone()
        mask_data.clamp_(0, 1)
        rounded = torch.round(mask_data)

        masked_directions = basis_directions * rounded.unsqueeze(-1)
        proj_matrix = torch.matmul(masked_directions.T, masked_directions).to(lm.dtype)

        metadata = {"mask": rounded.tolist(), "rank": rounded.sum().item()}

    else:
        rounded = {}
        proj_matrix = {}
        for key in masks:
            mask_data = masks[key].data.clone()
            mask_data.clamp_(0, 1)
            rounded[key] = torch.round(mask_data)

            masked_directions = basis_directions[key] * rounded[key].unsqueeze(-1)
            proj_matrix[key] = torch.matmul(masked_directions.T, masked_directions).to(
                lm.dtype
            )

        metadata = {
            "mask": {k: v.tolist() for k, v in rounded.items()},
            "rank": {k: v.sum().item() for k, v in rounded.items()},
        }

    return proj_matrix, metadata


def run_experiment(
    experiment_name: Literal[
        "position_transmitter",
        "value_fetcher",
        "query_object",
        "query_character",
        "state_position",
        "vis_2nd",
        "vis_ques",
        "vis_2nd_and_ques",
        "vis_2nd_to_1st_and_ques",
        "pointer",
    ],
    lm: LanguageModel,
    layers: list[int] = list(range(34, 60, 2)),
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    verbose: bool = False,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.1,
    save_path: str = "results",
    save_outputs_on_val: bool = True,
    remote: bool = False,
):
    print("#" * 30)
    print(f"Running experiment: {experiment_name}")
    print("#" * 30)

    lm_shorthand = lm.config._name_or_path.split("/")[-1]
    if remote:
        lm_shorthand = f"{lm_shorthand}-8bit-remote"
    save_path = os.path.join(save_path, lm_shorthand, experiment_name)
    os.makedirs(save_path, exist_ok=True)

    train_dataloader, valid_dataloader = prepare_dataset(
        lm=lm,
        experiment_name=experiment_name,
        train_size=train_size,
        valid_size=validation_size,
        batch_size=batch_size,
        remote=remote,
    )

    singular_vectors, principal_components = None, None
    exclude_projections = []
    # exclude_projections = [
    #     "position_transmitter",
    #     "value_fetcher",
    #     "pointer",
    #     "query_object",
    #     "query_character",
    # ]

    skip_low_rank_projection = (experiment_name in exclude_projections) or (
        "405B" in lm.config._name_or_path
    )

    if skip_low_rank_projection == False:
        if experiment_name.startswith("vis") == False or experiment_name == "vis_2nd":
            direction_type = exp_to_vec_type[experiment_name]
            if isinstance(direction_type, list):
                directions = {
                    d: load_basis_directions("singular_vecs", d) for d in direction_type
                }
                singular_vectors = {
                    l: {d: directions[d][l] for d in direction_type} for l in layers
                }
            else:
                singular_vectors = load_basis_directions(
                    "singular_vecs", exp_to_vec_type[experiment_name]
                )

        if experiment_name.startswith("vis") == False:
            if isinstance(direction_type, list):
                directions = {
                    d: load_basis_directions("principal_components", d)
                    for d in direction_type
                }
                principal_components = {
                    l: {d: directions[d][l] for d in direction_type} for l in layers
                }
            else:
                principal_components = load_basis_directions(
                    "principal_components", exp_to_vec_type[experiment_name]
                )

    # performances = {
    #     "full_state_patching": {l: None for l in layers},
    #     "singular_vector_patching": {l: None for l in layers},
    #     "principal_component_patching": {l: None for l in layers},
    # }

    print("Running experiment === > ", experiment_name)
    progress_bar = tqdm(layers)
    for layer in progress_bar:
        progress_bar.set_description(f"Layer: {layer}")
        print(f"\n{'-' * 10} Layer: {layer} {'-' * 10}")

        layer_performance = {}

        # full state patching
        full_acc = validate(
            experiment_name,
            lm,
            layer,
            valid_dataloader,
            verbose=verbose,
            save_outputs=save_outputs_on_val,
            projection_type="full_rank",
            remote=remote,
        )
        print("-" * 30)
        print(f"Full state patching val: {full_acc}")
        print("-" * 30)
        layer_performance["full_rank"] = Accuracy(
            accuracy=full_acc, rank=None
        ).to_dict()

        if singular_vectors is not None:
            # singular vector patching
            training_metadata = {
                "learning_rate": learning_rate,
                "n_epochs": n_epochs,
                "lamb": lamb,
            }
            print(f"Training singular vectors with {training_metadata}")
            singular_projection, singular_metadata = get_low_rank_projection(
                exp_name=experiment_name,
                lm=lm,
                layer_idx=layer,
                train_loader=train_dataloader,
                basis_directions=singular_vectors[layer],
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                lamb=lamb,
                verbose=verbose,
                remote=remote,
            )

            print("validating ...")
            singular_acc = validate(
                experiment_name,
                lm,
                layer,
                valid_dataloader,
                projection=singular_projection,
                verbose=verbose,
                save_outputs=save_outputs_on_val,
                projection_type="singular_vector",
                remote=remote,
            )
            print("-" * 30)
            print(
                f"Singular vector patching val: {singular_acc} | Rank: {singular_metadata['rank']}"
            )
            print("-" * 30)

            layer_performance["singular_vector"] = Accuracy(
                accuracy=singular_acc,
                rank=singular_metadata["rank"],
                metadata={
                    "training_args": training_metadata,
                    "mask": singular_metadata["mask"],
                },
            ).to_dict()

        if principal_components is not None:
            # principal component patching
            print(f"Training principal components with {training_metadata}")
            principal_projection, principal_metadata = get_low_rank_projection(
                exp_name=experiment_name,
                lm=lm,
                layer_idx=layer,
                train_loader=train_dataloader,
                basis_directions=principal_components[layer],
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                lamb=lamb,
                verbose=verbose,
                remote=remote,
            )

            print("validating ...")
            principal_acc = validate(
                experiment_name,
                lm,
                layer,
                valid_dataloader,
                projection=principal_projection,
                verbose=verbose,
                save_outputs=save_outputs_on_val,
                projection_type="principal_component",
                remote=remote,
            )
            print("-" * 30)
            print(
                f"Principal component patching val: {principal_acc} | Rank: {principal_metadata['rank']}"
            )
            print("-" * 30)

            layer_performance["principal_component"] = Accuracy(
                accuracy=principal_acc,
                rank=principal_metadata["rank"],
                metadata={
                    "training_args": training_metadata,
                    "mask": principal_metadata["mask"],
                },
            ).to_dict()

        # save results after each layer
        with open(os.path.join(save_path, f"{layer}.json"), "w") as f:
            json.dump(
                layer_performance,
                f,
                indent=4,
            )


experiment_layers = {
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "position_transmitter": list(range(30, 60, 1)),
        # "position_transmitter": [],
        "value_fetcher": list(range(50, 80, 1)),
        # "value_fetcher": [],
        "query_object": list(range(10, 40, 1)),
        "query_character": list(range(10, 40, 1)),
        "state_position": list(range(25, 45, 1)),
        "vis_2nd": list(range(0, 40, 1)),
        "vis_ques": list(range(0, 60, 1)),
        "vis_2nd_and_ques": list(range(0, 60, 1)),
        "vis_2nd_to_1st_and_ques": list(range(0, 60, 1)),
        "pointer": list(range(10, 40, 1)),
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "value_fetcher": list(range(60, 126, 2)),
        "position_transmitter": list(range(40, 80, 1)),
        "query_object": list(range(10, 50, 2)),
        "query_character": list(range(10, 50, 2)),
        "state_position": list(range(20, 70, 2)),
        "vis_2nd": list(range(0, 50, 2)),
        "vis_ques": list(range(40, 80, 2)),
        "vis_2nd_and_ques": list(range(0, 80, 2)),
        # "pointer":
    },
    "meta-llama/Meta-Llama-3.1-405B": {
        "value_fetcher": list(range(60, 126, 2)),
        "position_transmitter": list(range(40, 80, 1)),
        "query_object": list(range(10, 50, 2)),
        "query_character": list(range(10, 50, 2)),
        "state_position": list(range(20, 70, 2)),
        "vis_2nd": list(range(0, 50, 2)),
        "vis_ques": list(range(40, 80, 2)),
        "vis_2nd_and_ques": list(range(0, 80, 2)),
        # "pointer":
    },
}


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,5,6,7"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "position_transmitter",
            "value_fetcher",
            "query_object",
            "query_character",
            "state_position",
            "vis_2nd",
            "vis_ques",
            "vis_2nd_and_ques",
            "vis_2nd_to_1st_and_ques",
            "pointer",
        ],
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        choices=[
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "meta-llama/Meta-Llama-3.1-405B",
        ],
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        # default=list(range(50, 80, 1)), # value_fetcher
        # default=list(range(30, 60, 1)),  # position_transmitter
        # default=[14, 20, 32],
        # default=[14, 34, 40],
    )
    parser.add_argument("--train_size", type=int, default=80)
    parser.add_argument("--validation_size", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--lamb", type=float)
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/",
    )
    parser.add_argument("--save_outputs", type=bool, default=True)
    parser.add_argument("--remote", action="store_true", default=False)

    args = parser.parse_args()

    if args.lamb is None:
        if args.experiment == "vis_2nd":
            args.lamb = 0.03
        else:
            args.lamb = 0.1

    print(args)

    # is_remote = "405B" in args.model_key or args.remote
    is_remote = args.remote

    if is_remote:
        NDIF_KEY = env_utils.load_env_var("NDIF_KEY")
        HF_KEY = env_utils.load_env_var("HF_WRITE")

        print(f"NDIF_KEY: {NDIF_KEY}")
        print(f"HF_KEY: {HF_KEY}")

    if is_remote:
        # CONFIG.set_default_api_key(env_utils.load_env_var("NDIF_KEY"))
        os.environ["HF_TOKEN"] = env_utils.load_env_var("HF_WRITE")
        print("Loading model remotely")
        lm = LanguageModel(
            args.model_key,
        )
        print(lm.device)

    else:
        lm = LanguageModel(
            args.model_key,
            cache_dir="/disk/u/arnab/.cache/huggingface/hub/",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            dispatch=True,
            # quantization_config=BitsAndBytesConfig(      # larger models might need to be quantized to fit in memory
            #     # load_in_4bit=True
            #     load_in_8bit=True
            # ),
        )

    if args.layers is None:
        n_layer = lm.config.num_hidden_layers
        args.layers = sorted(
            list(
                set(
                    list(range(0, n_layer, 10))
                    + experiment_layers[args.model_key].get(args.experiment, [])
                    + [n_layer - 1]
                )
            )
        )

    set_seed(123456)
    run_experiment(
        experiment_name=args.experiment,
        lm=lm,
        layers=args.layers,
        train_size=args.train_size,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        verbose=args.verbose,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        lamb=args.lamb,
        save_path=args.save_path,
        save_outputs_on_val=args.save_outputs and not is_remote,
        remote=is_remote,
    )
