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
from transformers import BitsAndBytesConfig

from scripts.run_exp_utils import (
    Accuracy,
    exp_to_vec_type,
    free_gpu_cache,
    load_basis_directions,
    prepare_dataset,
    set_seed,
)
from src.utils import env_utils

charac_indices = [131, 133, 146, 147, 158, 159]
reversed_charac_indices = [133, 131, 158, 159, 146, 147]

object_indices = [150, 151, 162, 163]
reversed_object_indices = [162, 163, 150, 151]

state_indices = [155, 156, 167, 168]
query_character_indices = [-8, -7]
query_object_indices = [-5, -4]

retain_full_indices = {
    "object_position": state_indices,
    "character_position": object_indices + state_indices,
    "source_1": state_indices,
    "source_2": [],
}
retain_upto_indices = {
    "object_position": query_character_indices,
    "character_position": query_object_indices,
    "source_1": [],
    "source_2": [],
}
patch_indices = {
    "object_position": reversed_object_indices,
    "character_position": reversed_charac_indices,
    "source_1": reversed_charac_indices + reversed_object_indices,
    "source_2": reversed_charac_indices + reversed_object_indices,
}


def is_mixed_projections(experiment_name):
    return experiment_name in ["source_1", "source_2"]


@torch.inference_mode()
def validate(
    exp_name: Literal["object_position", "character_position", "source_1", "source_2"],
    lm: LanguageModel,
    layer_idx: int,
    validation_loader: DataLoader,
    projections: (
        dict[str, torch.Tensor] | dict[str, dict[int, torch.Tensor]] | None
    ) = None,
    verbose: bool = False,
    restore_state: bool = True,
    save_outputs: bool = True,
    projection_type: Literal[
        "full_rank", "singular_vector", "principal_component"
    ] = "full_rank",
    remote: bool = False,
) -> float:
    # lm.tokenizer.padding_side = "left"
    # lm.model.eval()
    save_outputs = save_outputs and not remote

    cuda_last_index = torch.cuda.device_count() - 1
    device_p = f"cuda:{cuda_last_index}"

    if save_outputs:
        exp_dir = exp_name
        if restore_state is False:
            exp_dir += "_wo_state"
        save_path = os.path.join(
            "results",
            lm.config._name_or_path.split("/")[-1],
            "lm_pred_on_val_set",
            exp_dir,
            f"layer_{layer_idx}",
            projection_type,
        )
        os.makedirs(save_path, exist_ok=True)
        valid_data = validation_loader.dataset

    interesting_positions = (
        charac_indices
        + object_indices
        + state_indices
        + query_character_indices
        + query_object_indices
    )

    patch_to_cache_map = {
        k: v
        for k, v in zip(
            charac_indices + object_indices,
            reversed_charac_indices + reversed_object_indices,
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
        alt_acts, org_acts = defaultdict(dict), defaultdict(dict)

        def nnsight_request(return_logits: bool = False):
            with lm.trace(remote=remote) as tracer:
                with tracer.invoke(alt_prompts):
                    for layer in range(0, layer_idx + 1):
                        for t in interesting_positions:
                            alt_acts[layer][t] = (
                                lm.model.layers[layer].output[0][:, t].clone()
                            )

                with tracer.invoke(org_prompts):
                    for layer in range(0, lm.config.num_hidden_layers):
                        for t in interesting_positions:
                            org_acts[layer][t] = (
                                lm.model.layers[layer].output[0][:, t].clone()
                            )

                with tracer.invoke(org_prompts):
                    for layer in range(0, layer_idx + 1):
                        if projections is not None:
                            if is_mixed_projections(exp_name):
                                projection = {
                                    k: v[layer].to(device=device_p)
                                    for k, v in projections.items()
                                }
                            else:
                                projection = projections[layer].to(device=device_p)
                        if exp_name == "object_position":
                            for t in reversed_charac_indices:
                                lm.model.layers[layer].output[0][:, t] = alt_acts[
                                    layer
                                ][patch_to_cache_map[t]]

                        for t in patch_indices[exp_name]:
                            curr_output = lm.model.layers[layer].output[0][:, t].clone()
                            if projections is not None:
                                if is_mixed_projections(exp_name):
                                    if t in charac_indices:
                                        proj = projection["character_ordering_id"]
                                    elif t in object_indices:
                                        proj = projection["object_ordering_id"]
                                    else:
                                        raise ValueError("Invalid patch index")
                                else:
                                    proj = projection

                                alt_proj = torch.matmul(
                                    alt_acts[layer][patch_to_cache_map[t]], proj
                                )
                                org_proj = torch.matmul(curr_output, proj)
                                patch = curr_output - org_proj + alt_proj
                            else:
                                patch = alt_acts[layer][patch_to_cache_map[t]]

                            lm.model.layers[layer].output[0][:, t] = patch

                        for t in retain_upto_indices[exp_name]:
                            lm.model.layers[layer].output[0][:, t] = org_acts[layer][t]

                    for layer in range(lm.config.num_hidden_layers):
                        for t in retain_full_indices[exp_name]:
                            if restore_state == False:
                                if t in state_indices:
                                    continue
                            lm.model.layers[layer].output[0][:, t] = org_acts[layer][t]

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
        print(f"{pred=}")

        for i in range(batch_size):
            total += 1
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

        del (alt_acts, alt_prompts, org_prompts, targets, target_tokens, logits, pred)

        # if projections is not None:
        #     try:
        #         del projection
        #     except:
        #         pass
        free_gpu_cache()

    return correct / total


def get_low_rank_projection(
    exp_name: Literal["object_position", "character_position", "source_1", "source_2"],
    lm: LanguageModel,
    layer_idx: int,
    train_loader: DataLoader,
    all_basis_directions: dict[int, torch.Tensor] | dict[str, dict[int, torch.Tensor]],
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.01,
    verbose: bool = False,
    restore_state: bool = True,
    remote: bool = False,
) -> tuple[torch.Tensor, dict]:
    # lm.tokenizer.padding_side = "left"
    # lm.model.eval()

    if remote is True:
        raise NotImplementedError("Training not tested for remote yet")

    cuda_last_index = torch.cuda.device_count() - 1
    device_p = f"cuda:{cuda_last_index}"

    interesting_positions = (
        charac_indices
        + object_indices
        + state_indices
        + query_character_indices
        + query_object_indices
    )

    patch_to_cache_map = {
        k: v
        for k, v in zip(
            charac_indices + object_indices,
            reversed_charac_indices + reversed_object_indices,
        )
    }
    # print(patch_to_cache_map)

    if is_mixed_projections(exp_name) == False:
        basis_indices = list(range(all_basis_directions[0].size(0)))
        mask = torch.ones(
            (layer_idx + 1, len(basis_indices)),
            requires_grad=True,
            device=device_p,
            dtype=torch.bfloat16,
        )
        optimizer = torch.optim.Adam([mask], lr=learning_rate)
    else:
        mask = {}
        for key in all_basis_directions:
            basis_indices = list(range(all_basis_directions[key][0].size(0)))
            mask[key] = torch.ones(
                (layer_idx + 1, len(basis_indices)),
                requires_grad=True,
                device=device_p,
                dtype=torch.bfloat16,
            )
        optimizer = torch.optim.Adam([{"params": mask.values()}], lr=learning_rate)

    for epoch in range(n_epochs):
        epoch_loss = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            alt_prompts = batch["corrupt_prompt"]
            org_prompts = batch["clean_prompt"]
            targets = batch["target"] if "target" in batch else batch["corrupt_target"]
            target_tokens = lm.tokenizer(targets, return_tensors="pt").input_ids[:, -1]
            batch_size = target_tokens.size(0)
            alt_acts, org_acts = defaultdict(dict), defaultdict(dict)

            with lm.trace() as tracer:
                with tracer.invoke(alt_prompts):
                    for layer in range(0, layer_idx + 1):
                        for t in interesting_positions:
                            alt_acts[layer][t] = (
                                lm.model.layers[layer].output[0][:, t].clone()
                            )

                with tracer.invoke(org_prompts):
                    for layer in range(0, lm.config.num_hidden_layers):
                        for t in interesting_positions:
                            org_acts[layer][t] = (
                                lm.model.layers[layer].output[0][:, t].clone()
                            )

                with tracer.invoke(org_prompts):
                    for layer in range(0, layer_idx + 1):
                        if exp_name == "object_position":
                            for t in reversed_charac_indices:
                                lm.model.layers[layer].output[0][:, t] = alt_acts[
                                    layer
                                ][patch_to_cache_map[t]]

                        if is_mixed_projections(exp_name) == False:
                            masked_directions = all_basis_directions[layer].to(
                                device=device_p
                            ) * mask[layer].unsqueeze(-1)
                            projection = torch.matmul(
                                masked_directions.T, masked_directions
                            ).half()
                        else:
                            projection = {}
                            for key in all_basis_directions:
                                masked_directions = all_basis_directions[key][layer].to(
                                    device=device_p
                                ) * mask[key][layer].unsqueeze(-1)
                                projection[key] = torch.matmul(
                                    masked_directions.T, masked_directions
                                ).half()

                        for t in patch_indices[exp_name]:
                            if is_mixed_projections(exp_name):
                                if t in charac_indices:
                                    proj = projection["character_ordering_id"]
                                elif t in object_indices:
                                    proj = projection["object_ordering_id"]
                                else:
                                    raise ValueError("Invalid patch index")
                            else:
                                proj = projection

                            curr_output = lm.model.layers[layer].output[0][:, t].clone()
                            alt_proj = torch.matmul(
                                alt_acts[layer][patch_to_cache_map[t]], proj
                            )
                            org_proj = torch.matmul(curr_output, proj)
                            patch = curr_output - org_proj + alt_proj
                            lm.model.layers[layer].output[0][:, t] = patch

                        for t in retain_upto_indices[exp_name]:
                            lm.model.layers[layer].output[0][:, t] = org_acts[layer][t]

                    for layer in range(lm.config.num_hidden_layers):
                        for t in retain_full_indices[exp_name]:
                            if restore_state == False:
                                if t in state_indices:
                                    continue
                            lm.model.layers[layer].output[0][:, t] = org_acts[layer][t]

                    logits = lm.lm_head.output[:, -1].save()

            target_logit = logits[torch.arange(batch_size), target_tokens]
            task_loss = -torch.mean(target_logit)
            if is_mixed_projections(exp_name) == False:
                l1_loss = lamb * torch.norm(mask, p=1)
            else:
                l1_loss = 0
                for key in mask:
                    l1_loss += lamb * torch.norm(mask[key], p=1)
            loss = task_loss + l1_loss.to(task_loss.device)

            if verbose:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()} |>> l_task: {task_loss.item()}, l1: {l1_loss.item()}"
                )
                if is_mixed_projections(exp_name) == False:
                    mask_data = mask.data.clone().clamp(0, 1).round()
                    print(
                        {
                            l: int(mask_data[l].sum().item())
                            for l in range(mask_data.size(0))
                        }
                    )
                else:
                    for key in mask:
                        print(key)
                        mask_data = mask[key].data.clone().clamp(0, 1).round()
                        print(
                            {
                                l: int(mask_data[l].sum().item())
                                for l in range(mask_data.size(0))
                            }
                        )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clamp after optimizer step
            with torch.no_grad():
                if is_mixed_projections(exp_name) == False:
                    mask.data.clamp_(0, 1)
                else:
                    for key in mask:
                        mask[key].data.clamp_(0, 1)

            epoch_loss += loss.item()

            del (
                alt_acts,
                alt_prompts,
                org_prompts,
                targets,
                target_tokens,
                logits,
                projection,
            )
            free_gpu_cache()

    # build the projections after training
    if is_mixed_projections(exp_name) == False:
        projections = {}
        meta_data = {}
        for layer in range(layer_idx + 1):
            mask_data = mask[layer].data.clone().clamp(0, 1).round()
            masked_directions = all_basis_directions[layer].to(
                device=device_p
            ) * mask_data.unsqueeze(-1)
            projections[layer] = (
                torch.matmul(masked_directions.T, masked_directions).half().cpu()
            )
            # projections[layer] = projection.cpu()
            meta_data[layer] = {
                "mask": mask_data.tolist(),
                "rank": mask_data.sum().item(),
            }
    else:
        projections = {}
        meta_data = {}
        for key in mask:
            projections[key] = {}
            meta_data[key] = {}
            for layer in range(layer_idx + 1):
                mask_data = mask[key][layer].data.clone().clamp(0, 1).round()
                masked_directions = all_basis_directions[key][layer].to(
                    device=device_p
                ) * mask_data.unsqueeze(-1)
                projections[key][layer] = (
                    torch.matmul(masked_directions.T, masked_directions).half().cpu()
                )
                meta_data[key][layer] = {
                    "mask": mask_data.tolist(),
                    "rank": mask_data.sum().item(),
                }

    for param in lm.model.parameters():
        param.grad = None
    free_gpu_cache()

    return projections, meta_data


def run_experiment(
    experiment_name: Literal["object_position", "character_position"],
    lm: LanguageModel,
    layers: list[int] = list(range(34, 60, 2)),
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    verbose: bool = False,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.01,
    save_path: str = "results/",
    restore_state: bool = True,
    save_outputs_on_val: bool = True,
    remote: bool = False,
):
    print("#" * 30)
    print(f"Running experiment: {experiment_name}")
    print("#" * 30)

    exp_subdir = experiment_name
    if restore_state == False:
        exp_subdir += "_wo_state"
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

    # train_dataset = train_dataloader.dataset
    # idx = 5
    # tokens = lm.tokenizer.encode(
    #     train_dataset[idx]["corrupt_prompt"], return_tensors="pt"
    # )
    # print(lm.tokenizer.decode(tokens[0][object_indices]))

    # tokens = lm.tokenizer.encode(
    #     train_dataset[idx]["clean_prompt"], return_tensors="pt"
    # )
    # print(lm.tokenizer.decode(tokens[0][object_indices]))

    singular_vectors, principal_components = None, None
    # exclude_projections = ["source_1", "source_2"]
    exclude_projections = []

    skip_low_rank_projection = (experiment_name in exclude_projections) or (
        "405B" in lm.config._name_or_path
    )

    if skip_low_rank_projection == False:
        direction_type = exp_to_vec_type[experiment_name]
        if isinstance(direction_type, list):
            singular_vectors = {
                d: load_basis_directions("singular_vecs", d) for d in direction_type
            }
        else:
            singular_vectors = load_basis_directions("singular_vecs", direction_type)

        if isinstance(direction_type, list):
            principal_components = {
                d: load_basis_directions("principal_components", d)
                for d in direction_type
            }
        else:
            principal_components = load_basis_directions(
                "principal_components", direction_type
            )

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
            projection_type="full_rank",
            save_outputs=save_outputs_on_val,
            restore_state=restore_state,
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
                all_basis_directions=singular_vectors,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                lamb=lamb,
                verbose=verbose,
                remote=remote,
            )

            print("validating ...")
            singular_acc = validate(
                exp_name=experiment_name,
                lm=lm,
                layer_idx=layer,
                validation_loader=valid_dataloader,
                projections=singular_projection,
                verbose=verbose,
                save_outputs=save_outputs_on_val,
                restore_state=restore_state,
                projection_type="singular_vector",
                remote=remote,
            )
            print("-" * 30)
            print(f"Singular vector patching val: {singular_acc}")
            print("-" * 30)

            layer_performance["singular_vector"] = Accuracy(
                accuracy=singular_acc,
                metadata={
                    "training_args": training_metadata,
                    "metadata": singular_metadata,
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
                all_basis_directions=principal_components,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                lamb=lamb,
                verbose=verbose,
                remote=remote,
            )

            print("validating ...")
            principal_acc = validate(
                exp_name=experiment_name,
                lm=lm,
                layer_idx=layer,
                validation_loader=valid_dataloader,
                projections=principal_projection,
                verbose=verbose,
                save_outputs=save_outputs_on_val,
                restore_state=restore_state,
                projection_type="principal_component",
                remote=remote,
            )
            print("-" * 30)
            print(f"Principal component patching val: {principal_acc}")
            print("-" * 30)

            layer_performance["principal_component"] = Accuracy(
                accuracy=principal_acc,
                metadata={
                    "training_args": training_metadata,
                    "metadata": principal_metadata,
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
        # "object_position": list(range(10, 40, 1)),
        "object_position": [],
        # "character_position": list(range(10, 40, 1)),
        "character_position": [],
        "source_1": list(range(10, 40, 1)),
        "source_2": list(range(10, 40, 1)),
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "source_1": list(range(20, 80, 2)),
        "source_2": list(range(20, 80, 2)),
        "object_position": list(range(20, 80, 2)),
        "character_position": list(range(20, 80, 2)),
    },
}

if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["object_position", "character_position", "source_1", "source_2"],
    )
    parser.add_argument(
        "--model_key", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        # default=list(range(0, 80, 10)),
        # default=[14, 22, 32],
        # default=[30, 34, 38],
    )
    parser.add_argument("--train_size", type=int, default=80)
    parser.add_argument("--validation_size", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--lamb", type=float, default=0.01)
    parser.add_argument(
        "--save_path",
        type=str,
        default="results",
    )
    parser.add_argument("--save_outputs", type=bool, default=True)
    parser.add_argument(
        "--restore_state", action="store_true", default=True, help="Restore state"
    )
    parser.add_argument(
        "--no-restore_state",
        dest="restore_state",
        action="store_false",
        help="Don't restore state",
    )
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    print(args)

    is_remote = "405B" in args.model_key
    print(f"<><><><> {is_remote=}")

    if is_remote:
        NDIF_KEY = env_utils.load_env_var("NDIF_KEY")
        HF_KEY = env_utils.load_env_var("HF_WRITE")

        print(f"NDIF_KEY: {NDIF_KEY}")
        print(f"HF_KEY: {HF_KEY}")
        local_loading_kwargs = dict()

    else:
        local_loading_kwargs = dict(
            cache_dir="/disk/u/arnab/.cache/huggingface/hub/",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            dispatch=True,
        )

    if "405B" in args.model_key:
        local_loading_kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True),
        )

    # print(f"loading model with {is_remote = } | {local_loading_kwargs=}")

    if is_remote:
        # CONFIG.set_default_api_key(env_utils.load_env_var("NDIF_KEY"))
        os.environ["HF_TOKEN"] = env_utils.load_env_var("HF_WRITE")
        print("Loading model remotely")
        lm = LanguageModel(
            args.model_key,
        )
        # print(lm.device)

    else:
        lm = LanguageModel(
            args.model_key,
            cache_dir="/disk/u/arnab/.cache/huggingface/hub/",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            dispatch=True,
            # quantization_config=BitsAndBytesConfig(
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
        restore_state=args.restore_state,
        save_outputs_on_val=args.save_outputs,
        remote=is_remote,
    )
