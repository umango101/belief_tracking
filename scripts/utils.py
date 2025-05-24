import json
import os
from collections import defaultdict
from typing import Literal

import torch
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.run_exp_utils import (
    exp_to_intervention_positions,
    free_gpu_cache,
)

query_object_indices = [-5, -4]
query_character_indices = [-8, -7]


@torch.inference_mode()
def validate(
    exp_name: Literal[
        "answer_lookback-pointer",
        "answer_lookback-payload",
        "binding_lookback-pointer_object",
        "binding_lookback-pointer_character",
        "binding_lookback-address_and_payload",
        "binding_lookback-object_oi",
        "binding_lookback-character_oi",
        "visibility_lookback-source",
        "visibility_lookback-payload",
        "visibility_lookback-address_and_pointer",
        "binding_lookback-pointer_character_and_object",
    ],
    lm: LanguageModel,
    layer_idx: int,
    validation_loader: DataLoader,
    projection: torch.Tensor | dict[str, torch.Tensor] | None = None,
    verbose: bool = False,
    save_outputs: bool = False,
    projection_type: Literal[
        "full_rank", "singular_vector", "principal_component"
    ] = "full_rank",
    remote: bool = False,
) -> float:
    save_outputs = save_outputs and not remote

    if save_outputs:
        save_path = os.path.join(
            "results",
            "lm_pred_on_val_set",
            lm.config._name_or_path.split("/")[-1],
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
                with tracer.invoke(alt_prompts):
                    for t in intervention_positions["cache"]:
                        alt_acts[t] = lm.model.layers[layer_idx].output[0][:, t].clone()

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
        "answer_lookback-pointer",
        "answer_lookback-payload",
        "binding_lookback-pointer_object",
        "binding_lookback-pointer_character",
        "binding_lookback-address_and_payload",
        "visibility_lookback-source",
        "visibility_lookback-payload",
        "visibility_lookback-address_and_pointer",
        "binding_lookback-pointer_character_and_object",
        "character_oi",
        "object_oi",
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
    if remote:
        raise NotImplementedError("Training not tested for remote yet")

    intervention_positions = exp_to_intervention_positions[exp_name]
    patch_to_cache_map = {
        k: v
        for k, v in zip(
            intervention_positions["patch"], intervention_positions["cache"]
        )
    }

    if isinstance(basis_directions, dict) == False:
        print("Basis directions is a tensor")
        basis_indices = list(range(basis_directions.size(0)))
        mask = torch.ones(
            len(basis_indices), requires_grad=True, device="cuda", dtype=torch.bfloat16
        )
        basis_directions = basis_directions.to("cuda")
        optimizer = torch.optim.Adam([mask], lr=learning_rate)
    else:
        print("Basis directions is a dict")
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
