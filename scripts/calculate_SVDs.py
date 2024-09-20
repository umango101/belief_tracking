import argparse
import json
import os
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
import spacy
import torch
from nnsight import LanguageModel
from openai import OpenAI
from tqdm.auto import tqdm

from scripts.collect_binding_id_states import CachedBindingIDState, SampleV3
from src.functional import free_gpu_cache
from src.models import load_LM
from src.utils import env_utils


def grab_layer_state(states, layer_name):
    keys = states.keys()
    for key in keys:
        if key.startswith(layer_name):
            return torch.Tensor(states[key])


def build_representation_matrix_from_cache(
    cache_path: str,
    layer_name: str,
    fields: list[Literal["protagonist", "perpetrator", "objects", "containers"]],
    limit: int = 10000,
) -> np.ndarray | torch.Tensor:

    subtrations = []
    for file_name in tqdm(os.listdir(cache_path)[:limit]):
        if not file_name.endswith(".json"):
            continue
        with open(os.path.join(cache_path, file_name), "r") as f:
            loaded_dict = json.load(f)
            sample_variables = []
            for field_name in fields:
                field_value = loaded_dict[field_name]
                if isinstance(field_value, list):
                    sample_variables.extend(field_value)
                else:
                    sample_variables.append(field_value)

            for var in sample_variables:
                aligned_state = grab_layer_state(
                    var["story_informed_states"]["states"], layer_name
                )
                unaligned_state = grab_layer_state(
                    var["story_unaware_states"]["states"], layer_name
                )
                subtrations.append(aligned_state - unaligned_state)

        if len(subtrations) >= limit:
            break

    return torch.stack(subtrations)


def calculate_and_save_SVD(
    cache_dir: str,
    model_name: str,
    save_dir: str,
    layer_name: str,
    fields: list[Literal["protagonist", "perpetrator", "objects", "containers"]],
    limit: int = 10000,
):
    cache_dir = os.path.join(env_utils.DEFAULT_RESULTS_DIR, cache_dir, model_name)
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, save_dir, model_name, "_".join(fields)
    )
    os.makedirs(save_dir, exist_ok=True)

    representation_matrix = build_representation_matrix_from_cache(
        cache_path=cache_dir,
        layer_name=layer_name,
        fields=fields,
        limit=limit,
    )

    print("loaded delta representations", representation_matrix.shape)
    print("saving representations ...")

    representations_dir = os.path.join(save_dir, "representations")
    os.makedirs(representations_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(representations_dir, f"{layer_name}.npz"),
        representation_matrix=representation_matrix.cpu().numpy(),
    )

    _, singular_values, Vh = torch.linalg.svd(
        representation_matrix.cuda(), full_matrices=False
    )

    print(f"{singular_values.shape=} | {Vh.shape=}")

    print("saving SVDs ...")

    projection_dir = os.path.join(save_dir, "projections")
    os.makedirs(projection_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(projection_dir, f"{layer_name}.npz"),
        Vh=Vh.cpu().numpy(),
        singlar_values=singular_values.cpu().numpy(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "Meta-Llama-3-8B-Instruct",
            "Meta-Llama-3-70B-Instruct",
        ],
        default="Meta-Llama-3-70B-Instruct",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="SVDs",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="BID_cache_V3",
    )

    parser.add_argument(
        "--layer-idx",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=["protagonist", "perpetrator"],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
    )

    args = parser.parse_args()
    print(args)

    LAYER_NAME_FORMAT = "model.layers.{}"

    calculate_and_save_SVD(
        cache_dir=args.cache_dir,
        model_name=args.model,
        save_dir=args.save_dir,
        layer_name=LAYER_NAME_FORMAT.format(int(args.layer_idx)),
        fields=args.fields,
        limit=args.limit,
    )
