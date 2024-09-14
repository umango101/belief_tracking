import argparse
import logging
import os

import numpy as np
import torch
import transformers
from src.dataset import DatasetV2, SampleV2
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel
from src.utils import env_utils
from src.models import load_LM
import json
from typing import Optional
import random

from src.functional import get_hs, find_token_range, prepare_input, logit_lens


def collect_actor_latent_in_question(
    lm: LanguageModel,
    question: str,
    actor: str,
    layers: list = list(range(7, 28)),
    layer_name_format: str = "model.layers.{}",
) -> dict[int, torch.Tensor | np.ndarray]:
    inputs = prepare_input(prompts=question, tokenizer=lm, return_offsets_mapping=True)
    action_last_range = find_token_range(
        string=question,
        substring=actor,
        occurrence=-1,
        offset_mapping=inputs["offset_mapping"][0],
    )
    actor_last_idx = action_last_range[1] - 1
    inputs.pop("offset_mapping")
    print(
        actor_last_idx,
        " >> ",
        lm.tokenizer.decode(inputs["input_ids"][0][actor_last_idx]),
    )

    last_loc = (
        layer_name_format.format(lm.config.num_hidden_layers - 1),
        inputs.input_ids.shape[1] - 1,
    )
    locations = [(layer_name_format.format(i), actor_last_idx) for i in layers] + [
        last_loc
    ]

    hs = get_hs(lm=lm, input=inputs, locations=locations, return_dict=True)
    predicted_ans = logit_lens(lm=lm, h=hs[last_loc], k=2)[0]
    print(f"{predicted_ans=}")

    hs_return = {}
    for loc in hs:
        module_name, token_idx = loc
        hs_return[f"{module_name}_<>_{token_idx}"] = (
            hs[loc].cpu().numpy().astype(np.float32).tolist()
        )
    return hs_return


@dataclass(frozen=True)
class CachedBindingIDState(DataClassJsonMixin):
    sample: SampleV2
    question: str
    answer: str
    unaligned_actor: str
    context_informed_actor: dict[str, torch.Tensor]
    context_unaware_actor: dict[str, torch.Tensor]


@dataclass(frozen=False)
class ExperimentResults(DataClassJsonMixin):
    model_name: str
    cached_states: list[CachedBindingIDState]


def collect_binding_id_states(
    model_key: str,
    save_dir: str,
    layers: list = list(range(7, 28)),
    layer_name_format: str = "model.layers.{}",
    limit: Optional[int] = None,
):

    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, save_dir, model_key.split("/")[-1]
    )
    os.makedirs(save_dir, exist_ok=True)

    lm = load_LM(
        model_key=model_key,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    with open(os.path.join(env_utils.DEFAULT_DATA_DIR, "dataset_v2.json"), "r") as f:
        dataset_dict = json.load(f)

    dataset = DatasetV2.from_dict(dataset_dict)
    print(f"dataset loaded, {len(dataset)} samples")

    with open(os.path.join(env_utils.DEFAULT_DATA_DIR, "english_names.json"), "r") as f:
        names = json.load(f)

    results = ExperimentResults(model_name=model_key, cached_states=[])
    limit = len(dataset) if limit is None else limit
    for idx in range(limit):
        print(f"\nprocessing {idx+1}/{limit}")

        ans = random.choice(["yes", "no"])
        sample = dataset.samples[idx]
        question, answer = dataset.__getitem__(idx, set_ans=ans)
        actor = sample.actor
        print(f"{question=}")
        print(f"{answer=}")
        print(f"{actor=}")

        context_informed_actor = collect_actor_latent_in_question(
            lm=lm,
            question=question,
            actor=actor,
            layers=layers,
            layer_name_format=layer_name_format,
        )

        unaligned_actor = random.choice(list(set(names) - {actor}))
        print(f"{unaligned_actor=}")
        unaligned_question, answer = dataset.__getitem__(
            idx, set_ans=ans, unaligned_actor=unaligned_actor
        )
        print(f"{unaligned_question=}")
        context_unaware_actor = collect_actor_latent_in_question(
            lm=lm,
            question=unaligned_question,
            actor=actor,
            layers=layers,
            layer_name_format=layer_name_format,
        )

        results.cached_states.append(
            CachedBindingIDState(
                sample=sample,
                question=question,
                answer=answer,
                unaligned_actor=unaligned_actor,
                context_informed_actor=context_informed_actor,
                context_unaware_actor=context_unaware_actor,
            )
        )

        if idx % 100 == 0:
            with open(os.path.join(save_dir, "results.json"), "w") as f:
                json.dump(results.to_dict(), f)

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results.to_dict(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ],
        default="meta-llama/Meta-Llama-3-70B-Instruct",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="binding_id_states",
    )

    args = parser.parse_args()
    print(args)

    collect_binding_id_states(
        model_key=args.model,
        save_dir=args.save_dir,
        limit=args.limit if args.limit > 0 else None,
    )
