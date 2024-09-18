import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import transformers
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel

from src.dataset import DatasetV3, SampleV3
from src.functional import (
    find_token_range,
    free_gpu_cache,
    get_hs,
    logit_lens,
    prepare_input,
)
from src.models import load_LM
from src.utils import env_utils
from src.utils.typing import PredictedToken


@dataclass(frozen=True)
class TokenStates(DataClassJsonMixin):
    value: str
    prompt: str
    answer: str
    token_position: int
    predicted_answer: PredictedToken
    states: dict[str, torch.Tensor | np.ndarray]


def collect_token_latent_in_question(
    lm: LanguageModel,
    prompt: str,
    answer: str,
    token_of_interest: str,
    layers: list = list(range(7, 28)),
    layer_name_format: str = "model.layers.{}",
) -> TokenStates:
    inputs = prepare_input(prompts=prompt, tokenizer=lm, return_offsets_mapping=True)
    token_range = find_token_range(
        string=prompt,
        substring=token_of_interest,
        occurrence=-1,
        offset_mapping=inputs["offset_mapping"][0],
    )
    token_last_idx = token_range[1] - 1
    inputs.pop("offset_mapping")
    print(
        f'{token_last_idx} {lm.tokenizer.decode(inputs["input_ids"][0][token_last_idx])}'
    )

    last_loc = (
        layer_name_format.format(lm.config.num_hidden_layers - 1),
        inputs.input_ids.shape[1] - 1,
    )
    locations = [(layer_name_format.format(i), token_last_idx) for i in layers] + [
        last_loc
    ]

    hs = get_hs(lm=lm, input=inputs, locations=locations, return_dict=True)
    predicted_ans = logit_lens(lm=lm, h=hs[last_loc], k=2)[0]
    print(f"{answer=} | {predicted_ans=}")

    hs_return = {}
    for loc in hs:
        module_name, token_idx = loc
        hs_return[f"{module_name}_<>_{token_idx}"] = (
            hs[loc].cpu().numpy().astype(np.float32).tolist()
        )
    return TokenStates(
        value=token_of_interest,
        prompt=prompt,
        answer=answer,
        token_position=token_last_idx,
        predicted_answer=predicted_ans,
        states=hs_return,
    )


@dataclass(frozen=True)
class SampleV3Variable(DataClassJsonMixin):
    field: str
    value: str
    unaligned_value: str
    story_informed_states: TokenStates
    story_unaware_states: TokenStates


def collect_variable_contrast_information(
    lm: LanguageModel,
    dataset: DatasetV3,
    sample_idx: int,
    field: Literal[
        "protagonist",
        "perpetrator",
        "objects_0",
        "objects_1",
        "containers_0",
        "containers_1",
    ],
) -> SampleV3Variable:

    sample = dataset.samples[sample_idx]
    if field in ["protagonist", "perpetrator"]:
        value = getattr(sample, field)
    else:
        field, f_idx = field.split("_")
        value = getattr(sample, field)[int(f_idx)]

    print(f"{field=} {value=}")

    kwargs = {}
    if field in ["protagonist", "perpetrator"]:
        kwargs["set_actor"] = field
    else:
        if field == "objects":
            kwargs["set_obj"] = int(f_idx)
        elif field == "containers":
            kwargs["set_container"] = int(f_idx)

    prompt, answer = dataset.__getitem__(
        sample_idx,
        **kwargs,
    )

    context_informed_states = collect_token_latent_in_question(
        lm=lm,
        prompt=prompt,
        answer=answer,
        token_of_interest=value,
    )

    if field in ["protagonist", "perpetrator"]:
        file_name = "actor.json"
        exclude = [sample.protagonist, sample.perpetrator]
    elif field == "objects":
        file_name = "object.json"
        exclude = sample.objects
    elif field == "containers":
        file_name = "container.json"
        exclude = sample.containers

    root = os.path.join(env_utils.DEFAULT_DATA_DIR, "synthetic_entities")
    names = json.load(open(os.path.join(root, file_name), "r"))
    names = list(set(names) - set(exclude))
    unaligned_value = random.choice(names)

    Q_idx = prompt.index("Question:")
    unaligned_prompt = prompt[:Q_idx].replace(value, unaligned_value) + prompt[Q_idx:]
    context_unaware_states = collect_token_latent_in_question(
        lm=lm,
        prompt=unaligned_prompt,
        answer="no",
        token_of_interest=value,  # its always the `value` in the Question that we wanna track
    )

    return SampleV3Variable(
        field=field,
        value=value,
        unaligned_value=unaligned_value,
        story_informed_states=context_informed_states,
        story_unaware_states=context_unaware_states,
    )


@dataclass(frozen=True)
class CachedBindingIDState(DataClassJsonMixin):
    sample: SampleV3
    protagonist: SampleV3Variable
    perpetrator: SampleV3Variable
    objects: list[SampleV3Variable]
    containers: list[SampleV3Variable]

    def __post_init__(self):
        assert len(self.objects) == 2 and len(self.containers) == 2


# @dataclass(frozen=False)
# class ExperimentResults(DataClassJsonMixin):
#     model_name: str
#     cached_states: list[CachedBindingIDState]


def cache_states(
    model_key: str,
    save_dir: str,
    layers: list = list(range(7, 28)),
    layer_name_format: str = "model.layers.{}",
    idx_range: Optional[tuple[int, int]] = None,
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

    with open(os.path.join(env_utils.DEFAULT_DATA_DIR, "dataset_v3.json"), "r") as f:
        dataset_dict = json.load(f)

    dataset = DatasetV3.from_dict(dataset_dict)
    print(f"dataset loaded, {len(dataset)} samples")

    # results = ExperimentResults(model_name=model_key, cached_states=[])
    idx_range = idx_range if idx_range is not None else (0, len(dataset))
    idx_range = (idx_range[0], min(idx_range[1], len(dataset)))
    limit = idx_range[1] - idx_range[0]
    for idx in range(*idx_range):
        print(f"\nprocessing {idx+1} ... {idx - idx_range[0]}/{limit}")

        variable_states = {
            field: None
            for field in [
                "protagonist",
                "perpetrator",
                "objects_0",
                "objects_1",
                "containers_0",
                "containers_1",
            ]
        }
        for field in variable_states:
            variable_states[field] = collect_variable_contrast_information(
                lm=lm,
                dataset=dataset,
                sample_idx=idx,
                field=field,
            )

        cur_states = CachedBindingIDState(
            sample=dataset.samples[idx],
            protagonist=variable_states["protagonist"],
            perpetrator=variable_states["perpetrator"],
            objects=[variable_states["objects_0"], variable_states["objects_1"]],
            containers=[
                variable_states["containers_0"],
                variable_states["containers_1"],
            ],
        )

        # results.cached_states.append(
        #     cur_results
        # )
        # if idx % 100 == 0:
        #     with open(os.path.join(save_dir, "results.json"), "w") as f:
        #         json.dump(results.to_dict(), f)

        with open(os.path.join(save_dir, f"doc_idx_{idx}.json"), "w") as f:
            json.dump(cur_states.to_dict(), f)

        print("-" * 50)
        free_gpu_cache()

    # with open(os.path.join(save_dir, "results.json"), "w") as f:
    #     json.dump(results.to_dict(), f)


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
        "--idx-from",
        type=int,
        default=111111,
    )
    parser.add_argument(
        "--idx-to",
        type=int,
        default=111111,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="binding_id_states_split",
    )

    args = parser.parse_args()
    print(args)

    idx_range = (args.idx_from, args.idx_to) if args.idx_from != 111111 else None
    if args.idx_from == 111111 and args.idx_to != 111111:
        idx_range = (0, args.idx_to)

    cache_states(
        model_key=args.model,
        save_dir=args.save_dir,
        idx_range=idx_range,
    )
