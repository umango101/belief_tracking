import ctypes
import gc
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch
from nnsight import LanguageModel
from openai import OpenAI
from tqdm import tqdm

from src.models import get_module_nnsight, prepare_input, unwrap_tokenizer
from src.utils.typing import PredictedToken, Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


@torch.inference_mode()
def interpret_logits(
    tokenizer: LanguageModel | Tokenizer,
    logits: torch.Tensor,
    k: int = 5,
) -> list[PredictedToken]:
    tokenizer = unwrap_tokenizer(tokenizer)
    logits = logits.squeeze()
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    top_k_indices = logits.topk(dim=-1, k=k).indices.squeeze().tolist()

    return [
        PredictedToken(
            token=tokenizer.decode(t),
            prob=probs[t].item(),
            logit=logits[t].item(),
            token_id=t,
        )
        for t in top_k_indices
    ]


@torch.inference_mode()
def logit_lens(
    lm: LanguageModel,
    h: torch.Tensor,
    interested_tokens: list[int] = [],
    k: int = 5,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    with lm.trace(get_dummy_input(lm), scan=False, validate=False) as tr:
        lnf = lm.model.norm
        lnf.input = (h.view(1, 1, h.squeeze().shape[0]), lnf.input[1])
        logits = lm.output.logits.save()
    logits = logits.squeeze()
    candidates = interpret_logits(lm, logits, k=k)
    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        interested_logits = {
            t: (
                rank_tokens.index(t) + 1,
                PredictedToken(
                    token=lm.tokenizer.decode(t),
                    prob=probs[t].item(),
                    logit=logits[t].item(),
                    token_id=t,
                ),
            )
            for t in interested_tokens
        }
        return candidates, interested_logits
    free_gpu_cache()
    return candidates


def untuple(object: Any):
    if isinstance(object, tuple) or (
        "LanguageModelProxy" in str(type(object)) and len(object) > 1
    ):
        return object[0]
    return object


def free_gpu_cache():
    before = torch.cuda.memory_allocated()
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    freed = before - after


@dataclass(frozen=False)
class PatchSpec:
    location: tuple[str, int]
    patch: torch.Tensor
    clean: Optional[torch.Tensor] = None


@torch.inference_mode()
def get_hs(
    lm: LanguageModel,
    input: str | TokenizerOutput,
    locations: tuple[str, int] | list[tuple[str, int]],
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
    return_dict: bool = False,
) -> dict[tuple[str, int], torch.Tensor]:

    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=lm)

    if isinstance(locations, tuple):
        locations = [locations]
    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]

    layer_names = [layer_name for layer_name, _ in locations]
    layer_names = list(set(layer_names))
    layer_states = {layer_name: torch.empty(0) for layer_name in layer_names}
    with lm.trace(input, scan=False, validate=False) as tracer:
        if patches is not None:
            for cur_patch in patches:
                module_name, index = cur_patch.location
                module = get_module_nnsight(lm, module_name)
                current_state = (
                    module.output
                    if ("mlp" in module_name or module_name == "model.embed_tokens")
                    else module.output[0]
                )
                current_state[0, index, :] = cur_patch.patch

        for layer_name in layer_names:
            module = get_module_nnsight(lm, layer_name)
            layer_states[layer_name] = module.output.save()

    hs = {}

    for layer_name, index in locations:
        hs[(layer_name, index)] = untuple(layer_states[layer_name])[
            :, index, :
        ].squeeze()

    free_gpu_cache()
    # print(f"==========> {len(hs)=}")
    if len(hs) == 1 and not return_dict:
        return list(hs.values())[0]
    return hs


@torch.inference_mode()
def predict_next_token(
    lm: LanguageModel,
    input: str | TokenizerOutput,
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
    k: int = 5,
    interested_tokens: list[int] = [],
):
    free_gpu_cache()
    if isinstance(input, str):
        input = prepare_input(prompts=input, tokenizer=lm)

    location = (f"model.layers.{lm.config.num_hidden_layers-1}", -1)
    hs = get_hs(
        lm=lm,
        input=input,
        locations=location,
        patches=patches,
    )
    free_gpu_cache()
    return logit_lens(lm=lm, h=hs, k=k, interested_tokens=interested_tokens)


def get_dummy_input(
    tokenizer: LanguageModel | Tokenizer,
):
    dummy_prompt = "The quick brown fox"
    return prepare_input(prompts=dummy_prompt, tokenizer=tokenizer)
