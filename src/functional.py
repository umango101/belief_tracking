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

from src.models import (
    get_module_nnsight,
    prepare_input,
    unwrap_tokenizer,
    reset_forwards,
)
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
    with lm.trace(get_dummy_input(lm), scan=False, validate=False) as trace:
        lnf = get_module_nnsight(lm, "model.norm")
        lnf.input = h.view(1, 1, h.squeeze().shape[0])
        logits = lm.output.logits.save()

    logits = logits.squeeze()
    candidates = interpret_logits(tokenizer=lm, logits=logits, k=k)
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


def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[Tokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')

    # logger.debug(f"Found substring in string {string.count(substring)} times")

    if occurrence < 0:
        # If occurrence is negative, count from the right.
        char_start = string.rindex(substring)
        for _ in range(-1 - occurrence):
            try:
                char_start = string.rindex(substring, 0, char_start)
            except ValueError as error:
                raise ValueError(
                    f"could not find {-occurrence} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    else:
        char_start = string.index(substring)
        for _ in range(occurrence):
            try:
                char_start = string.index(substring, char_start + 1)
            except ValueError as error:
                raise ValueError(
                    f"could not find {occurrence + 1} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    char_end = char_start + len(substring)

    # logger.debug(
    #     f"char range: [{char_start}, {char_end}] => `{string[char_start:char_end]}`"
    # )

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = prepare_input(
            string, return_offsets_mapping=True, tokenizer=tokenizer, **kwargs
        )
        offset_mapping = tokens.offset_mapping

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        # logger.debug(f"{index=} | token range: [{token_char_start}, {token_char_end}]")
        if token_char_start == token_char_end:
            # Skip special tokens # ! Is this the proper way of doing this?
            continue
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()
