import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, overload

import baukit
import torch
import transformers
from nnsight import LanguageModel

from src.utils.env_utils import DEFAULT_MODELS_DIR
from src.utils.typing import Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)

CACHEABLE_FUNCS = [
    "forward",
    # "ssm", "selective_scan" , # specific to Mamba models
]

_cached_forwards: dict = {}


def cache_forwards(lm: LanguageModel):
    global _cached_forwards
    _cached_forwards = {}
    for name, module in lm._model.named_modules():
        _cached_forwards[name] = {}
        for func_name in CACHEABLE_FUNCS:
            if hasattr(module, func_name):
                _cached_forwards[name][func_name] = getattr(module, func_name)


def reset_forwards(lm: LanguageModel):
    for name, module in lm._model.named_modules():
        for func_name in CACHEABLE_FUNCS:
            if hasattr(module, func_name):
                setattr(module, func_name, _cached_forwards[name][func_name])


def load_LM(model_key: str, **kwargs) -> LanguageModel:
    model_key = get_full_model_path(model_key)
    kwargs["device_map"] = "auto"
    kwargs["dispatch"] = True
    lm = LanguageModel(model_key=model_key, **kwargs)
    lm._model.eval()
    cache_forwards(lm)
    print(f"loaded {model_key} | size: {get_model_size(lm._model)}")
    return lm


def get_full_model_path(model_name: str) -> str:
    full_path = os.path.join(DEFAULT_MODELS_DIR, model_name)
    if os.path.exists(full_path):
        return full_path
    else:
        logger.warning(
            f"""{model_name} not found in {DEFAULT_MODELS_DIR}
If not found in cache, model will be downloaded from HuggingFace to cache directory"""
        )
        return model_name


def get_model_size(
    model: torch.nn.Module, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all = param_size + buffer_size

    return bytes_to_human_readable(size_all, unit)


def bytes_to_human_readable(
    size: int, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return f"{size / denom:.3f} {unit}"


def prepare_offset_mapping(string, tokenized, special_tokens):
    """LLaMA3 tokenizer in Huggingface is buggy. This function is a workaround for the bug."""
    """
    <Test>

    prompts = ["The Eiffle Tower is located in", "The Space Needle is located in"]
    inp = prepare_input(
        prompts = prompts,
        tokenizer=mt,
        return_offsets_mapping=True,
        device="cuda"
    )

    i=1 # <to be changed>
    for token_id, offset in zip(inp["input_ids"][i], inp["offset_mapping"][i]):
        print(f"`{tokenizer.decode(token_id)}`, {offset=} | `{prompts[i][offset[0]:offset[1]]}`")

    """
    # logger.debug(f"{special_tokens}")
    offset_mapping = []
    end = 0
    for token in tokenized:
        if token in special_tokens:
            offset_mapping.append((end, end))
            continue
        # print(f"{string[end:].find(token)} | {end=}, {token=}, {string[end:]}")
        next_tok_idx = string[end:].find(token)
        assert next_tok_idx != -1, f"{token} not found in {string[end:]}"
        assert next_tok_idx in [
            0,
            1,
        ], f"{token} not found at the beginning of the string"

        start = end
        end = start + string[end:].find(token) + len(token)
        offset_mapping.append((start, end))
    return offset_mapping


def prepare_input(
    prompts: str | list[str],
    tokenizer: LanguageModel | Tokenizer,
    n_gen_per_prompt: int = 1,
    device: torch.device = "cpu",
    add_bos_token: bool = False,
    return_offsets_mapping=False,
) -> TokenizerOutput:
    """Prepare input for the model."""
    if isinstance(tokenizer, LanguageModel):
        device = unwrap_model(tokenizer).device
    calculate_offsets = return_offsets_mapping and (
        isinstance(tokenizer, LanguageModel)
        and "llama-3" in tokenizer.config._name_or_path.lower()
    )

    tokenizer = unwrap_tokenizer(tokenizer)
    prompts = [prompts] if isinstance(prompts, str) else prompts
    if add_bos_token:
        prompts = [maybe_prefix_bos(tokenizer, p) for p in prompts]
    prompts = [p for p in prompts for _ in range(n_gen_per_prompt)]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        return_offsets_mapping=return_offsets_mapping,
    )

    if calculate_offsets:
        offsets = []
        for i in range(len(prompts)):
            tokenized = [tokenizer.decode(t) for t in inputs["input_ids"][i]]
            offsets.append(
                prepare_offset_mapping(
                    string=prompts[i],
                    tokenized=tokenized,
                    special_tokens=tokenizer.all_special_tokens,
                )
            )
        inputs["offset_mapping"] = torch.tensor(offsets)

    inputs = inputs.to(device)
    return inputs


def unwrap_model(
    net: LanguageModel | torch.nn.Module,
) -> torch.nn.Module:
    if isinstance(net, LanguageModel):
        return net._model
    if isinstance(net, torch.nn.Module):
        return net
    raise ValueError("mt must be a nnsight.LanguageModel or a torch.nn.Module")


def unwrap_tokenizer(mt: LanguageModel | Tokenizer) -> Tokenizer:
    if isinstance(mt, LanguageModel):
        return mt.tokenizer
    return mt


def maybe_prefix_bos(tokenizer, prompt: str) -> str:
    """Prefix prompt with EOS token if model has no special start token."""
    tokenizer = unwrap_tokenizer(tokenizer)
    if hasattr(tokenizer, "bos_token"):
        prefix = tokenizer.bos_token
        if not prompt.startswith(prefix):
            prompt = prefix + " " + prompt
    return prompt


def get_module_nnsight(model, layer_name):
    layer = model
    for name in layer_name.split("."):
        layer = layer[int(name)] if name.isdigit() else getattr(layer, name)
    return layer
