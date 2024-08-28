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

from src.models import unwrap_tokenizer
from src.utils.typing import PredictedToken, Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


@torch.inference_mode()
def interpret_logits(
    tokenizer: LanguageModel | Tokenizer,
    logits: torch.Tensor,
    k: int = 10,
    get_proba: bool = False,
) -> list[tuple[str, float]]:
    tokenizer = unwrap_tokenizer(tokenizer)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    token_ids = logits.topk(dim=-1, k=k).indices.squeeze().tolist()
    logit_values = logits.topk(dim=-1, k=k).values.squeeze().tolist()
    return [(tokenizer.decode(t), round(v, 3)) for t, v in zip(token_ids, logit_values)]


@torch.inference_mode()
def logit_lens(
    mt: LanguageModel,
    h: torch.Tensor,
    after_layer_norm: bool = False,
    interested_tokens: list[int] = [],
    get_proba: bool = False,
    k: int = 10,
) -> tuple[list[tuple[str, float]], dict]:
    lm_head = mt.lm_head if not after_layer_norm else mt.lm_head.lm_head
    h = untuple(h) if after_layer_norm else h
    logits = lm_head(h)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    # don't pass `get_proba` or softmax will be applied twice with `get_proba=True`
    candidates = interpret_logits(mt, logits, k=k)
    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        interested_logits = {
            t: {
                "p": logits[t].item(),
                "rank": rank_tokens.index(t) + 1,
                "token": mt.tokenizer.decode(t),
            }
            for t in interested_tokens
        }
        return candidates, interested_logits
    return candidates


def untuple(object: Any):
    if isinstance(object, tuple):
        return object[0]
    return object
