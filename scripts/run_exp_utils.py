import ctypes
import gc
import inspect
import json
import os
import random
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import STORY_TEMPLATES
from src.utils import env_utils


def free_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()


def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(tid), ctypes.py_object(exctype)
    )
    if res == 0:
        raise ValueError("Invalid thread id")
    elif res != 1:
        # If it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class ThreadWithId(threading.Thread):
    def get_id(self):
        # Returns the thread ID
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for tid, thread in threading._active.items():
            if thread is self:
                self._thread_id = tid
                return tid
        return None

    def terminate(self):
        # Terminates the thread by raising an exception in it
        thread_id = self.get_id()
        if thread_id is not None:
            _async_raise(thread_id, SystemExit)


def send_request_to_ndif(nnsight_request: Callable, timeout=72000, n_try=5) -> Any:
    """
    Execute a remote request with timeout handling and retries.
    This implementation forcibly terminates the worker thread if it exceeds the timeout.

    Args:
        nnsight_request: Callable function that executes a request on a remote server
        timeout: Maximum time in seconds to wait for the request to complete (default value: 5 minutes)
        n_try: Maximum number of retry attempts

    Returns:
        The result of the successful nnsight_request call

    Raises:
        RuntimeError: If the request fails after n_try attempts
    """
    for attempt in range(1, n_try + 1):
        result_container = []
        exception_container = []

        def worker_function():
            try:
                result = nnsight_request()
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)

        worker_thread = ThreadWithId(target=worker_function)
        worker_thread.daemon = (
            True  # Allow the thread to be terminated when main thread exits
        )
        worker_thread.start()

        # Wait for the thread to complete or timeout
        worker_thread.join(timeout=timeout)

        # Check if the thread is still alive after the timeout
        if worker_thread.is_alive():
            print(f"Request timed out (attempt {attempt}/{n_try})")
            # Forcibly terminate the thread
            worker_thread.terminate()
            if attempt == n_try:
                raise RuntimeError(
                    f"Request failed after {n_try} attempts (timeout: {timeout}s)"
                )
        else:
            # Thread completed within the timeout
            if exception_container:
                # An exception occurred in the worker thread
                print(
                    f"Request failed with error: {str(exception_container[0])} (attempt {attempt}/{n_try})"
                )
                if attempt == n_try:
                    raise RuntimeError(
                        f"Request failed after {n_try} attempts: {str(exception_container[0])}"
                    )
            else:
                # Thread completed successfully
                return result_container[0]

    raise RuntimeError(
        f"END OF FUNCTION (shouldn't reach): Request failed after {n_try} attempts"
    )


from utils import (
    get_charac_pos_exp,
    get_obj_pos_exps,
    get_pos_trans_exps,
    get_state_pos_exps,
    get_unidirectional_visibility_exps,
    get_value_fetcher_exps,
    query_charac_pos,
    query_obj_pos,
)

exp_to_ds_func_map = {
    # "position_transmitter": get_pos_trans_exps,
    "position_transmitter": get_pos_trans_exps,
    "position_transmitter_nikhil": get_pos_trans_exps,
    "value_fetcher": get_value_fetcher_exps,
    "query_object": query_obj_pos,
    "query_character": query_charac_pos,
    "state_position": get_state_pos_exps,
    "object_position": get_obj_pos_exps,
    "character_position": get_charac_pos_exp,
    "vis_2nd": get_unidirectional_visibility_exps,
    "vis_ques": get_unidirectional_visibility_exps,
    "vis_2nd_and_ques": get_unidirectional_visibility_exps,
    "vis_2nd_to_1st_and_ques": get_unidirectional_visibility_exps,
    "source_1": get_pos_trans_exps,
    "source_2": get_pos_trans_exps,
    "pointer": get_pos_trans_exps,
}

exp_to_vec_type = {
    "position_transmitter": "last_token",
    "value_fetcher": "last_token",
    "query_object": "query_obj_ordering_id",
    "query_character": "query_charac_ordering_id",
    "state_position": "state_ordering_id",
    "object_position": "object_ordering_id",
    "character_position": "character_ordering_id",
    "vis_2nd": "second_visibility_sent",
    "vis_ques": None,
    "vis_2nd_and_ques": None,
    "vis_2nd_to_1st_and_ques": None,
    "source_1": ["object_ordering_id", "character_ordering_id"],
    "source_2": ["object_ordering_id", "character_ordering_id"],
    "pointer": ["query_obj_ordering_id", "query_charac_ordering_id"],
}

exp_to_intervention_positions = {
    "position_transmitter": {
        "cache": [-1],
        "patch": [-1],
    },
    "value_fetcher": {
        "cache": [-1],
        "patch": [-1],
    },
    "query_object": {
        "cache": [-5, -4],
        "patch": [-5, -4],
    },
    "query_character": {
        "cache": [-8, -7],
        "patch": [-8, -7],
    },
    "state_position": {
        "cache": [155, 156, 167, 168],
        "patch": [167, 168, 155, 156],
    },
    "vis_2nd": {
        "cache": [i for i in range(176, 183)],
        "patch": [i for i in range(176, 183)],
    },
    "vis_ques": {
        "cache": [i for i in range(183, 195)],
        "patch": [i for i in range(183, 195)],
    },
    "vis_2nd_and_ques": {
        "cache": [i for i in range(176, 195)],  # second visibility + question
        "patch": [i for i in range(176, 195)],  # second visibility + question
    },
    "vis_2nd_to_1st_and_ques": {
        "cache": [i for i in range(176, 183)]
        + [i for i in range(183, 195)],  # second visibility + question
        "patch": [i for i in range(169, 176)]
        + [i for i in range(183, 195)],  # first visibility + question
    },
    "pointer": {
        "cache": [-5, -4] + [-8, -7],  # query_object + query_character
        "patch": [-5, -4] + [-8, -7],
    },
}


def set_seed(seed: int) -> None:
    """Globally set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


directions_folder = {
    "singular_vecs": "svd_results",
    "principal_components": "pca_results",
}


def load_basis_directions(
    direction_type: Literal["singular_vecs", "principal_components"],
    vector_type: (
        Literal[
            "last_token",
            "state_ordering_id",
            "object_ordering_id",
            "character_ordering_id",
            "query_charac_ordering_id",
            "query_obj_ordering_id",
            "second_visibility_sent",
        ]
        | None
    ),
    path: str = "belief_tracking",
    prefix: str = "",
) -> dict[int, torch.Tensor]:
    if vector_type is None:
        print("WARNING: No direction type specified")
        return None
    path = os.path.join(
        prefix, directions_folder[direction_type], path, vector_type, direction_type
    )
    directions = defaultdict(torch.Tensor)
    for file in os.listdir(path):
        idx = int(file.split(".")[0])
        directions[idx] = torch.load(os.path.join(path, file)).cpu()

    print(
        f"""Loaded {len(directions)} {direction_type} directions for {vector_type} | shape: {directions[0].shape}"""
    )

    return directions


@torch.inference_mode()
def check_lm_on_sample(lm, sample, remote=False):
    lm.tokenizer.padding_side = "left"
    # lm.model.eval()

    prompts = [sample["corrupt_prompt"], sample["clean_prompt"]]
    answers = [
        sample["corrupt_ans"] if "corrupt_ans" in sample else sample["corrupt_target"],
        sample["clean_ans"] if "clean_ans" in sample else sample["clean_target"],
    ]
    inputs = lm.tokenizer(prompts, return_tensors="pt", padding=True)
    if remote == False:
        inputs = inputs.to(lm.device)

    def nnsight_request():
        # print(inputs)
        with lm.trace(inputs, remote=remote) as tracer:
            logits = lm.lm_head.output[:, -1]
            predicted = torch.argmax(logits, dim=-1).save()
        return predicted

    # predicted = process_nnsight_request(nnsight_request, timeout=100, n_try=5)
    # predicted = nnsight_request()
    predicted = (
        nnsight_request()
        if remote == False
        else send_request_to_ndif(nnsight_request, timeout=100, n_try=5)
    )

    # with lm.trace(inputs, remote=remote) as tracer:
    #     logits = lm.lm_head.output[:, -1]
    #     predicted = torch.argmax(logits, dim=-1).save()

    predicted = predicted.value if remote == True else predicted
    predicted = [lm.tokenizer.decode(pred) for pred in predicted]

    ok = True
    for ans, pred in zip(answers, predicted):
        # print(f'Predicted: "{pred}", Target: "{ans}"')
        if pred.lower().strip() != ans.lower().strip():
            ok = False

    # print(f"{sample}")
    # print(f"{answers=} | {predicted=}")
    # print(f"{ok=}")
    # print("-" * 30)

    return ok


@torch.inference_mode()
def filter_dataset_on_lm(lm, dataset, size=None, remote=False):
    filtered_dataset = []
    size = size if size is not None else len(dataset)
    correct, total = 0, 0
    progress_bar = tqdm(dataset)
    for sample in progress_bar:
        total += 1
        if check_lm_on_sample(lm, sample, remote=remote):
            filtered_dataset.append(sample)
            correct += 1
        if len(filtered_dataset) == size:
            break
        progress_bar.set_description(
            f"LM accuracy: {correct / total:.2f}({correct}/{total})"
        )
    return filtered_dataset, correct / total


def prepare_dataset(
    experiment_name: str,
    train_size: int = 80,
    valid_size: int = 80,
    batch_size: int = 4,
    lm: LanguageModel | None = None,
    remote: bool = False,
):
    all_states = {}
    all_containers = {}
    all_characters = json.load(
        open(
            os.path.join(
                env_utils.DEFAULT_DATA_DIR, "synthetic_entities", "characters.json"
            ),
            "r",
        )
    )

    for TYPE, DCT in {"states": all_states, "containers": all_containers}.items():
        ROOT = os.path.join(env_utils.DEFAULT_DATA_DIR, "synthetic_entities", TYPE)
        for file in os.listdir(ROOT):
            file_path = os.path.join(ROOT, file)
            with open(file_path, "r") as f:
                names = json.load(f)
            DCT[file.split(".")[0]] = names

    specific_kwargs = {}
    if experiment_name.startswith("vis"):
        specific_kwargs["additional_characs"] = (
            experiment_name == "vis_2nd_to_1st_and_ques"
        )
    else:
        specific_kwargs["question_type"] = "belief_question"
    dataset = exp_to_ds_func_map[experiment_name](
        STORY_TEMPLATES,
        all_characters,
        all_containers,
        all_states,
        2 * (train_size + valid_size),
        **specific_kwargs,
    )

    if lm is not None:
        print("Filtering dataset on LM...")
        dataset, lm_acc = filter_dataset_on_lm(
            lm, dataset, train_size + valid_size, remote=remote
        )
        print("-" * 30)
        print(f"Len: {len(dataset)} | LM accuracy: {lm_acc:.2f}")
        print("-" * 30)
        if len(dataset) < train_size + valid_size:
            print("WARNING: Dataset is smaller than train_size + valid_size")
        if len(dataset) < train_size + valid_size // 2:
            raise ValueError("Dataset is too small")

    train_dataset = dataset[:train_size]
    valid_dataset = dataset[train_size:]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader


@dataclass(frozen=False)
class Accuracy(DataClassJsonMixin):
    accuracy: float
    rank: int = None  # full rank if None
    metadata: dict = None
