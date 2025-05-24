import json
import os
import sys
from typing import Literal

import fire
import torch
from nnsight import LanguageModel
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_exp_utils import (
    exp_to_vec_type,
    load_basis_directions,
    prepare_dataset,
    set_seed,
)
from scripts.utils import (
    get_low_rank_projection,
    validate,
)
from src.utils import env_utils

os.environ["NDIF_KEY"] = env_utils.load_env_var("NDIF_KEY")
os.environ["HF_TOKEN"] = env_utils.load_env_var("HF_WRITE")


def run_experiment(
    experiment_name: Literal[
        "answer_lookback-pointer",
        "answer_lookback-payload",
        "binding_lookback-pointer_object",
        "binding_lookback-pointer_character",
        "binding_lookback-address_and_payload",
        "visibility_lookback-source",
        "visibility_lookback-payload",
        "visibility_lookback-address_and_pointer",
        "binding_lookback-pointer_character_and_object",
    ],
    lm: LanguageModel,
    layers: list[int] = list(range(34, 60, 2)),
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 4,
    verbose: bool = False,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.1,
    save_path: str = "results",
    save_outputs_on_val: bool = False,
    remote: bool = False,
):
    print("#" * 30)
    print(f"Running experiment: {experiment_name}")
    print("#" * 30)

    lm_shorthand = lm.config._name_or_path.split("/")[-1]

    save_path = os.path.join(
        save_path,
        lm_shorthand,
        experiment_name.split("-")[0],
        experiment_name.split("-")[1],
    )
    os.makedirs(save_path, exist_ok=True)

    train_dataloader, valid_dataloader = prepare_dataset(
        lm=lm,
        experiment_name=experiment_name,
        train_size=train_size,
        valid_size=validation_size,
        batch_size=batch_size,
        remote=remote,
    )

    singular_vectors = None
    exclude_projections = []

    skip_low_rank_projection = (experiment_name in exclude_projections) or (
        "405B" in lm.config._name_or_path
    )

    if not skip_low_rank_projection:
        if (
            not experiment_name.startswith("vis")
            or experiment_name == "visibility_lookback-source"
        ):
            direction_type = exp_to_vec_type[experiment_name]
            if isinstance(direction_type, list):
                directions = {
                    d: load_basis_directions("singular_vecs", d) for d in direction_type
                }
                singular_vectors = {
                    l: {d: directions[d][l] for d in direction_type} for l in layers
                }
            else:
                singular_vectors = load_basis_directions(
                    direction_type="singular_vecs",
                    vector_type=exp_to_vec_type[experiment_name],
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
            save_outputs=save_outputs_on_val,
            projection_type="full_rank",
            remote=remote,
        )
        print("-" * 30)
        print(f"Full state patching val: {full_acc}")
        print("-" * 30)
        layer_performance["full_rank"] = {"accuracy": full_acc, "rank": None}

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
                basis_directions=singular_vectors[layer],
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                lamb=lamb,
                verbose=verbose,
                remote=remote,
            )

            print("validating ...")
            singular_acc = validate(
                experiment_name,
                lm,
                layer,
                valid_dataloader,
                projection=singular_projection,
                verbose=verbose,
                save_outputs=save_outputs_on_val,
                projection_type="singular_vector",
                remote=remote,
            )
            print("-" * 30)
            print(
                f"Singular vector patching val: {singular_acc} | Rank: {singular_metadata['rank']}"
            )
            print("-" * 30)

            layer_performance["singular_vector"] = {
                "accuracy": singular_acc,
                "rank": singular_metadata["rank"],
                "metadata": {
                    "training_args": training_metadata,
                    "mask": singular_metadata["mask"],
                },
            }

        # save results after each layer
        with open(os.path.join(save_path, f"{layer}.json"), "w") as f:
            json.dump(
                layer_performance,
                f,
                indent=4,
            )


experiment_layers = {
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "answer_lookback-pointer": list(range(30, 60, 1)),
        "answer_lookback-payload": list(range(50, 80, 1)),
        "binding_lookback-pointer_object": list(range(10, 40, 1)),
        "binding_lookback-pointer_character": list(range(10, 40, 1)),
        "binding_lookback-address_and_payload": list(range(25, 45, 1)),
        "visibility_lookback-source": list(range(0, 40, 1)),
        "visibility_lookback-payload": list(range(0, 60, 1)),
        "visibility_lookback-address_and_pointer": list(range(0, 60, 1)),
        "vis_2nd_to_1st_and_ques": list(range(0, 60, 1)),
        "pointer": list(range(10, 40, 1)),
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "answer_lookback-payload": list(range(60, 126, 2)),
        "answer_lookback-pointer": list(range(40, 80, 1)),
        "binding_lookback-pointer_object": list(range(10, 50, 2)),
        "binding_lookback-pointer_character": list(range(10, 50, 2)),
        "binding_lookback-address_and_payload": list(range(20, 70, 2)),
        "visibility_lookback-source": list(range(0, 50, 2)),
        "visibility_lookback-payload": list(range(40, 80, 2)),
        "visibility_lookback-address_and_pointer": list(range(0, 80, 2)),
    },
}


def main(
    experiment: Literal[
        "answer_lookback-pointer",
        "answer_lookback-payload",
        "binding_lookback-pointer_object",
        "binding_lookback-pointer_character",
        "binding_lookback-address_and_payload",
        "binding_lookback-character_oi",
        "binding_lookback-object_oi",
        "visibility_lookback-source",
        "visibility_lookback-payload",
        "visibility_lookback-address_and_pointer",
        "vis_2nd_to_1st_and_ques",
        "pointer",
    ],
    model_key: str = "meta-llama/Meta-Llama-3-70B-Instruct",
    layers: list[int] = None,
    train_size: int = 80,
    validation_size: int = 80,
    batch_size: int = 1,
    verbose: bool = False,
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = None,
    save_path: str = "experiments/causalToM_novis/results",
    save_outputs: bool = False,
    remote: bool = False,
):
    """
    Run single layer patching experiments.

    Args:
        experiment: Type of experiment to run
        model_key: Model to use
        layers: List of layers to run experiments on
        train_size: Number of training samples
        validation_size: Number of validation samples
        batch_size: Batch size for training and validation
        verbose: Whether to print verbose output
        learning_rate: Learning rate for training
        n_epochs: Number of training epochs
        lamb: L1 regularization parameter
        save_path: Path to save results
        save_outputs: Whether to save outputs
        remote: Whether to run experiments remotely
    """
    if lamb is None:
        if experiment == "visibility_lookback-source":
            lamb = 0.03
        else:
            lamb = 0.1

    print(f"Running experiment: {experiment}")
    print(f"Model: {model_key}")
    print(f"Layers: {layers}")
    print(f"Train size: {train_size}")
    print(f"Validation size: {validation_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}")
    print(f"Lambda: {lamb}")
    print(f"Save path: {save_path}")
    print(f"Save outputs: {save_outputs}")
    print(f"Remote: {remote}")

    if remote:
        lm = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct")
    else:
        lm = LanguageModel(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            dispatch=True,
        )

    if layers is None:
        n_layer = lm.config.num_hidden_layers
        layers = sorted(
            list(
                set(
                    list(range(0, n_layer, 10))
                    + experiment_layers[model_key].get(experiment, [])
                    + [n_layer - 1]
                )
            )
        )

    set_seed(10)
    run_experiment(
        experiment_name=experiment,
        lm=lm,
        layers=layers,
        train_size=train_size,
        validation_size=validation_size,
        batch_size=batch_size,
        verbose=verbose,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        lamb=lamb,
        save_path=save_path,
        save_outputs_on_val=save_outputs and not remote,
        remote=remote,
    )


if __name__ == "__main__":
    fire.Fire(main)
