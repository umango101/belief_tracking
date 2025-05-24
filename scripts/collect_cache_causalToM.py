import json
import os
import random
import sys

import fire
import nnsight
import torch
from nnsight import CONFIG, LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import env_utils
from src.dataset import Dataset, Sample

os.environ["NDIF_KEY"] = env_utils.load_env_var("NDIF_KEY")
os.environ["HF_TOKEN"] = env_utils.load_env_var("HF_WRITE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(10)

CONFIG.APP.REMOTE_LOGGING = True


class CacheCollector:
    def __init__(
        self,
        n_samples: int = 500,
        batch_size: int = 1,
        is_remote: bool = False,
        save_interval: int = 5,
        seed: int = 10,
        remote: bool = False,
    ):
        """Initialize the cache collector.

        Args:
            n_samples: Number of samples to collect
            batch_size: Batch size for processing
            is_remote: Whether to use remote model
            model_name: Name of the model to use (if None, will be set based on is_remote)
            save_interval: How often to save intermediate results
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.is_remote = is_remote
        self.save_interval = save_interval
        self.remote = remote
        random.seed(seed)

        # Load data
        self.all_characters = json.load(
            open(
                os.path.join(
                    env_utils.DEFAULT_DATA_DIR, "synthetic_entities", "characters.json"
                ),
                "r",
            )
        )
        self.all_objects = json.load(
            open(
                os.path.join(
                    env_utils.DEFAULT_DATA_DIR, "synthetic_entities", "bottles.json"
                ),
                "r",
            )
        )
        self.all_states = json.load(
            open(
                os.path.join(
                    env_utils.DEFAULT_DATA_DIR, "synthetic_entities", "drinks.json"
                ),
                "r",
            )
        )

        # Set up model
        if is_remote:
            self.model = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct")
            self.model_name = "llama-405B-Instruct"
        else:
            self.model = LanguageModel(
                "meta-llama/Meta-Llama-3-70B-Instruct",
                device_map="auto",
                torch_dtype=torch.float16,
                dispatch=True,
            )
            self.model_name = "llama-3-70B-Instruct"

        # Define token indices
        self.first_visibility_sent = list(range(169, 176))
        self.second_visibility_sent = list(range(176, 183))
        self.charac_indices = [131, 133, 146, 147, 158, 159]
        self.object_indices = [150, 151, 162, 163]
        self.state_indices = [155, 156, 167, 168]
        self.query_charac_indices = [-8, -7]
        self.query_object_indices = [-5, -4]
        self.query_sent_with_vis = list(range(183, 195))
        self.query_sent_no_vis = list(range(169, 181))

    def generate_configs(self):
        """Generate sample configurations."""
        configs = []
        for _ in range(self.n_samples):
            template_idx = 1
            characters = random.sample(self.all_characters, 2)
            containers = random.sample(self.all_objects, 2)
            states = random.sample(self.all_states, 2)

            sample = Sample(
                template_idx=template_idx,
                characters=characters,
                containers=containers,
                states=states,
            )
            configs.append(sample)
        return configs

    def initialize_activation_tensors(self):
        """Initialize tensors to store activations."""
        return {
            "last_token_acts": torch.zeros(
                self.n_samples,
                self.model.config.num_hidden_layers,
                1,
                self.model.config.hidden_size,
            ).cpu(),
            "query_charac_acts": torch.zeros(
                self.n_samples,
                self.model.config.num_hidden_layers,
                len(self.query_charac_indices),
                self.model.config.hidden_size,
            ).cpu(),
            "query_object_acts": torch.zeros(
                self.n_samples,
                self.model.config.num_hidden_layers,
                len(self.query_object_indices),
                self.model.config.hidden_size,
            ).cpu(),
            "charac_token_acts": torch.zeros(
                self.n_samples,
                self.model.config.num_hidden_layers,
                len(self.charac_indices),
                self.model.config.hidden_size,
            ).cpu(),
            "object_tokens_acts": torch.zeros(
                self.n_samples,
                self.model.config.num_hidden_layers,
                len(self.object_indices),
                self.model.config.hidden_size,
            ).cpu(),
            "state_tokens_acts": torch.zeros(
                self.n_samples,
                self.model.config.num_hidden_layers,
                len(self.state_indices),
                self.model.config.hidden_size,
            ).cpu(),
            "second_vis_acts": torch.zeros(
                self.n_samples,
                self.model.config.num_hidden_layers,
                len(self.second_visibility_sent),
                self.model.config.hidden_size,
            ).cpu(),
        }

    def save_activations(self, activations):
        """Save activation tensors to disk."""
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = f"{ROOT_DIR}/caches/{self.model_name}"
        os.makedirs(cache_dir, exist_ok=True)

        for name, tensor in activations.items():
            torch.save(tensor, os.path.join(cache_dir, f"{name}.pt"))

    def collect_activations(self):
        """Main method to collect activations."""
        CONFIG.APP.REMOTE_LOGGING = True

        configs = self.generate_configs()
        activations = self.initialize_activation_tensors()

        for i in tqdm(range(len(configs) // 20)):
            n_samples = 20
            dataset = Dataset(configs[i * 20 : (i + 1) * 20])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            with torch.inference_mode():
                with self.model.session(remote=self.remote) as session:
                    # Initialize temporary tensors
                    tmp_tensors = {
                        "last_token": torch.zeros(
                            n_samples,
                            self.model.config.num_hidden_layers,
                            1,
                            self.model.config.hidden_size,
                        )
                        .cpu()
                        .save(),
                        "query_charac": torch.zeros(
                            n_samples,
                            self.model.config.num_hidden_layers,
                            len(self.query_charac_indices),
                            self.model.config.hidden_size,
                        )
                        .cpu()
                        .save(),
                        "query_object": torch.zeros(
                            n_samples,
                            self.model.config.num_hidden_layers,
                            len(self.query_object_indices),
                            self.model.config.hidden_size,
                        )
                        .cpu()
                        .save(),
                        "charac_token": torch.zeros(
                            n_samples,
                            self.model.config.num_hidden_layers,
                            len(self.charac_indices),
                            self.model.config.hidden_size,
                        )
                        .cpu()
                        .save(),
                        "object_tokens": torch.zeros(
                            n_samples,
                            self.model.config.num_hidden_layers,
                            len(self.object_indices),
                            self.model.config.hidden_size,
                        )
                        .cpu()
                        .save(),
                        "state_tokens": torch.zeros(
                            n_samples,
                            self.model.config.num_hidden_layers,
                            len(self.state_indices),
                            self.model.config.hidden_size,
                        )
                        .cpu()
                        .save(),
                        "second_vis": torch.zeros(
                            n_samples,
                            self.model.config.num_hidden_layers,
                            len(self.second_visibility_sent),
                            self.model.config.hidden_size,
                        )
                        .cpu()
                        .save(),
                    }

                    bi = nnsight.list([0])
                    with session.iter(dataloader) as data:
                        prompt = data["prompt"]

                        with self.model.trace(prompt) as tracer:
                            for l in range(self.model.config.num_hidden_layers):
                                tmp_tensors["last_token"][bi[-1], l, 0] = (
                                    self.model.model.layers[l].output[0][0, -1].cpu()
                                )

                                for t_idx, t in enumerate(self.query_charac_indices):
                                    tmp_tensors["query_charac"][bi[-1], l, t_idx] = (
                                        self.model.model.layers[l].output[0][0, t].cpu()
                                    )

                                for t_idx, t in enumerate(self.query_object_indices):
                                    tmp_tensors["query_object"][bi[-1], l, t_idx] = (
                                        self.model.model.layers[l].output[0][0, t].cpu()
                                    )

                                for t_idx, t in enumerate(self.charac_indices):
                                    tmp_tensors["charac_token"][bi[-1], l, t_idx] = (
                                        self.model.model.layers[l].output[0][0, t].cpu()
                                    )

                                for t_idx, t in enumerate(self.object_indices):
                                    tmp_tensors["object_tokens"][bi[-1], l, t_idx] = (
                                        self.model.model.layers[l].output[0][0, t].cpu()
                                    )

                                for t_idx, t in enumerate(self.state_indices):
                                    tmp_tensors["state_tokens"][bi[-1], l, t_idx] = (
                                        self.model.model.layers[l].output[0][0, t].cpu()
                                    )

                                for t_idx, t in enumerate(self.second_visibility_sent):
                                    tmp_tensors["second_vis"][bi[-1], l, t_idx] = (
                                        self.model.model.layers[l].output[0][0, t].cpu()
                                    )

                        bi.append(bi[-1] + 1)

                # Update main activation tensors
                activations["last_token_acts"][i * 20 : (i + 1) * 20] = tmp_tensors[
                    "last_token"
                ].value
                activations["query_charac_acts"][i * 20 : (i + 1) * 20] = tmp_tensors[
                    "query_charac"
                ].value
                activations["query_object_acts"][i * 20 : (i + 1) * 20] = tmp_tensors[
                    "query_object"
                ].value
                activations["charac_token_acts"][i * 20 : (i + 1) * 20] = tmp_tensors[
                    "charac_token"
                ].value
                activations["object_tokens_acts"][i * 20 : (i + 1) * 20] = tmp_tensors[
                    "object_tokens"
                ].value
                activations["state_tokens_acts"][i * 20 : (i + 1) * 20] = tmp_tensors[
                    "state_tokens"
                ].value
                activations["second_vis_acts"][i * 20 : (i + 1) * 20] = tmp_tensors[
                    "second_vis"
                ].value

                # Save intermediate results
                if i % self.save_interval == 0:
                    self.save_activations(activations)
                    print(f"Saved activations for {i * 20} samples")

                del tmp_tensors
                torch.cuda.empty_cache()

        # Save final results
        self.save_activations(activations)
        return "Cache collection completed successfully!"


def main(
    n_samples: int = 500,
    batch_size: int = 1,
    is_remote: bool = False,
    save_interval: int = 5,
    seed: int = 10,
    remote: bool = False,
):
    """Main function to collect cache activations.

    Args:
        n_samples: Number of samples to collect
        batch_size: Batch size for processing
        is_remote: Whether to use remote model
        save_interval: How often to save intermediate results
        seed: Random seed for reproducibility
    """
    collector = CacheCollector(
        n_samples=n_samples,
        batch_size=batch_size,
        is_remote=is_remote,
        save_interval=save_interval,
        seed=seed,
        remote=remote,
    )
    return collector.collect_activations()


if __name__ == "__main__":
    fire.Fire(main)
