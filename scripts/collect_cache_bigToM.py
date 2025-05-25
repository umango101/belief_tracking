import os
import random
import sys

import fire
import pandas as pd
import torch
from nnsight import CONFIG, LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from experiments.bigToM.utils import (
    get_bigtom_samples,
    get_ques_start_token_idx,
    get_visitibility_sent_start_idx,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(10)

CONFIG.APP.REMOTE_LOGGING = True


class BigToMCacheCollector:
    def __init__(
        self,
        batch_size: int = 1,
        dataset_path: str = "data/bigtom/",
        limit: int = None,
        is_remote: bool = False,
        save_interval: int = 5,
        seed: int = 10,
    ):
        """Initialize the cache collector.

        Args:
            batch_size: Batch size for processing
            dataset_path: Path to the dataset
            limit: Limit number of samples to process
            is_remote: Whether to use remote model
            save_interval: How often to save intermediate results
            seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.limit = limit
        self.is_remote = is_remote
        self.save_interval = save_interval
        random.seed(seed)

        if not os.path.isabs(self.dataset_path):
            self.dataset_path = os.path.abspath(
                os.path.join(root_dir, self.dataset_path)
            )

        # Set up model
        if is_remote:
            self.model = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct")
            self.model_name = "llama-3.1-405B-Instruct"
        else:
            self.model = LanguageModel(
                "meta-llama/Meta-Llama-3-70B-Instruct",
                device_map="auto",
                torch_dtype=torch.float16,
                dispatch=True,
            )
            self.model_name = "llama-3-70B-Instruct"

    def load_data(self):
        """Load and prepare the dataset."""
        df_false = pd.read_csv(
            os.path.join(
                self.dataset_path, "0_forward_belief_false_belief", "stories.csv"
            ),
            delimiter=";",
        )
        df_true = pd.read_csv(
            os.path.join(
                self.dataset_path, "0_forward_belief_true_belief", "stories.csv"
            ),
            delimiter=";",
        )

        fb_samples = get_bigtom_samples(
            df_false, df_true, len(df_false), "false_belief"
        )
        tb_samples = get_bigtom_samples(df_false, df_true, len(df_true), "true_belief")

        if self.limit is not None:
            fb_samples = fb_samples[: self.limit]
            tb_samples = tb_samples[: self.limit]

        return DataLoader(
            tb_samples + fb_samples, batch_size=self.batch_size, shuffle=False
        )

    def compute_max_length(self, dataloader):
        """Compute the maximum length of visibility sentences."""
        max_len = 0
        for data in dataloader:
            prompt = data["prompt"][0]
            visibility_sent_start_idx = get_visitibility_sent_start_idx(
                self.model.tokenizer, prompt
            )
            ques_start_idx = get_ques_start_token_idx(self.model.tokenizer, prompt)
            max_len = max(max_len, ques_start_idx - visibility_sent_start_idx)
        return max_len

    def initialize_activation_tensors(self, n_samples, max_len):
        """Initialize tensors to store activations."""
        return {
            "last_token_acts": torch.zeros(
                n_samples,
                self.model.config.num_hidden_layers,
                1,
                self.model.config.hidden_size,
            ),
            "query_charac_acts": torch.zeros(
                n_samples,
                self.model.config.num_hidden_layers,
                2,
                self.model.config.hidden_size,
            ),
            "visibility_sent_acts": torch.zeros(
                n_samples,
                self.model.config.num_hidden_layers,
                max_len,
                self.model.config.hidden_size,
            ),
            "visibility_sent_lens": torch.zeros(n_samples),
        }

    def save_activations(self, activations):
        """Save activation tensors to disk."""
        cache_dir = os.path.join(root_dir, "caches", self.model_name, "bigToM")
        os.makedirs(cache_dir, exist_ok=True)

        for name, tensor in activations.items():
            torch.save(tensor, os.path.join(cache_dir, f"{name}.pt"))

    def collect_activations(self):
        """Main method to collect activations."""
        dataloader = self.load_data()
        max_len = self.compute_max_length(dataloader)
        n_samples = len(dataloader.dataset)
        activations = self.initialize_activation_tensors(n_samples, max_len)

        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            prompt = data["prompt"][0]
            ques_start_idx = get_ques_start_token_idx(self.model.tokenizer, prompt)
            visibility_sent_start_idx = get_visitibility_sent_start_idx(
                self.model.tokenizer, prompt
            )
            activations["visibility_sent_lens"][bi] = (
                ques_start_idx - visibility_sent_start_idx
            )

            with torch.no_grad():
                with self.model.trace() as tracer:
                    with tracer.invoke(prompt):
                        for l in range(self.model.config.num_hidden_layers):
                            activations["last_token_acts"][bi, l] = (
                                self.model.model.layers[l].output[0][0, -1].cpu().save()
                            )

                            activations["query_charac_acts"][bi, l] = (
                                self.model.model.layers[l]
                                .output[0][0, ques_start_idx + 3 : ques_start_idx + 5]
                                .cpu()
                                .save()
                            )

                            for t_idx, t in enumerate(
                                range(visibility_sent_start_idx, ques_start_idx)
                            ):
                                activations["visibility_sent_acts"][bi, l, t_idx] = (
                                    self.model.model.layers[l]
                                    .output[0][0, t]
                                    .cpu()
                                    .save()
                                )

            if (bi + 1) % self.save_interval == 0:
                self.save_activations(activations)

        self.save_activations(activations)
        print("Done!")


def main(
    batch_size: int = 1,
    dataset_path: str = "data/bigtom/",
    limit: int = None,
    is_remote: bool = False,
    save_interval: int = 5,
    seed: int = 10,
):
    """Main entry point for the script.

    Args:
        batch_size: Batch size for processing
        dataset_path: Path to the dataset
        limit: Limit number of samples to process
        is_remote: Whether to use remote model
        save_interval: How often to save intermediate results
        seed: Random seed for reproducibility
    """
    collector = BigToMCacheCollector(
        batch_size=batch_size,
        dataset_path=dataset_path,
        limit=limit,
        is_remote=is_remote,
        save_interval=save_interval,
        seed=seed,
    )
    collector.collect_activations()


if __name__ == "__main__":
    fire.Fire(main)
