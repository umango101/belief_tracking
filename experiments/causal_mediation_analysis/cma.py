"""
Causal Mediation Analysis experiments with language models
"""

import json
import os
import random
import sys
from enum import Enum

import fire
from nnsight import CONFIG
from torch.utils.data import DataLoader

from utils import (
    find_correct_samples,
    get_character_tracing_exps,
    get_object_tracing_exps,
    get_state_tracing_exps,
    load_entity_data,
    load_model,
    run_tracing_experiment,
)

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
from src.dataset import STORY_TEMPLATES
from src.utils import env_utils


# Define entity types
class EntityType(str, Enum):
    CHARACTER = "character"
    OBJECT = "object"
    STATE = "state"


# Get the absolute path to the data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STORY_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "story_templates.json")


class Tracer:
    """Run Causal Mediation Analysis experiments with language models"""

    def __init__(
        self,
        entity_type="character",
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        data_dir="data",
        results_dir="tracing_results",
        num_samples=50,
        batch_size=10,
        tracing_batch_size=25,
        random_seed=10,
        start_token=180,
        start_layer=0,
        is_remote=False,
    ):
        """
        Initialize Causal Mediation Analysis experiment

        Args:
            entity_type: Type of entity to trace (character, object, or state)
            model_name: Name of the model to use
            cache_dir: Directory to cache model files
            data_dir: Directory containing the dataset
            results_dir: Directory to save results
            num_samples: Number of samples to generate
            batch_size: Initial batch size for finding correct samples
            tracing_batch_size: Batch size for tracing experiments
            random_seed: Random seed for reproducibility
            start_token: Starting token index for tracing
            start_layer: Starting layer index for tracing
        """
        # Validate entity type
        if entity_type not in [e.value for e in EntityType]:
            raise ValueError(
                f"Invalid entity type: {entity_type}. Must be one of: {', '.join([e.value for e in EntityType])}"
            )

        self.entity_type = entity_type
        self.model_name = model_name

        # Convert relative paths to absolute paths
        self.data_dir = os.path.join(PROJECT_ROOT, data_dir)
        self.results_dir = os.path.join(PROJECT_ROOT, results_dir)

        self.num_samples = num_samples
        self.batch_size = batch_size
        self.tracing_batch_size = tracing_batch_size
        self.random_seed = random_seed
        self.start_token = start_token
        self.start_layer = start_layer

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Set up configuration
        random.seed(self.random_seed)
        CONFIG.APP.REMOTE_LOGGING = False

        # Get credentials from environment variables
        nnsight_api_key = env_utils.load_env_var("NDIF_KEY")
        hf_token = env_utils.load_env_var("HF_WRITE")

        # Set credentials
        CONFIG.set_default_api_key(nnsight_api_key)
        os.environ["HF_TOKEN"] = hf_token

    def _get_dataset_generator(self):
        """Get the appropriate dataset generator based on entity type"""
        if self.entity_type == EntityType.CHARACTER:
            return get_character_tracing_exps
        elif self.entity_type == EntityType.OBJECT:
            return get_object_tracing_exps
        elif self.entity_type == EntityType.STATE:
            return get_state_tracing_exps
        else:
            raise ValueError(f"Unsupported entity type: {self.entity_type}")

    def run(self):
        """Run the Causal Mediation Analysis experiment"""
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        results_path = os.path.join(self.results_dir, f"{self.entity_type}.json")

        print(f"Running {self.entity_type} tracing experiment...")

        all_characters, all_states, all_containers = load_entity_data(self.data_dir)

        model = load_model(self.model_name, is_remote=self.is_remote)

        dataset_generator = self._get_dataset_generator()

        dataset = dataset_generator(
            STORY_TEMPLATES,
            all_characters,
            all_containers,
            all_states,
            self.num_samples * 2,
        )

        corrects = find_correct_samples(
            model, dataset, batch_size=self.batch_size, num_samples=self.num_samples
        )

        correct_dataset = [dataset[i] for i in corrects]
        correct_dataloader = DataLoader(
            correct_dataset, batch_size=self.tracing_batch_size, shuffle=False
        )
        print(
            f"Dataset size: {len(correct_dataloader.dataset)}, "
            f"Batch size: {self.tracing_batch_size}"
        )

        tracing_results = run_tracing_experiment(
            model, correct_dataloader, self.start_token, self.start_layer, results_path
        )

        with open(results_path, "w") as f:
            json.dump(tracing_results, f, indent=4)

        print(f"Results saved to {results_path}")
        return results_path


class CharacterTracer(Tracer):
    """Run character tracing experiments with language models"""

    def __init__(self, **kwargs):
        """Initialize character tracer"""
        super().__init__(entity_type=EntityType.CHARACTER, **kwargs)


class ObjectTracer(Tracer):
    """Run object tracing experiments with language models"""

    def __init__(self, **kwargs):
        """Initialize object tracer"""
        super().__init__(entity_type=EntityType.OBJECT, **kwargs)


class StateTracer(Tracer):
    """Run state tracing experiments with language models"""

    def __init__(self, **kwargs):
        """Initialize state tracer"""
        super().__init__(entity_type=EntityType.STATE, **kwargs)


def main():
    """
    Main entry point for CLI usage.

    Example usage:
    python cma.py --entity_type=character --model_name=meta-llama/Meta-Llama-3-70B-Instruct --results_dir=causal_mediation_analysis/tracing_results
    """

    def run_tracer(**kwargs):
        """Create and run the appropriate tracer based on entity_type"""
        entity_type = kwargs.pop("entity_type", "character")

        if entity_type == EntityType.CHARACTER:
            tracer = CharacterTracer(**kwargs)
        elif entity_type == EntityType.OBJECT:
            tracer = ObjectTracer(**kwargs)
        elif entity_type == EntityType.STATE:
            tracer = StateTracer(**kwargs)
        else:
            tracer = Tracer(entity_type=entity_type, **kwargs)

        return tracer.run()

    fire.Fire(run_tracer)


if __name__ == "__main__":
    main()
