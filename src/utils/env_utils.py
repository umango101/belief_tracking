"""Tools for reading and controlling the runtime environment."""

import logging
import os
from typing import Union

import yaml

ENV_DATA_DIR = "RELATIONS_DATA_DIR"
ENV_MODELS_DIR = "RELATIONS_MODELS_DIR"
ENV_RESULTS_DIR = "RELATIONS_RESULTS_DIR"

logger = logging.getLogger(__name__)

try:
    PROJECT_ROOT = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])
    with open(os.path.join(PROJECT_ROOT, "env.yml"), "r") as f:
        config = yaml.safe_load(f)
        DEFAULT_MODELS_DIR = config["MODEL_DIR"]
        DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, config["DATA_DIR"])
        DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, config["RESULTS_DIR"])

        for dir in [
            DEFAULT_MODELS_DIR,
            DEFAULT_DATA_DIR,
            DEFAULT_RESULTS_DIR,
        ]:
            if not os.path.exists(dir):
                os.makedirs(dir)

except FileNotFoundError:
    logger.error(
        f'''env.yml not found in {PROJECT_ROOT}!
Setting MODEL_ROOT="". Models will now be downloaded to conda env cache, if not already there
Other defaults are set to:
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    HPARAMS_DIR = "hparams"'''
    )
    DEFAULT_MODELS_DIR = ""
    DEFAULT_DATA_DIR = "data"
    DEFAULT_RESULTS_DIR = "results"


def load_env_var(var: str) -> Union[str, None]:
    try:
        PROJECT_ROOT = "/".join(
            os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]
        )
        with open(os.path.join(PROJECT_ROOT, "env.yml"), "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"""env.yml not found in {PROJECT_ROOT}!""")

    if var not in config or config[var] is None or config[var] == "":
        logger.error(f"{var} not set in env.yml!")
        return None

    else:
        return config[var]
