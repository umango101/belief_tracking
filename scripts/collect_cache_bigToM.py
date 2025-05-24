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
root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(root_dir)


from experiments.bigToM.utils import (
    get_bigtom_samples,
    get_ques_start_token_idx,
    get_visitibility_sent_start_idx,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(10)

CONFIG.APP.REMOTE_LOGGING = True


def main(
    is_remote: bool = False,
    batch_size: int = 1,
    dataset_path: str = "data/bigtom/",
    limit: int = None,
):
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(os.path.join(root_dir, dataset_path))

    if is_remote:
        model = LanguageModel("meta-llama/Meta-Llama-3.1-405B-Instruct")
        model_name = "Llama-3.1-405B-Instruct"
    else:
        model = LanguageModel(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            dispatch=True,
        )
        model_name = "Llama-3-70B-Instruct"

    df_false = pd.read_csv(
        os.path.join(dataset_path, "0_forward_belief_false_belief", "stories.csv"),
        delimiter=";",
    )
    df_true = pd.read_csv(
        os.path.join(dataset_path, "0_forward_belief_true_belief", "stories.csv"),
        delimiter=";",
    )

    fb_samples = get_bigtom_samples(df_false, df_true, len(df_false), "false_belief")
    tb_samples = get_bigtom_samples(df_false, df_true, len(df_true), "true_belief")
    if limit is not None:
        fb_samples = fb_samples[:limit]
        tb_samples = tb_samples[:limit]
    dataloader = DataLoader(
        tb_samples + fb_samples, batch_size=batch_size, shuffle=False
    )

    # Compute the length of longest visibility sentence in dataloader
    max_len = 0
    for data in dataloader:
        prompt = data["prompt"][0]
        visibility_sent_start_idx = get_visitibility_sent_start_idx(
            model.tokenizer, prompt
        )
        ques_start_idx = get_ques_start_token_idx(model.tokenizer, prompt)
        max_len = max(max_len, ques_start_idx - visibility_sent_start_idx)

    n_samples = len(dataloader.dataset)
    last_token_acts = torch.zeros(
        n_samples, model.config.num_hidden_layers, model.config.hidden_size
    )
    query_charac_acts = torch.zeros(
        n_samples, model.config.num_hidden_layers, 2, model.config.hidden_size
    )
    visibility_sent_acts = torch.zeros(
        n_samples, model.config.num_hidden_layers, max_len, model.config.hidden_size
    )
    # Visibility sentence length
    visibility_sent_lens = torch.zeros(n_samples)

    for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        prompt = data["prompt"][0]
        ques_start_idx = get_ques_start_token_idx(model.tokenizer, prompt)
        visibility_sent_start_idx = get_visitibility_sent_start_idx(
            model.tokenizer, prompt
        )
        visibility_sent_lens[bi] = ques_start_idx - visibility_sent_start_idx

        with torch.no_grad():
            with model.trace() as tracer:
                with tracer.invoke(prompt):
                    for l in range(model.config.num_hidden_layers):
                        last_token_acts[bi, l] = (
                            model.model.layers[l].output[0][0, -1].cpu().save()
                        )

                        query_charac_acts[bi, l] = (
                            model.model.layers[l]
                            .output[0][0, ques_start_idx + 3 : ques_start_idx + 5]
                            .cpu()
                            .save()
                        )

                        for t_idx, t in enumerate(
                            range(visibility_sent_start_idx, ques_start_idx)
                        ):
                            visibility_sent_acts[bi, l, t_idx] = (
                                model.model.layers[l].output[0][0, t].cpu().save()
                            )

    cache_dir = os.path.join(root_dir, "experiments", "bigToM", "caches", model_name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    torch.save(last_token_acts, os.path.join(cache_dir, "last_token_acts.pt"))
    torch.save(query_charac_acts, os.path.join(cache_dir, "query_charac_acts.pt"))
    torch.save(
        visibility_sent_acts,
        os.path.join(cache_dir, "visibility_sent_acts.pt"),
    )
    torch.save(
        visibility_sent_lens,
        os.path.join(cache_dir, "visibility_sent_lens.pt"),
    )
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
