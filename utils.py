import json
import torch
from datasets import Dataset


def get_dataset(datafiles: list[str]) -> Dataset:
    with open(datafiles[0]) as f:
        unexpected_contents = [json.loads(line) for line in f]

    with open(datafiles[1]) as f:
        unexpected_transfer = [json.loads(line) for line in f]

    tasks = []
    for data in unexpected_contents + unexpected_transfer:
        for i in range(3):
            inp = {"input": data["prompts"][i], "target": data[f"target_{i+1}"]}
            tasks.append(inp)

    return Dataset.from_list(tasks).with_format("torch")


def collate_fn(model, tokenizer, examples) -> dict[str, torch.Tensor]:
    inputs = tokenizer(
        [ex["input"] for ex in examples],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    if (
        model.config.architectures[0] == "GPT2LMHeadModel"
        or model.config.architectures[0] == "GPTNeoXForCausalLM"
        or model.config.architectures[0] == "GPTJForCausalLM"
    ):
        inputs["target"] = [tokenizer.encode(" " + ex["target"])[0] for ex in examples]
    elif (
        model.config.architectures[0] == "LlamaForCausalLM"
        or model.config.architectures[0] == "LlaMAForCausalLM"
    ):
        inputs["target"] = [tokenizer.encode(ex["target"])[1] for ex in examples]
    elif model.config.architectures[0] == "MistralForCausalLM":
        inputs["target"] = [tokenizer.encode(" " + ex["target"])[2] for ex in examples]
    else:
        raise NotImplementedError
    return inputs
