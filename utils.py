import os
import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
from transformers import StoppingCriteria
from datasets import Dataset
from torch.utils.data import DataLoader


class NewLineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.newline_token_id = self.tokenizer.encode("\n")[-1]
        self.txt_generated = False

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """
        Checks if the current generated token is a new line character.
        Returns a tensor of booleans with the same shape as scores,
        indicating whether generation should stop for each sequence.
        """
        latest_token_ids = input_ids[:, -1]
        if latest_token_ids == self.newline_token_id and self.txt_generated:
            self.txt_generated = False
            return True
        else:
            if latest_token_ids != self.newline_token_id:
                self.txt_generated = True
            return False


# def get_dataset(datafiles: list[str]) -> Dataset:
#     with open(datafiles[0]) as f:
#         unexpected_contents = [json.loads(line) for line in f]

#     with open(datafiles[1]) as f:
#         unexpected_transfer = [json.loads(line) for line in f]

#     tasks = []
#     for data in unexpected_contents + unexpected_transfer:
#         for i in range(3):
#             inp = {"input": data["prompts"][i], "target": data[f"target_{i+1}"]}
#             tasks.append(inp)

#     return Dataset.from_list(tasks).with_format("torch")


def compute_final_roles(players: List[Dict[str, str]]):

    for player in players:
        player["final_role"] = player["role"]

    indices = [
        idx for idx, player in enumerate(players) if player["role"] != "Troublemaker"
    ]
    idx1, idx2 = random.sample(indices, 2)
    players[idx1]["final_role"] = players[idx2]["role"]
    players[idx2]["final_role"] = players[idx1]["role"]

    return players


def compute_role_description(players: List[Dict[str, str]]):
    for idx, player in enumerate(players):
        other_players = [p["name"] for p in players if p["name"] != player["name"]]
        concatenated_other_players = ", ".join(other_players)
        concatenated_other_players = concatenated_other_players[::-1].replace(
            ",", "dna ", 1
        )[::-1]
        if player["role"] != "Troublemaker":
            role_description = f"You are {player['name']}. You are playing Werewolf card game with your friends {concatenated_other_players}. Initially, you've been given the role of {player['role']}. First, understand the goals and actions of each player, then speak accordingly to increase your chances of winning."
        else:
            swapped_players = [p for p in players if p["role"] != p["final_role"]]

            role_description = f"You are {player['name']}. You are playing Werewolf card game with your friends {concatenated_other_players}. Initially, you've been given the role of {player['role']}. During the night phase, you swapped {swapped_players[0]['name']} and {swapped_players[1]['name']}'s cards. Share this information during the Discussion Phase to identify the Werewolf. First, understand the goals and actions of each player, then speak accordingly to increase your chances of winning."

        players[idx]["role_description"] = role_description
    return players


def ask_mental_state_questions(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    players: List[Dict[str, str]],
    game_description: str,
    game_idx: int,
    conversation: str,
):

    own_vote, others_vote = {}, {}
    with torch.no_grad():
        for player_idx in range(len(players)):
            player = players[player_idx]
            others_vote[player["name"]] = {}

            other_players = [p["name"] for p in players if p["name"] != player["name"]]

            all_players = [p["name"] for p in players]
            random.shuffle(all_players)
            options = {f"{chr(65+i)}": p for i, p in enumerate(all_players)}
            options[f"{chr(65+len(all_players))}"] = "Not sure"
            options_str = ", ".join([f"{k}: {v}" for k, v in options.items()])
            options_token_ids = [
                tokenizer.encode(f"{chr(65+i)}")[-1] for i in range(len(all_players))
            ]

            prompt = f"{game_description}\n\n{player['role_description']}\n\n"

            if os.path.exists(f"/conversations/{game_idx}.txt"):
                with open(f"/conversations/{game_idx}.txt", "r") as f:
                    conversation = f.read()
            else:
                conversation = ""

            prompt += f"DAY PHASE:\n{conversation}\n"
            question = (
                f"QUESTION: Who would you vote for?\nOptions: {options_str}\nAnswer:"
            )
            #             print(question)
            prompt += question

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            pred_option_logits = options_token_ids[logits[options_token_ids].argmax()]
            own_vote[player["name"]] = options[tokenizer.decode(pred_option_logits)]

            for other_player in other_players:
                #                 random.shuffle(all_players)
                #                 options = {f"{chr(65+i)}": p for i, p in enumerate(all_players)}
                #                 options_str = ", ".join([f"{k}: {v}" for k, v in options.items()])

                question = f"QUESTION: Who would {other_player} vote for?\nOptions: {options_str}\nAnswer:"
                #                 print(question)
                prompt += question

                outputs = model(**inputs)
                logits = outputs.logits[0, -1]
                pred_option_logits = options_token_ids[
                    logits[options_token_ids].argmax()
                ]
                others_vote[player["name"]][other_player] = options[
                    tokenizer.decode(pred_option_logits)
                ]

                for k, v in options.items():
                    if v == player["name"]:
                        options[k] = other_player

        return own_vote, others_vote


def load_model_and_tokenizer(model_name, precision, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        token="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",
    )
    tokenizer.pad_token = tokenizer.eos_token

    if precision == "fp32":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token="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",
        ).to(device)
    elif precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            token="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",
        ).to(device)
    elif precision == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            token="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",
        )
    elif precision == "int4":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16,
            token="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",
        )

    return model, tokenizer


def create_exps(data):
    prev_sent_idx = 0
    examples = []
    example = []

    for sentence in data:
        sent_idx = int(sentence.split(" ")[0])
        sentence = sentence[2:]

        if sent_idx > prev_sent_idx:
            example.append(sentence)
        else:
            context = "".join(example[:-1]).strip()
            question = example[-1].split("\t")[0].strip()
            answer = example[-1].split("\t")[1].strip()
            examples.append(
                {
                    "input": f"Context: {context}\nQuestion: {question}\nAnswer:",
                    "target": " " + answer,
                }
            )
            example = [sentence]

        prev_sent_idx = sent_idx

    return examples


def create_primings(data, num_exps, category):
    priming_exps = ""
    idx = 0
    while idx < num_exps:
        example = random.choice(data)
        if example["category"] == category:
            priming_exps += f"{example['input']}{example['target']}\n\n"
            idx += 1

    return priming_exps


def prepare_data(data, traces, n_priming_eps=3, priming_dist=None):
    processed_data = []

    for example, trace in zip(data, traces):
        trace = trace.split(",")
        category = trace[-2]
        question_type = trace[-1][:-1]

        if "first_order" in category:
            if "no_tom" in category and "true_belief" == question_type:
                category = "first_order_true_belief"
            elif "tom" in category and "false_belief" == question_type:
                category = "first_order_false_belief"
            else:
                continue

        elif "second_order" in category:
            if "no_tom" in category and "true_belief" == question_type:
                category = "second_order_true_belief"
            elif "tom" in category and "false_belief" == question_type:
                category = "second_order_false_belief"
            else:
                continue

        if priming_dist:
            priming_exps = create_primings(
                priming_dist,
                n_priming_eps,
                category,
            )
        else:
            priming_exps = ""

        processed_data.append(
            {
                "input": f'{priming_exps}{example["input"]}',
                "target": example["target"],
                "category": category,
            }
        )

    return processed_data


class Collator(object):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def __call__(self, examples):
        inputs = self.tokenizer(
            [ex["input"] for ex in examples],
            return_tensors="pt",
            padding=True,
        )

        if (
            (
                self.config.architectures[0] == "LlamaForCausalLM"
                and "Llama-3" not in self.config._name_or_path
            )
            or self.config.architectures[0] == "LlaMAForCausalLM"
            or self.config.architectures[0] == "GemmaForCausalLM"
            or self.config.architectures[0] == "MistralForCausalLM"
            or self.config.architectures[0] == "MixtralForCausalLM"
        ):
            inputs["target"] = torch.tensor(
                [self.tokenizer.encode(ex["target"])[2] for ex in examples]
            )

        elif "Llama-3" in self.config._name_or_path:
            inputs["target"] = torch.tensor(
                [self.tokenizer.encode(ex["target"])[1] for ex in examples]
            )

        else:
            inputs["target"] = torch.tensor(
                [self.tokenizer.encode(ex["target"])[0] for ex in examples]
            )

        inputs["category"] = [ex["category"] for ex in examples]
        return inputs


def load_tomi_data(config, tokenizer, current_dir, batch_size, n_priming_eps=3):
    data_path = "data/SymbolicToM Datasets/Fixed and Unambiguous ToMi/"
    path = f"{current_dir}/{data_path}"

    with open(f"{path}/train.txt", "r") as f:
        train_data = f.readlines()
    with open(f"{path}/train.trace", "r") as f:
        train_trace = f.readlines()

    with open(f"{path}/test.txt", "r") as f:
        test_data = f.readlines()
    with open(f"{path}/test.trace", "r") as f:
        test_trace = f.readlines()

    train_data = create_exps(train_data)
    processed_train_data = prepare_data(train_data, train_trace)

    test_data = create_exps(test_data)
    processed_data = prepare_data(
        test_data, test_trace, n_priming_eps, processed_train_data
    )
    print("Total dataset size: ", len(processed_data))

    dataset = Dataset.from_list(processed_data).with_format("torch")
    collator = Collator(config, tokenizer)
    dataloader = DataLoader(
        dataset, collate_fn=collator, batch_size=batch_size, shuffle=False
    )

    return dataloader
