import os
import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
from transformers import StoppingCriteria
from datasets import Dataset


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
        inputs["target"] = [
            tokenizer.decode(tokenizer.encode(" " + ex["target"])[0]) for ex in examples
        ]
    elif (
        model.config.architectures[0] == "LlamaForCausalLM"
        or model.config.architectures[0] == "LlaMAForCausalLM"
        or model.config.architectures[0] == "GemmaForCausalLM"
    ):
        inputs["target"] = [
            tokenizer.decode(tokenizer.encode(ex["target"])[1]) for ex in examples
        ]
    elif (
        model.config.architectures[0] == "MistralForCausalLM"
        or model.config.architectures[0] == "MixtralForCausalLM"
    ):
        inputs["target"] = [
            tokenizer.decode(tokenizer.encode(" " + ex["target"])[2]) for ex in examples
        ]
    else:
        raise NotImplementedError
    return inputs


def load_model_tokenzier(model_name: str, precision: str, device: torch.device):
    if precision == "fp32":
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    elif precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
    elif precision == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
        )
    elif precision == "int4":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


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
