import math
import os
import csv
import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
from transformers import StoppingCriteria
from datasets import Dataset
from torch.utils.data import DataLoader
from einops import einsum
from src.dataset import SampleV3, DatasetV3, SampleV2, DatasetV2

random.seed(10)


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


def get_story(data):
    story = []
    stories = []
    prev_sent_idx = 0

    for sentence in data:
        sent_idx = int(sentence.split(" ")[0])
        sentence = sentence[2:]

        if sent_idx > prev_sent_idx:
            story.append(sentence)
        else:
            context = "".join(story[:-1]).strip()
            question = story[-1].split("\t")[0].strip()
            answer = story[-1].split("\t")[1].strip()
            stories.append(
                {
                    "input": f"{context}\n",
                    "question": question,
                    "target": " " + answer,
                }
            )
            story = [sentence]

        prev_sent_idx = sent_idx

    return stories


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


def create_combined_primings(data):
    priming_exps = ""
    categories = [
        "first_order_true_belief",
        "first_order_false_belief",
        "second_order_true_belief",
        "second_order_false_belief",
    ]

    for category in categories:
        priming_exps += create_primings(data, 1, category)

    return priming_exps


def create_order_based_primings(data):
    priming_exps = {
        "first_order": "",
        "second_order": "",
    }

    for key in priming_exps.keys():
        priming_exps[key] += create_primings(data, 2, f"{key}_true_belief")
        priming_exps[key] += create_primings(data, 2, f"{key}_false_belief")

    return priming_exps


def prepare_data(data, traces):
    processed_data = []

    for example, trace in zip(data, traces):
        trace = trace.split(",")
        category = trace[-1].strip()

        processed_data.append(
            {
                "input": f'{example["input"]}',
                "target": example["target"],
                "category": category,
            }
        )

    return processed_data


class BigTomCollator(object):

    def __init__(self, config, tokenizer, method_name):
        self.config = config
        self.tokenizer = tokenizer
        self.method_name = method_name

        with open(f"prompt_instructions/{self.method_name}.txt", "r") as f:
            self.instructions = f.read()

    def __call__(self, examples):
        prompts, targets = [], []
        for example in examples:
            story, question, correct_answer, wrong_answer = example
            answers = [correct_answer, wrong_answer]
            random.shuffle(answers)

            question = f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"

            prompt = f"Instructions: {self.instructions}\nStory: {story}\nQuestion: {question}\nAnswer:"
            prompts.append(prompt)

            if answers[0] == correct_answer:
                targets.append("a")
            else:
                targets.append("b")

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        inputs["target"] = torch.tensor([self.tokenizer.encode(t)[1] for t in targets])

        return inputs


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
    path = "/home/local_nikhil/Projects/ToMi/data"

    with open(f"{path}/train.txt", "r") as f:
        test_data = f.readlines()
    with open(f"{path}/train.trace", "r") as f:
        test_trace = f.readlines()

    test_data = create_exps(test_data)
    processed_data = prepare_data(test_data, test_trace)

    # priming_exps = create_combined_primings(processed_data)
    # for example in processed_data:
    #     example["input"] = f"{priming_exps}{example['input']}"

    priming_exps = create_order_based_primings(processed_data)
    for example in processed_data:
        if "first_order" in example["category"]:
            example["input"] = f"{priming_exps['first_order']}{example['input']}"
        elif "second_order" in example["category"]:
            example["input"] = f"{priming_exps['second_order']}{example['input']}"

    for example in processed_data:
        instructions = "Instructions: You are doing Sally-Anne test. Keep track of people's knowledge and object location defined in the context. People's knowledge will get updated only when an action is taken in their presence. Use this information to answer the question in a single word.\n"
        example["input"] = f"{instructions}{example['input']}"

    print("Total dataset size: ", len(processed_data))

    dataset = Dataset.from_list(processed_data).with_format("torch")
    collator = Collator(config, tokenizer)
    dataloader = DataLoader(
        dataset, collate_fn=collator, batch_size=batch_size, shuffle=False
    )

    return dataloader


def add_paraphrased_priming_exps(data):
    categories = [
        "first_order_true_belief",
        "first_order_false_belief",
        "second_order_true_belief",
        "second_order_false_belief",
    ]

    priming_exps = {}
    for category in categories:
        with open(f"priming_examples/paraphrases/{category}.json", "r") as f:
            examples = json.load(f)

        priming_exps[category] = ""
        for example in examples:
            priming_exps[category] += f"{example['input']}{example['target']}\n\n"

    for example in data:
        category = example["category"]
        example["input"] = f"{priming_exps[category]}{example['input']}"
        print(example["input"])

    return data


def load_paraphrase_tomi(config, tokenizer, current_dir, batch_size):
    path = f"{current_dir}/data/paraphrased_ToMi/dataset.json"

    with open(path, "r") as f:
        data = json.load(f)
    data = add_paraphrased_priming_exps(data)
    collator = Collator(config, tokenizer)
    dataloader = DataLoader(
        data, collate_fn=collator, batch_size=batch_size, shuffle=False
    )

    return dataloader


def load_bigtom(
    config, tokenizer, current_dir, batch_size, method_name, variable, condition
):
    path = f"{current_dir}/data/bigtom/{variable}_{condition}/stories.csv"

    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        data = list(reader)

    collator = BigTomCollator(config, tokenizer, method_name)
    dataloader = DataLoader(
        data, collate_fn=collator, batch_size=batch_size, shuffle=False
    )

    return dataloader


def get_both_stories(tb_samples, fb_samples, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    for idx in range(n_samples):
        tb_story, tb_question, tb_correct_answer, tb_wrong_answer = tb_samples[idx]
        fb_story, fb_question, fb_correct_answer, fb_wrong_answer = fb_samples[idx]
        answers = [tb_correct_answer, tb_wrong_answer]
        random.shuffle(answers)

        clean_question = f"{tb_question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"

        if answers[0] == tb_correct_answer:
            clean_target = " a"
            corrupt_target = " b"
            corrupt_question = f"{fb_question}\nChoose one of the following:\na){fb_wrong_answer}\nb){fb_correct_answer}"
        else:
            clean_target = " b"
            corrupt_target = " a"
            corrupt_question = f"{fb_question}\nChoose one of the following:\na){fb_correct_answer}\nb){fb_wrong_answer}"

        clean_prompt = f"Instructions: {instructions}\nStory: {tb_story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {fb_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "clean_target": clean_target,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
            }
        )

    return samples


def get_altered_option_letters_data(data, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    for idx in range(n_samples):
        story, question, correct_answer, wrong_answer = data[idx]
        answers = [correct_answer, wrong_answer]
        random.shuffle(answers)

        clean_question = (
            f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        )
        corrupt_question = (
            f"{question}\nChoose one of the following:\na){answers[1]}\nb){answers[0]}"
        )

        if answers[0] == correct_answer:
            clean_target = " a"
            corrupt_target = " b"
        else:
            clean_target = " b"
            corrupt_target = " a"

        clean_prompt = f"Instructions: {instructions}\nStory: {story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "clean_target": clean_target,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
            }
        )

    return samples


def get_tb_fb_data(tb_data, fb_data, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    for idx in range(n_samples):
        tb_story, tb_question, tb_correct_answer, tb_wrong_answer = tb_data[idx]
        fb_story, fb_question, fb_correct_answer, fb_wrong_answer = fb_data[idx]
        answers = [tb_correct_answer, tb_wrong_answer]
        random.shuffle(answers)

        clean_question = f"{tb_question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"

        if answers[0] == tb_correct_answer:
            clean_target = " a"
            corrupt_target = " b"
            corrupt_question = f"{fb_question}\nChoose one of the following:\na){fb_wrong_answer}\nb){fb_correct_answer}"
        else:
            clean_target = " b"
            corrupt_target = " a"
            corrupt_question = f"{fb_question}\nChoose one of the following:\na){fb_correct_answer}\nb){fb_wrong_answer}"

        clean_prompt = f"Instructions: {instructions}\nStory: {tb_story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {fb_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "clean_target": clean_target,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
            }
        )

    return samples


def get_options_reversed_data(data, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    for idx in range(n_samples):
        story, question, correct_answer, wrong_answer = data[idx]
        answers = [correct_answer, wrong_answer]
        random.shuffle(answers)

        clean_question = (
            f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        )
        corrupt_question = (
            f"{question}\nChoose one of the following:\na){answers[1]}\nb){answers[0]}"
        )

        if answers[0] == correct_answer:
            clean_target = " a"
            corrupt_target = " b"
        else:
            clean_target = " b"
            corrupt_target = " a"

        clean_prompt = f"Instructions: {instructions}\nStory: {story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "clean_target": clean_target,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
            }
        )

    return samples


def get_option_pairs(data, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    for idx in range(n_samples):
        story, question, correct_answer, wrong_answer = data[idx]
        answers = [correct_answer, wrong_answer]
        random.shuffle(answers)

        random_idx = random.randint(0, len(data) - 1)
        while random_idx == idx:
            random_idx = random.randint(0, len(data) - 1)
        (
            random_story,
            random_question,
            random_correct_answer,
            random_wrong_answer,
        ) = data[random_idx]

        clean_question = f"{question}\nChoose one of the following:\na){random_correct_answer}\nb){random_wrong_answer}"
        if answers[0] == correct_answer:
            target = " a"
            corrupt_question = f"{random_question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        else:
            target = " b"
            corrupt_question = f"{random_question}\nChoose one of the following:\na){answers[1]}\nb){answers[0]}"

        clean_prompt = f"Instructions: {instructions}\nStory: {story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {random_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "target": target,
            }
        )

    return samples


def get_agent_perspective_pairs(tb_data, fb_data, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    for idx in range(n_samples):
        story, question, correct_answer, wrong_answer = tb_data[idx]
        (
            fb_story,
            fb_question,
            fb_correct_answer,
            fb_wrong_answer,
        ) = fb_data[idx]
        answers = [correct_answer, wrong_answer]
        random.shuffle(answers)

        clean_question = (
            f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        )

        random_idx = random.randint(0, len(tb_data) - 1)
        while random_idx == idx:
            random_idx = random.randint(0, len(tb_data) - 1)
        (
            random_story,
            random_question,
            random_correct_answer,
            random_wrong_answer,
        ) = tb_data[random_idx]

        if answers[0] == correct_answer:
            clean_target = " a"
            corrupt_target = " b"
            corrupt_question = f"{fb_question}\nChoose one of the following:\na){correct_answer}\nb){wrong_answer}"
        else:
            clean_target = " b"
            corrupt_target = " a"
            corrupt_question = f"{fb_question}\nChoose one of the following:\na){wrong_answer}\nb){correct_answer}"

        clean_prompt = f"Instructions: {instructions}\nStory: {random_story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {fb_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "target": corrupt_target,
            }
        )

    return samples


def get_diff_exps(clean_data, corrupt_data, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    random.shuffle(clean_data)
    for idx in range(n_samples):
        story, question, correct_answer, wrong_answer = clean_data[idx]
        answers = [correct_answer, wrong_answer]
        random.shuffle(answers)
        clean_question = (
            f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        )

        random_idx = random.randint(0, len(corrupt_data) - 1)
        while random_idx == idx:
            random_idx = random.randint(0, len(corrupt_data) - 1)
        (
            control_story,
            control_question,
            control_correct_answer,
            control_wrong_answer,
        ) = corrupt_data[random_idx]

        if answers[0] == correct_answer:
            clean_target = " a"
            corrupt_target = " b"
            corrupt_question = f"{control_question}\nChoose one of the following:\na){control_wrong_answer}\nb){control_correct_answer}"
        else:
            clean_target = " b"
            corrupt_target = " a"
            corrupt_question = f"{control_question}\nChoose one of the following:\na){control_correct_answer}\nb){control_wrong_answer}"

        clean_prompt = f"Instructions: {instructions}\nStory: {story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {control_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "clean_target": clean_target,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
            }
        )

    return samples


def get_data_pp(model, clean_data, corrupt_data, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    random.shuffle(clean_data)
    while len(samples) < n_samples:
        idx = random.randint(0, len(clean_data) - 1)
        story, question, correct_answer, wrong_answer = clean_data[idx]
        answers = [wrong_answer, correct_answer]
        random.shuffle(answers)
        clean_question = (
            f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        )

        random_idx = random.randint(0, len(corrupt_data) - 1)
        while random_idx == idx:
            random_idx = random.randint(0, len(corrupt_data) - 1)
        (
            control_story,
            control_question,
            control_correct_answer,
            control_wrong_answer,
        ) = corrupt_data[random_idx]

        if answers[0] == correct_answer:
            clean_target = " a"
            corrupt_target = " b"
            corrupt_question = f"{control_question}\nChoose one of the following:\na){control_wrong_answer}\nb){control_correct_answer}"
        else:
            clean_target = " b"
            corrupt_target = " a"
            corrupt_question = f"{control_question}\nChoose one of the following:\na){control_correct_answer}\nb){control_wrong_answer}"

        clean_prompt = f"Instructions: {instructions}\nStory: {story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {control_story}\nQuestion: {corrupt_question}\nAnswer:"

        with torch.no_grad():
            with model.trace(clean_prompt, scan=False, validate=False):
                clean_output = model.lm_head.output[0, -1].save()
            with model.trace(corrupt_prompt, scan=False, validate=False):
                corrupt_output = model.lm_head.output[0, -1].save()

            if (
                corrupt_output[model.tokenizer.encode(corrupt_target)[1]]
                - corrupt_output[model.tokenizer.encode(clean_target)[1]]
                > 5
            ):
                samples.append(
                    {
                        "clean_prompt": clean_prompt,
                        "clean_target": clean_target,
                        "corrupt_prompt": corrupt_prompt,
                        "corrupt_target": corrupt_target,
                    }
                )
                print(f"Samples len: {len(samples)}")
            else:
                print(
                    f"{clean_output[model.tokenizer.encode(clean_target)[1]] - clean_output[model.tokenizer.encode(corrupt_target)[1]]}"
                )
                print(
                    f"{corrupt_output[model.tokenizer.encode(corrupt_target)[1]] - corrupt_output[model.tokenizer.encode(clean_target)[1]]}"
                )
                continue

            del clean_output, corrupt_output
            torch.cuda.empty_cache()

    return samples


def get_control_corrupt_data_v2(orgs, controls, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    for idx in range(n_samples):
        org = orgs[idx]
        control = controls[idx]

        org_story, org_question, org_correct_answer, org_wrong_answer = org
        (
            control_story,
            control_question,
            control_correct_answer,
            control_wrong_answer,
        ) = control
        answers = [org_correct_answer, org_wrong_answer]
        random.shuffle(answers)

        clean_question = f"{org_question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"

        if answers[0] == org_correct_answer:
            clean_target = " a"
            corrupt_target = " a"
            corrupt_question = f"{control_question}\nChoose one of the following:\na){control_correct_answer}\nb){control_wrong_answer}"
        else:
            clean_target = " b"
            corrupt_target = " b"
            corrupt_question = f"{control_question}\nChoose one of the following:\na){control_wrong_answer}\nb){control_correct_answer}"

        clean_prompt = f"Instructions: {instructions}\nStory: {org_story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {control_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "clean_target": clean_target,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
            }
        )

    return samples


def get_control_corrupt_data(orgs, controls, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    for idx in range(n_samples):
        org = orgs[idx]
        control = controls[idx]

        org_story, org_question, org_correct_answer, org_wrong_answer = org
        (
            control_story,
            control_question,
            control_correct_answer,
            control_wrong_answer,
        ) = control
        # control_story = ". ".join(control_story.split(". ")[:-2]) + '.'
        answers = [org_wrong_answer, org_correct_answer]
        random.shuffle(answers)

        clean_question = f"{org_question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"

        if answers[0] == org_correct_answer:
            clean_target = " a"
            corrupt_target = " b"
            corrupt_question = f"{control_question}\nChoose one of the following:\na){control_wrong_answer}\nb){control_correct_answer}"
        else:
            clean_target = " b"
            corrupt_target = " a"
            corrupt_question = f"{control_question}\nChoose one of the following:\na){control_correct_answer}\nb){control_wrong_answer}"

        clean_prompt = f"Instructions: {instructions}\nStory: {org_story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {control_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "clean_target": clean_target,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
            }
        )

    return samples


def get_example(data, method_name="0shot"):
    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    story, question, correct_answer, wrong_answer = data
    answers = [correct_answer, wrong_answer]

    exp = f"Instructions: {instructions}\nStory: {story}\nQuestion: {question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}\nAnswer:"

    return exp


def get_event_observation_data(
    clean_data, corrupt_data, n_samples, method_name="0shot"
):
    with open("prompt_instructions/0shot.txt", "r") as f:
        instructions = f.read()

    samples = []

    for idx in range(n_samples):
        clean_story, clean_question, clean_correct_answer, clean_wrong_answer = (
            clean_data[idx]
        )
        (
            corrupt_story,
            corrupt_question,
            corrupt_correct_answer,
            corrupt_wrong_answer,
        ) = corrupt_data[idx]
        if "does not" in clean_story.split(". ")[-1]:
            clean_story = (
                ". ".join(clean_story.split(". ")[:-1])
                + ". "
                + clean_story.split(". ")[-1].split(" ")[0]
                + " does not observe this event occurring."
            )
        else:
            clean_story = (
                ". ".join(clean_story.split(". ")[:-1])
                + ". "
                + clean_story.split(". ")[-1].split(" ")[0]
                + " observes this event occurring."
            )
        
        if "does not" in corrupt_story.split(". ")[-1]:
            corrupt_story = (
                ". ".join(corrupt_story.split(". ")[:-1])
                + ". "
                + corrupt_story.split(". ")[-1].split(" ")[0]
                + " does not observe this event occurring."
            )
        else:
            corrupt_story = (
                ". ".join(corrupt_story.split(". ")[:-1])
                + ". "
                + corrupt_story.split(". ")[-1].split(" ")[0]
                + " observes this event occurring."
            )
        answers = [corrupt_correct_answer, corrupt_wrong_answer]
        random.shuffle(answers)

        corrupt_question = f"{corrupt_question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        clean_question = f"{clean_question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        if answers[0] == corrupt_correct_answer:
            corrupt_target = " a"
            clean_target = " b"
        else:
            corrupt_target = " b"
            clean_target = " a"

        clean_prompt = f"Instructions: {instructions}\nStory: {clean_story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {corrupt_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
                "clean_target": clean_target,
            }
        )

    return samples


def get_event_type_data(clean_data, corrupt_data, n_samples, method_name="0shot"):
    with open("prompt_instructions/0shot.txt", "r") as f:
        instructions = f.read()

    samples = []

    for idx in range(n_samples):
        clean_story, clean_question, clean_correct_answer, clean_wrong_answer = (
            clean_data[idx]
        )
        (
            corrupt_story,
            corrupt_question,
            corrupt_correct_answer,
            corrupt_wrong_answer,
        ) = corrupt_data[idx]
        
        if "does not" in clean_story.split(". ")[-1]:
            clean_story = (
                ". ".join(clean_story.split(". ")[:-1])
                + ". "
                + clean_story.split(". ")[-1].split(" ")[0]
                + " does not observe this event occurring."
            )
        else:
            clean_story = (
                ". ".join(clean_story.split(". ")[:-1])
                + ". "
                + clean_story.split(". ")[-1].split(" ")[0]
                + " observes this event occurring."
            )
        
        if "does not" in corrupt_story.split(". ")[-1]:
            corrupt_story = (
                ". ".join(corrupt_story.split(". ")[:-1])
                + ". "
                + corrupt_story.split(". ")[-1].split(" ")[0]
                + " does not observe this event occurring."
            )
        else:
            corrupt_story = (
                ". ".join(corrupt_story.split(". ")[:-1])
                + ". "
                + corrupt_story.split(". ")[-1].split(" ")[0]
                + " observes this event occurring."
            )
        
        answers = [clean_correct_answer, clean_wrong_answer]
        random.shuffle(answers)

        clean_question = f"{clean_question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        if answers[0] == clean_correct_answer:
            clean_target = " a"
            corrupt_target = " b"
            corrupt_question = f"{corrupt_question}\nChoose one of the following:\na){corrupt_wrong_answer}\nb){corrupt_correct_answer}"
        else:
            clean_target = " b"
            corrupt_target = " a"
            corrupt_question = f"{corrupt_question}\nChoose one of the following:\na){corrupt_correct_answer}\nb){corrupt_wrong_answer}"
        
        clean_prompt = f"Instructions: {instructions}\nStory: {clean_story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {corrupt_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
                "clean_target": clean_target,
            }
        )

    return samples


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_causal_mask(attn_scores):
    ignore = torch.tensor(torch.finfo(attn_scores.dtype).min)
    mask = torch.triu(
        torch.ones(
            attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device
        ),
        diagonal=1,
    ).bool()
    attn_scores.masked_fill_(mask, ignore)

    return attn_scores


def get_attn_score(model, prompt, layer_idx):
    n_rep = 8
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    input_tokens = model.tokenizer(prompt, return_tensors="pt").input_ids
    bsz, q_len = input_tokens.size()
    positions = torch.arange(q_len)
    positions = torch.tensor(positions).unsqueeze(0).repeat(bsz, 1).to(model.device)
    scaled_attn = torch.zeros(bsz, n_heads, q_len, q_len)

    with torch.no_grad():
        with model.trace(prompt, scan=False, validate=False) as tracer:
            query_states = model.model.layers[layer_idx].self_attn.q_proj.output
            key_states = model.model.layers[layer_idx].self_attn.k_proj.output
            value_states = model.model.layers[layer_idx].self_attn.v_proj.output

            query_states = query_states.view(
                bsz, q_len, n_heads, head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, n_heads // n_rep, head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, n_heads // n_rep, head_dim
            ).transpose(1, 2)

            X = model.model.layers[layer_idx].self_attn.rotary_emb(
                value_states, positions
            )
            cos, sin = X[0], X[1]
            X = tracer.apply(
                apply_rotary_pos_emb,
                q=query_states,
                k=key_states,
                cos=cos,
                sin=sin,
                validate=False,
            )
            query_states, key_states = X[0], X[1]

            key_states = tracer.apply(
                repeat_kv, key_states, n_rep, validate=False
            )
            value_states = (
                tracer.apply(repeat_kv, value_states, n_rep, validate=False)
                .transpose(1, 2)
                .save()
            )

            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
            attn_weights = tracer.apply(
                apply_causal_mask,
                attn_scores=attn_weights,
                validate=False,
            )

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1).to(
                query_states.dtype
            )
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=model.config.attention_dropout, training=False
            ).save()

            # value_vectors_norm = torch.norm(value_states, dim=-1)
            # scaled_attn = einsum(
            #     value_vectors_norm,
            #     attn_weights,
            #     "batch k_seq_len n_heads, batch n_heads q_seq_len k_seq_len -> batch n_heads q_seq_len k_seq_len",
            # ).save()

            del query_states, key_states, value_states, X, cos, sin
            torch.cuda.empty_cache()

    return attn_weights


def get_imp_indices(input_tokens):
    period_token_indices = [i for i, x in enumerate(input_tokens[0]) if input_tokens[0][i] == 627]
    instruction_end_index = [i for i, x in enumerate(input_tokens[0]) if input_tokens[0][i] == 382][0]
    story_period_indices = [i for i, x in enumerate(input_tokens[0]) if input_tokens[0][i] == 13 and i > instruction_end_index and i < period_token_indices[0]]
    option_a = [i for i, x in enumerate(input_tokens[0]) if input_tokens[0][i] == 64][-1]
    option_b = [i for i, x in enumerate(input_tokens[0]) if input_tokens[0][i] == 65][-1]
    question_token = [i for i, x in enumerate(input_tokens[0]) if input_tokens[0][i] == 5380][-1]
    # or_token = [i for i, x in enumerate(input_tokens[0]) if input_tokens[0][i] == 477][-1]
    colon_token = [i for i, x in enumerate(input_tokens[0]) if input_tokens[0][i] == 25]

    return period_token_indices, instruction_end_index, story_period_indices, option_a, option_b, question_token, colon_token


def get_plain_exps(data, n_samples):
    with open("prompt_instructions/0shot.txt", "r") as f:
        instructions = f.read()
    
    samples = []
    for idx in range(n_samples):
        story, question, correct_answer, wrong_answer = data[idx]
        answers = [correct_answer, wrong_answer]
        random.shuffle(answers)

        random_idx = random.randint(0, len(data) - 1)
        while random_idx == idx:
            random_idx = random.randint(0, len(data) - 1)
        
        corrupt_story, corrupt_question, corrupt_correct_answer, corrupt_wrong_answer = data[random_idx]

        story = ". ".join(story.split(". ")[:-2]) + "."
        corrupt_story = ". ".join(corrupt_story.split(". ")[:-2]) + "."

        clean_question = f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        if answers[0] == correct_answer:
            clean_target = " a"
            corrupt_target = " b"
            corrupt_question = f"{corrupt_question}\nChoose one of the following:\na){corrupt_wrong_answer}\nb){corrupt_correct_answer}"
        else:
            clean_target = " b"
            corrupt_target = " a"
            corrupt_question = f"{corrupt_question}\nChoose one of the following:\na){corrupt_correct_answer}\nb){corrupt_wrong_answer}"
        
        clean_prompt = f"Instructions: {instructions}\nStory: {story}\nQuestion: {clean_question}\nAnswer:"
        corrupt_prompt = f"Instructions: {instructions}\nStory: {corrupt_story}\nQuestion: {corrupt_question}\nAnswer:"

        samples.append(
            {
                "clean_prompt": clean_prompt,
                "clean_target": clean_target,
                "corrupt_prompt": corrupt_prompt,
                "corrupt_target": corrupt_target,
            }
        )
    
    return samples


def get_diff_name(data, n_samples, model):
    instruction = '''Instructions: Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose \"yes\" or \"no" after the "Answer:" tag.'''
    samples = []

    for idx in range(n_samples):
        story, _, correct_answer, wrong_answer = data[idx]
        # story = ". ".join(story.split(". ")[:-2]) + "."
        agent_name = " " + story.split(' ', maxsplit=1)[0]

        if len(model.tokenizer.encode(agent_name)) - 1 == 1:
            corrupt_agent_name = " Sally"
        elif len(model.tokenizer.encode(agent_name)) - 1 == 2:
            corrupt_agent_name = " Nikhil"
        elif len(model.tokenizer.encode(agent_name)) - 1 == 3:
            corrupt_agent_name = " Koyena"
        else:
            raise ValueError("Agent name not found")

        story = story.replace("He", agent_name.strip())
        story = story.replace("She", agent_name.strip())
        corrupt_story = story.replace(agent_name.strip(), corrupt_agent_name.strip())
        question = f'Does {correct_answer.replace("believes", "believe").replace(".", "?")}'
        question = question.replace("?", " according to the story?")
        # corrupt_question = question.replace(agent_name.strip(), corrupt_agent_name.strip())

        correct_prompt = f'{instruction}\n\nStory: {story}\nQuestion: {question}\nAnswer:'
        wrong_prompt = f'{instruction}\n\nStory: {corrupt_story}\nQuestion: {question}\nAnswer:'

        samples.append({
            'corrupt_prompt': correct_prompt,
            'clean_prompt': wrong_prompt,
            "agent_name_len": len(model.tokenizer.encode(agent_name)) - 1,
        })

    return samples


def get_subject_exps_worldstate(clean, corrupt, n_samples):
    instruction = '''Instructions: Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose \"yes\" or \"no" after the "Answer:" tag.'''
    samples = []

    for idx in range(n_samples):
        clean_story, _, clean_correct_answer, clean_wrong_answer = clean[idx]
        corrupt_story, _, corrupt_correct_answer, corrupt_wrong_answer = corrupt[idx]

        if "is" in clean_correct_answer:
            clean_question = f"Is {clean_correct_answer.replace('The', 'the').replace('is ', '').replace('.', '?')}"
        elif "are" in clean_correct_answer:
            clean_question = f"Are {clean_correct_answer.replace('The', 'the').replace('are ', '').replace('.', '?')}"
        else:
            clean_question = f"Does {clean_correct_answer.replace('The', 'the').replace('contains', 'contain').replace('.', '?')}"
        clean_question = clean_question.replace("?\n", "?")

        if "is" in corrupt_correct_answer:
            corrupt_question = f"Is {corrupt_correct_answer.replace('The', 'the').replace('is ', '').replace('.', '?')}"
        elif "are" in corrupt_correct_answer:
            corrupt_question = f"Are {corrupt_correct_answer.replace('The', 'the').replace('are ', '').replace('.', '?')}"
        else:
            corrupt_question = f"Does {corrupt_correct_answer.replace('The', 'the').replace('contains', 'contain').replace('.', '?')}"
        wrong_question = corrupt_question.replace("?\n", "?")

        clean_prompt = f'{instruction}\n\nStory: {clean_story}\nQuestion: {clean_question}\nAnswer:'
        corrupt_prompt = f'{instruction}\n\nStory: {corrupt_story}\nQuestion: {wrong_question}\nAnswer:'
        samples.append({
            'correct_prompt': clean_prompt,
            'wrong_prompt': corrupt_prompt,
        })
    
    return samples


def get_yes_no_exps(data, n_samples):
    instruction = '''Instructions: Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose \"yes\" or \"no" after the "Answer:" tag.'''
    samples = []

    for idx in range(n_samples):
        random_idx = idx
        while random_idx == idx:
            random_idx = random.randint(0, len(data) - 1)
        story, _, correct_answer, wrong_answer = data[idx]

        correct_question = f'Does {correct_answer.replace("believes", "believe").replace(".", "?")}'
        wrong_question = f'Does {wrong_answer.replace("believes", "believe").replace(".", "?")}'

        # correct_question = correct_question.replace("?", " according to the story?")
        # wrong_question = wrong_question.replace("?", " according to the story?")
        correct_prompt = f'{instruction}\n\nStory: {story}\nQuestion: {correct_question}\nAnswer:'
        wrong_prompt = f'{instruction}\n\nStory: {story}\nQuestion: {wrong_question}\nAnswer:'
        samples.append({
            'correct_prompt': correct_prompt,
            'wrong_prompt': wrong_prompt,
        })
    
    return samples


def get_yes_no_simple_exps(data, n_samples):
    instruction = '''Instructions: Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose \"yes\" or \"no" after the "Answer:" tag.'''
    samples = []

    for idx in range(n_samples):
        story, _, correct_answer, wrong_answer = data[idx]
        story = ". ".join(story.split(". ")[:-2]) + "."
        correct_question = f'Does {correct_answer.replace("believes", "believe").replace(".", "?")}'
        wrong_question = f'Does {wrong_answer.replace("believes", "believe").replace(".", "?")}'
        correct_prompt = f'{instruction}\n\nStory: {story}\nQuestion: {correct_question}\nAnswer:'
        wrong_prompt = f'{instruction}\n\nStory: {story}\nQuestion: {wrong_question}\nAnswer:'
        samples.append({
            'correct_prompt': correct_prompt,
            'wrong_prompt': wrong_prompt,
        })
    
    return samples


def get_diff_event_obsr_exps(clean, corrupt, n_samples, model):
    instruction = '''Instructions: Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose \"yes\" or \"no" after the "Answer:" tag.'''
    samples = []

    for idx in range(n_samples):
        clean_story, _, clean_answer, _ = clean[idx]
        corrupt_story, _, _, corrupt_wrong = corrupt[idx]
        clean_question = f'Does {clean_answer.replace("believes", "believe").replace(".", "?")}'
        corrupt_question = f'Does {corrupt_wrong.replace("believes", "believe").replace(".", "?")}'

        clean_question = clean_question.replace("?", " according to the story?")
        corrupt_question = corrupt_question.replace("?", " according to the story?")

        clean_prompt = f'{instruction}\n\nStory: {clean_story}\nQuestion: {clean_question}\nAnswer:'
        corrupt_prompt = f'{instruction}\n\nStory: {corrupt_story}\nQuestion: {corrupt_question}\nAnswer:'
        agent_name = " " + clean_story.split(' ', maxsplit=1)[0]
        samples.append({
            'correct_prompt': clean_prompt,
            'wrong_prompt': corrupt_prompt,
            "agent_name_len": len(model.tokenizer.encode(agent_name)) - 1,
        })
    
    return samples


def align_markers(clean_data, reverse_data, n_samples):
    instruction = '''Instructions: Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose \"yes\" or \"no" after the "Answer:" tag.'''
    samples = []

    for idx in range(n_samples):
        story, _, clean_correct, clean_wrong = clean_data[idx]
        random_idx = idx
        # while random_idx == idx:
        #     random_idx = random.randint(0, len(reverse_data) - 1)
        reverse_story, _, reverse_correct, reverse_wrong = reverse_data[random_idx]
        # story = ". ".join(story.split(". ")[:-2]) + "."
        # reverse_story = ". ".join(reverse_story.split(". ")[:-2]) + "."
        
        if "is" in clean_wrong:
            clean_question = f"Is {clean_wrong.replace('The', 'the').replace('is ', '').replace('believes', 'believe').replace('.', '?')}"
        elif "are" in clean_wrong:
            clean_question = f"Are {clean_wrong.replace('The', 'the').replace('are ', '').replace('believes', 'believe').replace('.', '?')}"
        else:
            clean_question = f"Does {clean_wrong.replace('The', 'the').replace('contains', 'contain').replace('.', '?')}"
        clean_question = clean_question.replace("?\n", "?")

        if "is" in reverse_correct:
            reverse_question = f"Is {reverse_correct.replace('The', 'the').replace('is ', '').replace('believes', 'believe').replace('.', '?')}"
        elif "are" in reverse_correct:
            reverse_question = f"Are {reverse_correct.replace('The', 'the').replace('are ', '').replace('believes', 'believe').replace('.', '?')}"
        else:
            reverse_question = f"Does {reverse_correct.replace('The', 'the').replace('contains', 'contain').replace('.', '?')}"
        reverse_question = reverse_question.replace("?\n", "?")

        clean_prompt = f'{instruction}\n\nStory: {story}\nQuestion: {clean_question}\nAnswer:'
        reverse_prompt = f'{instruction}\n\nStory: {reverse_story}\nQuestion: {reverse_question}\nAnswer:'

        samples.append({
            'clean_prompt': clean_prompt,
            'corrupt_prompt': reverse_prompt,
        })
    
    return samples


def get_long_exps(data, n_samples):
    instruction = '''Instructions: Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose \"yes\" or \"no" after the "Answer:" tag.'''
    samples = []

    for idx in range(n_samples):
        story, _, correct_answer_1, wrong_answer_1 = data[idx]
        random_idx = idx
        while random_idx == idx:
            random_idx = random.randint(0, len(data) - 1)
        random_story, _, random_correct_answer_1, random_wrong_answer_1 = data[random_idx]

        story = story + ' ' + random_story
        for i, option in enumerate([correct_answer_1, random_correct_answer_1, wrong_answer_1, random_wrong_answer_1]):
            if "is" in option:
                question = f"Is {option.replace('The', 'the').replace('is ', '').replace('.', '?')}"
            else:
                question = f"Does {option.replace('The', 'the').replace('.', '?')}"
            question = question.replace("?\n", "?")
            
            prompt = f'{instruction}\n\nStory: {story}\nQuestion: {question}\nAnswer:'
            target = "yes" if i < 2 else "no"
            samples.append({
                'prompt': prompt,
                'target': target,
            })
    
    random.shuffle(samples)
    return samples


def get_world_state_exps(data, n_samples):
    instruction = '''Instructions: Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose \"yes\" or \"no" after the "Answer:" tag.'''
    samples = []

    for idx in range(n_samples):
        story, question, correct_answer, _ = data[idx]
        random_idx = idx
        # while random_idx == idx:
        #     random_idx = random.randint(0, len(data) - 1)
        random_story, _, _, random_wrong_answer = data[random_idx]

        correct_answer = correct_answer.replace("\n", "")
        random_wrong_answer = random_wrong_answer.replace("\n", "")
        if "is" in correct_answer:
            correct_answer = correct_answer.replace("is ", "")
            correct_question = f"Is {correct_answer.replace('The', 'the').replace('.', '?')}"
        else:
            correct_question = f"Does {correct_answer.replace('The', 'the').replace('contains', 'contain').replace('.', '?')}"
        
        if "is" in random_wrong_answer:
            random_wrong_answer = random_wrong_answer.replace("is ", "")
            wrong_question = f"Is {random_wrong_answer.replace('The', 'the').replace('.', '?')}"
        else:
            wrong_question = f"Does {random_wrong_answer.replace('The', 'the').replace('contains', 'contain').replace('.', '?')}"
        
        correct_question = correct_question.replace("?", " according to the story?")
        wrong_question = wrong_question.replace("?", " according to the story?")

        correct_prompt = f'{instruction}\n\nStory: {story}\nQuestion: {correct_question}\nAnswer:'
        wrong_prompt = f'{instruction}\n\nStory: {random_story}\nQuestion: {wrong_question}\nAnswer:'

        samples.append({
            'correct_prompt': correct_prompt,
            'wrong_prompt': wrong_prompt,
        })
    
    return samples


def get_new_template_exps(data, characters, n_samples, event_noticed=False, question_type='true_state'):
    configs, samples = [], []

    for idx in range(n_samples):
        idx = idx % len(data)
        protagonist, perpetrator = random.choices(characters, k=2)
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]
        event_idx = random.choices([0, 1], weights=[0.5, 0.5], k=1)[0]
        
        if event_noticed:
            obsr_event = data[idx]['true_belief']
        else:
            obsr_event = data[idx]['false_belief']

        config = SampleV3(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=containers,
            event_idx=event_idx,
            obsr_event=obsr_event,
            event_noticed=event_noticed
        )
        configs.append(config)

    dataset = DatasetV3(configs)

    for idx in range(n_samples):
        prompt, target = dataset.__getitem__(idx, question_type='belief')
        samples.append({
            'prompt': prompt,
            'target': target
        })
    
    return samples, configs


def get_container_marker_container2(data, characters, n_samples, event_noticed=False, question_type='true_state'):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        idx = idx % len(data)
        protagonist, perpetrator = random.choices(characters, k=2)
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]
        # Select a random state from data[idx]['states_1'] which is not in states
        random_containter = random.choice(data[idx]['containers_1'].split(', '))
        while random_containter in states:
            random_containter = random.choice(data[idx]['containers_1'].split(', '))
        obsr_event = data[idx]['true_belief'] if event_noticed else data[idx]['false_belief']

        story = data[idx]['story']
        story = story.replace("<state2>", "<state1>", 1)
        story = story.replace("<state_event>", "<state1>", 1)
        config = SampleV3(
            story=story,
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=containers,
            event_idx=0,
            obsr_event=obsr_event,
            event_noticed=event_noticed,
            diff_template=True
        )
        clean_configs.append(config)

        # story = story.replace("<container1>", "plate", 1)
        config = SampleV3(
            story=story,
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=list(reversed(containers)),
            event_idx=1,
            obsr_event=obsr_event,
            event_noticed=event_noticed,
            diff_template=True
        )
        corrupt_configs.append(config)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for i in range(n_samples):
        corrupt_prompt, corrupt_target = corrupt_dataset.__getitem__(i, set_ans="no", set_container=1, question_type=question_type)
        clean_prompt, clean_target = clean_dataset.__getitem__(i, set_ans="yes", set_container=1, question_type=question_type)

        samples.append({
            "clean_prompt": clean_prompt,
            "clean_target": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_target": corrupt_target,
            "containers": clean_configs[i].containers,
        })
    
    return samples


def get_container_marker_event(data, characters, n_samples, event_noticed=False, question_type='true_state'):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        idx = idx % len(data)
        protagonist, perpetrator = random.choices(characters, k=2)
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]
        obsr_event = data[idx]['true_belief'] if event_noticed else data[idx]['false_belief']

        story = data[idx]['story']
        story = story.replace("<state2>", "<state1>", 1)
        story = story.replace("<state_event>", "<state1>", 1)
        config = SampleV3(
            story=story,
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=containers,
            event_idx=0,
            obsr_event=obsr_event,
            event_noticed=event_noticed,
            diff_template=True
        )
        clean_configs.append(config)

        config = SampleV3(
            story=story,
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=list(reversed(containers)),
            event_idx=1,
            obsr_event=obsr_event,
            event_noticed=event_noticed,
            diff_template=True
        )
        corrupt_configs.append(config)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for i in range(n_samples):
        corrupt_prompt, corrupt_target = corrupt_dataset.__getitem__(i, set_ans="no", set_container=1, question_type=question_type)
        clean_prompt, clean_target = clean_dataset.__getitem__(i, set_ans="no", set_container=1, question_type=question_type)

        samples.append({
            "clean_prompt": clean_prompt,
            "clean_target": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_target": corrupt_target,
            "states": clean_configs[i].states,
            "containers": clean_configs[i].containers,
        })

    return samples    


def get_subject_marker_pairs(data, characters, n_samples, event_noticed=False, question_type='true_state'):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        idx = idx % len(data)
        protagonist, perpetrator = random.choices(characters, k=2)
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]
        obsr_event = data[idx]['true_belief'] if event_noticed else data[idx]['false_belief']

        sample = SampleV3(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=containers,
            event_idx=1,
            obsr_event=obsr_event,
            event_noticed=event_noticed
        )
        corrupt_configs.append(sample)

        # Replace the first instance of <state2> with <state1> in the story
        story = data[idx]['story']
        story = story.replace("<state2>", "<state1>", 1)
        story = story.replace("<state_event>", "<state1>", 1)

        sample = SampleV3(
            story=story,
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=list(reversed(containers)),
            event_idx=1,
            obsr_event=obsr_event,
            event_noticed=event_noticed,
            diff_template=True
        )
        clean_configs.append(sample)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for i in range(n_samples):
        corrupt_prompt, corrupt_target = corrupt_dataset.__getitem__(i, set_ans="no", set_container=1, question_type=question_type)
        clean_prompt, clean_target = clean_dataset.__getitem__(i, set_ans="no", set_container=0, question_type=question_type)

        samples.append({
            "clean_prompt": clean_prompt,
            "clean_target": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_target": corrupt_target,
            "states": clean_configs[i].states,
            "containers": clean_configs[i].containers,
        })
    
    return samples


def get_container_marker_pairs(data, characters, n_samples, event_noticed=False, question_type='true_state'):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        idx = idx % len(data)
        protagonist, perpetrator = random.choices(characters, k=2)
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]
        obsr_event = data[idx]['true_belief'] if event_noticed else data[idx]['false_belief']

        sample = SampleV3(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=containers,
            event_idx=1,
            obsr_event=obsr_event,
            event_noticed=event_noticed
        )
        corrupt_configs.append(sample)
        sample = SampleV3(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=list(reversed(states)),
            containers=containers,
            event_idx=0,
            obsr_event=obsr_event,
            event_noticed=event_noticed
        )
        clean_configs.append(sample)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for i in range(n_samples):
        corrupt_prompt, corrupt_target = corrupt_dataset.__getitem__(i, set_container=0, set_ans="no", question_type=question_type)
        clean_prompt, clean_target = clean_dataset.__getitem__(i, set_container=0, set_ans="no", question_type=question_type)

        samples.append({
            "clean_prompt": clean_prompt,
            "clean_target": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_target": corrupt_target,
            "states": clean_configs[i].states,
            "containers": clean_configs[i].containers,
        })
    
    return samples


def get_object_marker_pairs(data, characters, n_samples, event_noticed=False, question_type='true_state'):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        idx = idx % len(data)
        protagonist, perpetrator = random.choices(characters, k=2)
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]
        obsr_event = data[idx]['true_belief'] if event_noticed else data[idx]['false_belief']

        sample = SampleV3(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=containers,
            event_idx=1,
            obsr_event=obsr_event,
            event_noticed=event_noticed
        )
        corrupt_configs.append(sample)
        sample = SampleV3(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=list(reversed(states)),
            containers=containers,
            event_idx=0,
            obsr_event=obsr_event,
            event_noticed=event_noticed
        )
        clean_configs.append(sample)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for i in range(n_samples):
        corrupt_prompt, corrupt_target = corrupt_dataset.__getitem__(i, set_container=1, set_ans="no", question_type=question_type)
        clean_prompt, clean_target = clean_dataset.__getitem__(i, set_container=1, set_ans="no", question_type=question_type)

        samples.append({
            "clean_prompt": clean_prompt,
            "clean_target": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_target": corrupt_target
        })
    
    return samples



def get_consistency_pairs(data, characters, n_samples, event_noticed=False, question_type='true_state'):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        idx = idx % len(data)
        protagonist, perpetrator = random.choices(characters, k=2)
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]
        obsr_event = data[idx]['true_belief'] if event_noticed else data[idx]['false_belief']

        config = SampleV3(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=containers,
            event_idx=0,
            obsr_event=obsr_event,
            event_noticed=event_noticed
        )
        clean_configs.append(config)

        config = SampleV3(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=containers,
            event_idx=1,
            obsr_event=obsr_event,
            event_noticed=event_noticed
        )
        corrupt_configs.append(config)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for i in range(n_samples):
        container = 1
        clean_prompt, clean_target = clean_dataset.__getitem__(i, set_ans="no", set_container=container, question_type=question_type)
        corrupt_prompt, corrupt_target = corrupt_dataset.__getitem__(i, set_ans="yes", set_container=container, question_type=question_type)
        samples.append({
            "clean_prompt": clean_prompt,
            "clean_target": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_target": corrupt_target
        })
    
    return samples


def get_initial_worldstate(data, n_samples, characters):
    configs, samples = [], []

    for idx in range(n_samples):
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]

        sample = SampleV2(
            story=data[idx]['story'],
            protagonist=random.choice(characters),
            states=states,
            containers=containers
        )
        configs.append(sample)
    
    dataset = DatasetV2(configs)

    for idx in range(n_samples):
        prompt, target = dataset.__getitem__(idx)
        samples.append({
            'prompt': prompt,
            'target': target
        })
    
    return samples


def get_initial_worldstate_consistency(data, characters, n_samples):
    configs, samples = [], []

    for idx in range(n_samples):
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]

        sample = SampleV2(
            story=data[idx]['story'],
            protagonist=random.choice(characters),
            states=states,
            containers=containers
        )
        configs.append(sample)
    
    dataset = DatasetV2(configs)

    for idx in range(n_samples):
        clean_prompt, clean_target = dataset.__getitem__(idx, set_ans="no")
        corrupt_prompt, corrupt_target = dataset.__getitem__(idx, set_ans="yes")
        samples.append({
            "clean_prompt": clean_prompt,
            "clean_target": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_target": corrupt_target
        })
    
    return samples


def get_initial_worldstate_obj_marker(data, characters, n_samples):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]

        sample = SampleV2(
            story=data[idx]['story'],
            protagonist=random.choice(characters),
            states=states,
            containers=containers
        )
        clean_configs.append(sample)

        sample = SampleV2(
            story=data[idx]['story'],
            protagonist=random.choice(characters),
            states=list(reversed(states)),
            containers=containers
        )
        corrupt_configs.append(sample)
    
    clean_dataset = DatasetV2(clean_configs)
    corrupt_dataset = DatasetV2(corrupt_configs)

    for i in range(n_samples):
        clean_prompt, clean_target = clean_dataset.__getitem__(i, set_ans="no", set_container=1)
        corrupt_prompt, corrupt_target = corrupt_dataset.__getitem__(i, set_ans="no", set_container=0)
        samples.append({
            "clean_prompt": clean_prompt,
            "clean_target": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_target": corrupt_target
        })
    
    return samples


def get_initial_worldstate_obj_marker_2(data, characters, n_samples):
    clean_configs, corrupt_configs = [], []
    samples, random_containers = [], []

    for idx in range(n_samples):
        states = [random.choice(data[idx]['states_1'].split(', ')), random.choice(data[idx]['states_2'].split(', '))]
        containers = [random.choice(data[idx]['containers_1'].split(', ')), random.choice(data[idx]['containers_2'].split(', '))]
        protagonist, perpetrator = random.sample(characters, k=2)

        sample = SampleV2(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=containers
        )
        clean_configs.append(sample)
        
        random_container_1 = random.choice(data[idx]['containers_1'].split(', '))
        random_container_2 = random.choice(data[idx]['containers_2'].split(', '))
        while random_container_1 in containers or random_container_1 == random_container_2:
            random_container_1 = random.choice(data[idx]['containers_1'].split(', '))
        
        while random_container_2 in containers or random_container_2 == random_container_1:
            random_container_2 = random.choice(data[idx]['containers_2'].split(', '))
        
        random_containers.append([random_container_1, random_container_2])

        # story = data[idx]['story']
        # story = story.replace("<state1>", random_state_1)
        # story = story.replace("<state2>", random_state_2)
        sample = SampleV2(
            story=data[idx]['story'],
            protagonist=protagonist,
            perpetrator=perpetrator,
            states=states,
            containers=list(reversed(containers))
        )
        corrupt_configs.append(sample)
    
    clean_dataset = DatasetV2(clean_configs)
    corrupt_dataset = DatasetV2(corrupt_configs)

    for i in range(n_samples):
        clean_prompt, clean_target = clean_dataset.__getitem__(i, set_ans="no", set_container=1)
        corrupt_prompt, corrupt_target = corrupt_dataset.__getitem__(i, set_ans="yes", set_container=0)
        samples.append({
            "clean_prompt": clean_prompt,
            "clean_target": clean_target,
            "corrupt_prompt": corrupt_prompt,
            "corrupt_target": corrupt_target,
            "states": clean_configs[i].states,
            "clean_containers": clean_configs[i].containers,
            "corrupt_containers": clean_configs[i].containers,
        })
    
    return samples