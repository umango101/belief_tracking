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
            f"{question}\nChoose one of the following:\nx){answers[1]}\ny){answers[0]}"
        )

        if answers[0] == correct_answer:
            clean_target = " a"
            corrupt_target = " y"
        else:
            clean_target = " b"
            corrupt_target = " x"

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


def get_data_pp(data, n_samples, method_name="0shot"):
    samples = []

    with open(f"prompt_instructions/{method_name}.txt", "r") as f:
        instructions = f.read()

    random.shuffle(data)
    for idx in range(n_samples):
        story, question, correct_answer, wrong_answer = data[idx]
        answers = [correct_answer, wrong_answer]
        random.shuffle(answers)
        clean_question = (
            f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        )

        random_idx = random.randint(0, len(data) - 1)
        while random_idx == idx:
            random_idx = random.randint(0, len(data) - 1)
        (
            control_story,
            control_question,
            control_correct_answer,
            control_wrong_answer,
        ) = data[random_idx]

        if answers[0] == correct_answer:
            clean_target = " a"
            corrupt_target = " y"
            corrupt_question = f"{control_question}\nChoose one of the following:\nx){control_wrong_answer}\ny){control_correct_answer}"
        else:
            clean_target = " b"
            corrupt_target = " x"
            corrupt_question = f"{control_question}\nChoose one of the following:\nx){control_correct_answer}\ny){control_wrong_answer}"

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
        answers = [org_correct_answer, org_wrong_answer]
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
