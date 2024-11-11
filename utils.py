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


def get_value_fetcher_exps(STORY_TEMPLATES, 
                           all_characters, 
                           all_containers, 
                           all_states,
                           n_samples,
                           question_type='state_question'):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES['templates'])
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = SampleV3(
            template=template,
            characters=characters,
            containers=containers,
            states=states,
            event_idx=None,
            event_noticed=False,
        )
        clean_configs.append(sample)

        template = random.choice(STORY_TEMPLATES['templates'])
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)
        sample = SampleV3(
            template=template,
            characters=characters,
            containers=containers,
            states=states,
            event_idx=None,
            event_noticed=False,
        )
        corrupt_configs.append(sample)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        if question_type == "belief_question":
            set_character = random.choice([0, 1])
            clean = clean_dataset.__getitem__(idx, question_type=question_type, set_character=set_character, set_container=set_character)
            corrupt = corrupt_dataset.__getitem__(idx, question_type=question_type, set_character=1 ^ set_character, set_container=1^set_character)
        else:
            clean = clean_dataset.__getitem__(idx, question_type=question_type)
            corrupt = corrupt_dataset.__getitem__(idx, question_type=question_type)

        samples.append({
            "clean_prompt": clean['prompt'],
            "clean_target": clean['target'],
            "corrupt_prompt": corrupt['prompt'],
            "corrupt_target": corrupt['target']
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


def get_pos_trans_exps(STORY_TEMPLATES,
                       all_characters,
                       all_containers,
                       all_states,
                       n_samples,
                       question_type='state_question'):
    clean_configs, corrupt_configs, intervention_pos = [], [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES['templates'])
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = SampleV3(
            template=template,
            characters=characters,
            containers=containers,
            states=states,
            event_idx=None,
            event_noticed=False,
        )
        clean_configs.append(sample)

        new_states = random.sample(all_states[template["state_type"]], 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states[template["state_type"]], 2)

        sample = SampleV3(
            template=template,
            characters=characters if question_type == "state_question" else list(reversed(characters)),
            containers=list(reversed(containers)),
            states=new_states,
            event_idx=None,
            event_noticed=False
        )
        corrupt_configs.append(sample)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        set_container = random.choice([0, 1])

        if question_type == "belief_question":
            clean = clean_dataset.__getitem__(idx, set_container=set_container, set_character=set_container, question_type=question_type)
            corrupt = corrupt_dataset.__getitem__(idx, set_container=1 ^ set_container, set_character=1 ^ set_container, question_type=question_type)

        else:
            clean = clean_dataset.__getitem__(idx, set_container=set_container, question_type=question_type)
            corrupt = corrupt_dataset.__getitem__(idx, set_container=1 ^ set_container, question_type=question_type)

        samples.append({
            "clean_prompt": clean['prompt'],
            "clean_ans": clean['target'],
            "corrupt_prompt": corrupt['prompt'],
            "corrupt_ans": corrupt['target'],
            "target": clean_configs[idx].states[1 ^ set_container]
        })
    
    return samples


def get_obj_tracing_exps(STORY_TEMPLATES,
                          all_characters,
                          all_containers,
                          all_states,
                          n_samples):
    clean_configs, corrupt_configs, random_container_indices = [], [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES['templates'])
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = SampleV3(
            template=template,
            characters=characters,
            containers=containers,
            states=states,
            event_idx=None,
            event_noticed=False,
        )
        corrupt_configs.append(sample)

        random_container = random.choice(all_containers[template["container_type"]])
        while random_container in containers:
            random_container = random.choice(all_containers[template["container_type"]])
        random_container_indices.append(random.choice([0, 1]))
        new_containers = containers.copy()
        new_containers[random_container_indices[-1]] = random_container
        new_containers.append(containers[random_container_indices[-1]])

        sample = SampleV3(
            template=template,
            characters=characters,
            containers=new_containers,
            states=states,
            event_idx=None,
            event_noticed=False
        )
        clean_configs.append(sample)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        clean = clean_dataset.__getitem__(idx, set_container=-1)
        corrupt = corrupt_dataset.__getitem__(idx, set_container=random_container_indices[idx])
        samples.append({
            "clean_prompt": clean['prompt'],
            "clean_ans": clean['target'],
            "corrupt_prompt": corrupt['prompt'],
            "corrupt_ans": corrupt['target'],
            "target": clean_configs[idx].states[random_container_indices[idx]]
        })
    
    return samples


def get_state_tracing_exps(STORY_TEMPLATES,
                           all_characters,
                           all_containers,
                           all_states,
                           n_samples):
    clean_configs, corrupt_configs, random_state_indices = [], [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES['templates'])
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = SampleV3(
            template=template,
            characters=characters,
            containers=containers,
            states=states,
            event_idx=None,
            event_noticed=False,
        )
        corrupt_configs.append(sample)

        random_state = random.choice(all_states[template["state_type"]])
        while random_state in states:
            random_state = random.choice(all_states[template["state_type"]])
        random_state_indices.append(random.choice([0, 1]))
        new_states = states.copy()
        new_states[random_state_indices[-1]] = random_state
        new_states.append(states[random_state_indices[-1]])

        sample = SampleV3(
            template=template,
            characters=characters,
            containers=containers,
            states=new_states,
            event_idx=None,
            event_noticed=False
        )
        clean_configs.append(sample)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        clean = clean_dataset.__getitem__(idx, set_container=random_state_indices[idx])
        corrupt = corrupt_dataset.__getitem__(idx, set_container=random_state_indices[idx])
        samples.append({
            "clean_prompt": clean['prompt'],
            "clean_ans": clean['target'],
            "corrupt_prompt": corrupt['prompt'],
            "corrupt_ans": corrupt['target'],
            "target": corrupt_configs[idx].states[random_state_indices[idx]]
        })
    
    return samples


def get_state_pos_exps(STORY_TEMPLATES,
                       all_characters,
                       all_containers,
                       all_states,
                       n_samples,
                       question_type='state_question'):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES['templates'])
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = SampleV3(
            template=template,
            characters=characters,
            containers=containers,
            states=states,
            event_idx=None,
            event_noticed=False,
        )
        clean_configs.append(sample)

        sample = SampleV3(
            template=template,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=list(reversed(states)),
            event_idx=None,
            event_noticed=False
        )
        corrupt_configs.append(sample)

    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        random_container_idx = random.choice([0, 1])

        if question_type == "belief_question":
            clean = clean_dataset.__getitem__(idx, set_container=random_container_idx, set_character=random_container_idx, question_type=question_type)
            corrupt = corrupt_dataset.__getitem__(idx, set_container=random_container_idx, set_character=random_container_idx, question_type=question_type)
        else:
            clean = clean_dataset.__getitem__(idx, set_container=1 ^ random_container_idx, question_type=question_type)
            corrupt = corrupt_dataset.__getitem__(idx, set_container=random_container_idx, question_type=question_type)
        samples.append({
            "clean_prompt": clean['prompt'],
            "clean_ans": clean['target'],
            "corrupt_prompt": corrupt['prompt'],
            "corrupt_ans": corrupt['target'],
            "target": clean_configs[idx].states[1 ^ random_container_idx]
        })

    return samples


def get_character_tracing_exps(STORY_TEMPLATES,
                               all_characters,
                               all_containers,
                               all_states,
                               n_samples):

    clean_configs, corrupt_configs, random_character_indices = [], [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES['templates'])
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = SampleV3(
            template=template,
            characters=characters,
            containers=containers,
            states=states,
            event_idx=None,
            event_noticed=False,
        )
        corrupt_configs.append(sample)

        random_character = random.choice(all_characters)
        while random_character in characters:
            random_character = random.choice(all_characters)
        new_characters = [random_character, characters[0]]

        sample = SampleV3(
            template=template,
            characters=new_characters,
            containers=containers,
            states=states,
            event_idx=None,
            event_noticed=False
        )
        clean_configs.append(sample)

    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        clean = clean_dataset.__getitem__(idx, set_character=-1, set_container=0, question_type='belief_question')
        corrupt = corrupt_dataset.__getitem__(idx, set_character=0, set_container=0, question_type='belief_question')
        samples.append({
            "clean_prompt": clean['prompt'],
            "clean_ans": clean['target'],
            "corrupt_prompt": corrupt['prompt'],
            "corrupt_ans": corrupt['target'],
            "target": clean_configs[idx].states[0]
        })

    return samples


def get_character_pos_exps(STORY_TEMPLATES,
                           all_characters,
                           all_containers,
                           all_states,
                           n_samples,
                           question_type='state_question'):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES['templates'])
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = SampleV3(
            template=template,
            characters=characters,
            containers=containers,
            states=states,
            event_idx=None,
            event_noticed=False,
        )
        clean_configs.append(sample)

        sample = SampleV3(
            template=template,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=states,
            event_idx=None,
            event_noticed=False
        )
        corrupt_configs.append(sample)
    
    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        random_character_idx = random.choice([0, 1])
        clean = clean_dataset.__getitem__(idx, set_character=1 ^ random_character_idx, set_container=random_character_idx, question_type=question_type)
        corrupt = corrupt_dataset.__getitem__(idx, set_character=random_character_idx, set_container=random_character_idx, question_type=question_type)
        samples.append({
            "clean_prompt": clean['prompt'],
            "clean_ans": clean['target'],
            "corrupt_prompt": corrupt['prompt'],
            "corrupt_ans": corrupt['target'],
            "target": clean_configs[idx].states[random_character_idx]
        })
    
    return samples