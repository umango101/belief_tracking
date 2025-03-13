import json
import os
import random
from dataclasses import dataclass
from typing import Literal, Optional, List

import pandas as pd
from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

from src.utils import env_utils


STORY_TEMPLATE_PATH = os.path.join(env_utils.DEFAULT_DATA_DIR, "story_templates.json")

with open(STORY_TEMPLATE_PATH, "r") as f:
    STORY_TEMPLATES = json.load(f)


@dataclass(frozen=False)
class SampleV3(DataClassJsonMixin):
    template_idx: int
    characters: list[str]
    containers: list[str]
    states: list[str]

    story: str | None = None
    character_belief: list[dict[str, str]] = None

    def __post_init__(self):
        if len(self.characters) == 1:
            self.characters.append("<N/A>")
        assert len(self.states) == 2 and len(self.containers) == 2 and len(self.characters) == 2
        assert self.states[0] != self.states[1]
        assert self.containers[0] != self.containers[1]
        assert self.characters[0] != self.characters[1]

        self.set_story()

    def __eq__(self, other) -> bool:
        assert isinstance(other, SampleV3)
        return self.story == other.story

    def set_entity_names(self):
        # characters
        for i, character in enumerate(self.characters):
            self.story = self.story.replace(
                STORY_TEMPLATES["placeholders"]["entity"][f"character"][i], character
            )

        # containers
        self.story = self.story.replace(
            STORY_TEMPLATES["placeholders"]["entity"]["container"][0],
            self.containers[0],
        )
        self.story = self.story.replace(
            STORY_TEMPLATES["placeholders"]["entity"]["container"][1],
            self.containers[1],
        )

        # states
        self.story = self.story.replace(
            STORY_TEMPLATES["placeholders"]["entity"]["state"][0], self.states[0]
        )
        self.story = self.story.replace(
            STORY_TEMPLATES["placeholders"]["entity"]["state"][1], self.states[1]
        )

    def set_story(self):
        self.template = STORY_TEMPLATES["templates"][self.template_idx]
        self.story = self.template["context"]

        # true state
        self.world_state = {
            self.containers[0]: self.states[0],
            self.containers[1]: self.states[1],
        }
        self.character_belief = [self.world_state.copy(), self.world_state.copy()]

        # set the character beliefs
        if self.template_idx == 0 or self.template_idx == 2:
            self.character_belief[0][self.containers[1]] = "unknown"
            self.character_belief[1][self.containers[0]] = "unknown"
        elif self.template_idx == 1:
            self.character_belief[1][self.containers[0]] = "unknown"

        # set the common entity names
        self.set_entity_names()
        assert "<" not in self.story and ">" not in self.story

        return self.story

    def __str__(self):
        if self.story is None:
            self.set_story()
        return self.story


@dataclass(frozen=False)
class DatasetV3(DataClassJsonMixin):
    samples: list[SampleV3]
    instruction: str = (
        "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any beliefs about the container and its contents which they cannot observe. 4. To answer the question, predict only what is inside the queried container, strictly based on the belief of the character, mentioned in the question. 5. If the queried character has no belief about the container in question, then predict 'unknown'. 6. Do not predict container or character as the final output."
    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
        set_container: Literal[0, 1] | None = None,
        set_state: Literal[0, 1] | None = None,
        set_character: Literal[0, 1] | None = None,
        question_type: Literal["belief_question", "state_question"] = "belief_question",
    ):
        prompt = f"Instruction: {self.instruction.strip()}\n\n"
        prompt += f"Story: {self.samples[idx].story.strip()}\n"

        sample = self.samples[idx]
        
        set_character = random.choice([0, 1]) if set_character is None else set_character
        q_actor = sample.characters[set_character]
        belief_states = sample.character_belief[set_character]
        initial_states = sample.world_state
        
        set_container = random.choice([0, 1]) if set_container is None else set_container
        q_container = sample.containers[set_container]

        if question_type == "belief_question":
            if q_container in belief_states:
                ans = belief_states[q_container]
            else:
                ans = "unknown"
        else:
            if q_container in initial_states:
                ans = initial_states[q_container]
            else:
                ans = "unknown"

        question = sample.template[question_type]

        question = question.replace(
            STORY_TEMPLATES["placeholders"]["question"]["character"], q_actor
        )
        question = question.replace(
            STORY_TEMPLATES["placeholders"]["question"]["container"], q_container
        )

        prompt += f"Question: {question}\n"
        prompt += f"Answer:"
        return {
            "characters": sample.characters,
            "objects": sample.containers,
            "states": sample.states,
            "story": sample.story,
            "question": question,
            "target": ans,
            "prompt": prompt,
            "character_idx": set_character,
            "object_idx": set_container,
            "template_idx": sample.template_idx,
        }
