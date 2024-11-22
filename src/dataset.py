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


def swap_entities(story, entity_1, entity_2):
    story = story.replace(entity_1, "<<placeholder>>")
    story = story.replace(entity_2, entity_1)
    story = story.replace("<<placeholder>>", entity_2)
    return story


@dataclass(frozen=False)
class SampleV3(DataClassJsonMixin):
    template: dict[str, str]
    characters: list[str]
    containers: list[str]
    states: list[str]
    visibility: bool = False
    event_idx: Optional[Literal[0, 1, None]] = None  # which container is swapped?
    event_noticed: bool = False  # the protagonist sees the swap?

    story: str | None = None
    character_belief: list[dict[str, str]] = None

    def __post_init__(self):
        if len(self.characters) == 1:
            self.characters.append("<N/A>")
        # assert len(self.states) == 2 and (len(self.containers) == 2 or len(self.containers) == 3) and len(self.characters) == 2
        # assert self.states[0] != self.states[1]
        # assert self.containers[0] != self.containers[1]
        # assert self.characters[0] != self.characters[1]

        self.set_story()

    def __eq__(self, other) -> bool:
        assert isinstance(other, SampleV3)
        return self.story == other.story

    def set_entity_names(self):
        # characters
        self.story = self.story.replace(
            STORY_TEMPLATES["placeholders"]["entity"]["character"][0],
            self.characters[0],
        )
        self.story = self.story.replace(
            STORY_TEMPLATES["placeholders"]["entity"]["character"][1],
            self.characters[1],
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
        self.story = self.template["context"]

        # true state
        self.world_state = {
            self.containers[0]: self.states[0],
            self.containers[1]: self.states[1],
        }
        self.character_belief = [self.world_state.copy(), self.world_state.copy()]

        if not self.visibility:
            self.character_belief[0][self.containers[1]] = "unknown"
            self.character_belief[1][self.containers[0]] = "unknown"

        # event
        if self.event_idx is not None:  # Event happened
            self.story += f' {self.template["causal_event"]}'
            state_swap = self.states[1 ^ self.event_idx]
            state_event = self.states[self.event_idx]
            container_event = self.containers[self.event_idx]

            self.story += f' {self.template["event_noticed"]}'

            self.story = self.story.replace(
                STORY_TEMPLATES["placeholders"]["event"]["container_event"],
                container_event,
            )
            self.story = self.story.replace(
                STORY_TEMPLATES["placeholders"]["event"]["state_event"], state_event
            )
            self.story = self.story.replace(
                STORY_TEMPLATES["placeholders"]["event"]["state_swap"], state_swap
            )

            # did the protagonist see the event happening?
            observation = "observed" if self.event_noticed else "does not observe"
            self.story = self.story.replace(
                STORY_TEMPLATES["placeholders"]["notice"], observation
            )
            self.story = self.story.replace(
                STORY_TEMPLATES["placeholders"]["event"]["container_event"],
                container_event,
            )
            self.world_state[container_event] = state_swap
            self.character_belief[1][self.containers[self.event_idx]] = state_swap

            # protagonist belief
            if self.event_noticed == True:
                self.character_belief[0][self.containers[self.event_idx]] = state_swap

        # else:  # Event did not happen
        #     assert (
        #         self.event_noticed == False
        #     ), "If there is no causal event, there is nothing to observe"
        #     assert (
        #         STORY_TEMPLATES["placeholders"]["entity"]["character"][1] not in self.story
        #     ), "If there is no causal event, there is no perpetrator to blame"

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
        """1. Track each character's beliefs as defined in the story. 2. Update a character's belief only when they directly observe an event that alters their current belief or when they perform the event themselves. 3. If a character does not observe the event, their belief should remain unchanged, even if the event occurs. 4. To answer the question following the story, predict the attribute token associated with the container, based strictly on this final belief state. If no attribute is associated with the specific character or container in the question, predict 'unknown'."""
    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
        set_container: Literal[-1, 0, 1] | None = None,
        set_state: Literal[0, 1] | None = None,
        set_character: Literal[-1, 0, 1] | None = None,
        question_type: Literal["belief_question", "state_question"] = "belief_question",
    ) -> tuple[str, Literal["yes", "no"]]:
        prompt = f"Instruction: {self.instruction.strip()}\n\n"
        prompt += f"Story: {self.samples[idx].story.strip()}\n"

        sample = self.samples[idx]

        if set_character == -1:
            q_actor = sample.characters[1]
            belief_states = {}
        else:
            if sample.event_idx is None and set_character is None:
                # assert set_character != 1
                set_character = 0
            else:
                set_character = (
                    random.choice([0, 1]) if set_character is None else set_character
                )
            q_actor = sample.characters[set_character]
            belief_states = sample.character_belief[set_character]

        initial_states = sample.world_state

        if set_container is None:
            q_container = random.choice(sample.containers)
        elif set_container != -1:
            q_container = sample.containers[set_container]
        else:
            q_container = sample.containers[2]

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
        }



# world_state: "bigtom_worldstate.csv",
# TOM dataset: "bigtom/0_forward_belief_false_control/stories.csv"
def load_TOM_dataset(
    file_name: str = "bigtom/0_forward_belief_false_belief/stories.csv",
):
    ws_csv = pd.read_csv(
        os.path.join(env_utils.DEFAULT_DATA_DIR, file_name),
        delimiter=";",
    )
    samples: list[Sample] = []
    for idx, row in ws_csv.iterrows():
        samples.append(
            Sample(
                story=row["story"],
                question=row["question"],
                answer=row["answer"],
                distractor=row["distractor"],
            )
        )
    return Dataset(samples=samples)
