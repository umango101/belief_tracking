import json
import os
import random
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

from src.utils import env_utils


@dataclass(frozen=True)
class Sample(DataClassJsonMixin):
    story: str
    question: str
    answer: str
    distractor: str


@dataclass(frozen=True)
class Dataset(DataClassJsonMixin):
    samples: list[Sample]
    instruction: str = (
        """Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose the correct option by predicting the answer option after the "Answer:" tag."""
    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
        tags: tuple[int, int] = ["a", "b"],
        correct_ans_idx: Literal[0, 1] = None,
    ) -> Sample:
        question = f"Instruction: {self.instruction.strip().replace('<op1>', tags[0]).replace('<op2>', tags[1])}\n\n"
        question += f"Story: {self.samples[idx].story.strip()}\n\n"
        question += f"Question: {self.samples[idx].question.strip()}\n"

        correct_ans_idx = (
            random.choice([0, 1]) if correct_ans_idx is None else correct_ans_idx
        )
        distractor_idx = 1 - correct_ans_idx
        option_dict = {
            tags[correct_ans_idx]: self.samples[idx].answer,
            tags[distractor_idx]: self.samples[idx].distractor,
        }
        question += f"{tags[0]}) {option_dict[tags[0]].strip()}\n"
        question += f"{tags[1]}) {option_dict[tags[1]].strip()}\n"
        question += "Choose one of the following:\n"
        question += f"Answer:"

        return question, tags[correct_ans_idx]


@dataclass(frozen=False)
class SampleV2(DataClassJsonMixin):
    story: str
    protagonist: str
    perpetrator: str
    states: list[str]
    containers: list[str]

    def __post_init__(self):
        assert len(self.states) == 2 and len(self.containers) == 2
        assert self.states[0] != self.states[1]
        assert self.containers[0] != self.containers[1]

        self.set_story()

    def set_story(self):
        self.story = self.story.replace("<character1>", self.protagonist)
        self.story = self.story.replace("<character2>", self.perpetrator)
        self.story = self.story.replace("<state1>", self.states[0])
        self.story = self.story.replace("<state2>", self.states[1])
        self.story = self.story.replace("<container1>", self.containers[0])
        self.story = self.story.replace("<container2>", self.containers[1])

        self.true_state = {
            self.containers[0]: self.states[0],
            self.containers[1]: self.states[1],
        }

        return self.story


@dataclass(frozen=False)
class DatasetV2(DataClassJsonMixin):
    samples: list[SampleV2]
    instruction: str = (
        """Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose "yes" or "no" after the "Answer:" tag."""
    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
        set_ans: Optional[Literal["yes", "no"]] = None,
        set_container: Literal[0, 1] | None = None,
        set_obj: Literal[0, 1] | None = None,
    ) -> tuple[str, Literal["yes", "no"]]:
        prompt = f"Instruction: {self.instruction.strip()}\n\n"
        prompt += f"Story: {self.samples[idx].story.strip()}\n"

        if set_ans is not None:
            ans = set_ans
            assert (
                set_container is None or set_obj is None
            ), "if both the container and the obj is set true, then the answer is determined"

            if set_container is None and set_obj is None:
                set_container = random.choice([0, 1])

            if set_container is not None:
                q_container = self.samples[idx].containers[set_container]
                obj_yes = self.samples[idx].true_state[q_container]
                obj_no = (
                    self.samples[idx].states[0]
                    if self.samples[idx].states[1] == obj_yes
                    else self.samples[idx].states[1]
                )
                assert obj_yes != obj_no
                q_obj = obj_yes if set_ans == "yes" else obj_no
            elif set_obj is not None:
                q_obj = self.samples[idx].states[set_obj]
                c1, c2 = self.samples[idx].containers
                container_yes = c1 if self.samples[idx].true_state[c1] == q_obj else c2
                container_no = c1 if container_yes == c2 else c2
                assert container_yes != container_no
                q_container = container_yes if set_ans == "yes" else container_no

        else:
            q_container = (
                random.choice(self.samples[idx].containers)
                if set_container is None
                else self.samples[idx].containers[set_container]
            )
            q_obj = (
                random.choice(self.samples[idx].states)
                if set_obj is None
                else self.samples[idx].states[set_obj]
            )

            ans = "yes" if self.samples[idx].true_state[q_container] == q_obj else "no"

        prompt += f"Question: Does the {q_container} contain {q_obj}?\n"
        prompt += f"Answer:"
        return prompt, ans


# STORY_TEMPLATE = "<protagonist> is working in a busy restaurant. A customer asks <protagonist> for <obj_1>. <protagonist> grabs an opaque <container_1> and fills it with <obj_1>. Then <protagonist> grabs another opaque <container_2> and fills it with <obj_2>. A coworker named <perpetrator> observes <protagonist> pouring the contents in the <container_1> and the <container_2>. But <perpetrator> didn't hear the customer's request and swaps the <obj_event> in the <container_event> with <obj_swap> while <protagonist> was attending to another task. <protagonist> can't see what is in the <container_1> and the <container_2> without opening their lid. <protagonist> <saw/didn't see> <perpetrator> swapping the the contents of <container_event>."
# STORY_TEMPLATE = "<protagonist> is a magician performing at a grand theater. <protagonist> wants to amaze the audience with a trick involving a <obj_1>. <protagonist> places the <obj_1> in a <container_1> and sets it on the stage. Then <protagonist> prepares a backup <container_2> and places a <obj_2> inside. An assistant named <perpetrator>, who thinks the trick should be different, swaps the <obj_event> in the <container_event> with the <obj_swap> while <protagonist> is backstage. <protagonist> <saw/didn't see> <perpetrator> swapping the the contents of <container_event>."
# STORY_TEMPLATE = "<protagonist> is working in a busy restaurant. A customer asks <protagonist> for <obj_1>. <protagonist> grabs an opaque <container_1> and fills it with <obj_1>. Then <protagonist> grabs another opaque <container_2> and fills it with <obj_2>."

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
