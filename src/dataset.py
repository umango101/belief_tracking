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


@dataclass(frozen=True)
class SampleV2(DataClassJsonMixin):
    story: str
    actor: str
    obj_belief: str
    obj_true: str
    container: str

    def __post_init__(self):
        assert self.actor in self.story
        assert self.obj_belief in self.story
        assert self.obj_true in self.story
        assert self.container in self.story
        assert self.obj_belief != self.obj_true

    def __eq__(self, other) -> bool:
        assert isinstance(other, SampleV2)
        return (
            self.story == other.story
            and self.actor == other.actor
            and self.obj_belief == other.obj_belief
            and self.obj_true == other.obj_true
            and self.container == other.container
        )


STORY_TEMPLATE = "<protagonist> is working in a busy restaurant. A customer asks <protagonist> for <obj_1>. <protagonist> grabs an opaque <container_1> and fills it with <obj_1>. Then <protagonist> grabs another opaque <container_2> and fills it with <obj_2>. A coworker named <perpetrator> observes <protagonist> pouring the contents in the <container_1> and the <container_2>. But <perpetrator> didn't hear the customer's request and swaps the <obj_event> in the <container_event> with <obj_swap> while <protagonist> was attending to another task. <protagonist> can't see what is in the <container_1> and the <container_2> without opening their lid. <protagonist> <saw/didn't see> <perpetrator> swapping the the contents of <container_event>."


def swap_entities(story, entity_1, entity_2):
    story = story.replace(entity_1, "<<placeholder>>")
    story = story.replace(entity_2, entity_1)
    story = story.replace("<<placeholder>>", entity_2)
    return story


@dataclass(frozen=False)
class SampleV3(DataClassJsonMixin):
    protagonist: str
    perpetrator: str
    objects: list[str]
    containers: list[str]
    event_idx: Literal[0, 1] = 0  # which container is swapped?
    event_noticed: bool = False  # the protagonist sees the swap?

    story: str | None = None
    protagonist_belief: dict[str, str] = None
    true_state: dict[str, str] = None

    def __post_init__(self):
        assert len(self.objects) == 2 and len(self.containers) == 2
        assert self.objects[0] != self.objects[1]
        assert self.containers[0] != self.containers[1]

        self.set_story()

    def __eq__(self, other) -> bool:
        assert isinstance(other, SampleV3)
        return (
            self.story == other.story
            and self.protagonist == other.protagonist
            and self.perpetrator == other.perpetrator
            and self.objects == other.objects
            and self.containers == other.containers
        )

    def set_story(self):
        self.story = STORY_TEMPLATE
        self.story = self.story.replace("<protagonist>", self.protagonist)
        self.story = self.story.replace("<perpetrator>", self.perpetrator)
        self.story = self.story.replace("<obj_1>", self.objects[0])
        self.story = self.story.replace("<obj_2>", self.objects[1])
        self.story = self.story.replace("<container_1>", self.containers[0])
        self.story = self.story.replace("<container_2>", self.containers[1])

        # event
        self.story = self.story.replace("<obj_event>", self.objects[self.event_idx])
        self.story = self.story.replace(
            "<container_event>", self.containers[self.event_idx]
        )
        obj_swap = self.objects[1 ^ self.event_idx]
        self.story = self.story.replace("<obj_swap>", obj_swap)
        # protagonist observation
        observation = "saw" if self.event_noticed else "didn't see"
        self.story = self.story.replace("<saw/didn't see>", observation)

        # true state
        self.true_state = {
            self.containers[0]: self.objects[0],
            self.containers[1]: self.objects[1],
        }
        self.true_state[self.containers[self.event_idx]] = obj_swap

        # protagonist belief
        if self.event_idx == 0 and self.event_noticed == False:
            self.protagonist_belief = {
                self.containers[0]: self.objects[0],
                self.containers[1]: self.objects[1],
            }
        else:
            self.protagonist_belief = self.true_state.copy()

        return self.story

    def __str__(self):
        if self.story is None:
            self.set_story()
        return self.story


@dataclass(frozen=False)
class DatasetV3(DataClassJsonMixin):
    samples: list[SampleV3]
    instruction: str = (
        """Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose "yes" or "no" after the "Answer:" tag."""
    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
        set_ans: Optional[Literal["yes", "no"]] = None,
        set_actor: Literal["protagonist", "perpetrator"] = "protagonist",
        set_container: Literal[0, 1] = 0,
    ) -> tuple[str, Literal["yes", "no"]]:
        prompt = f"Instruction: {self.instruction.strip()}\n\n"
        prompt += f"Story: {self.samples[idx].story.strip()}\n\n"

        ans = random.choice(["yes", "no"]) if set_ans is None else set_ans
        actor = (
            self.samples[idx].protagonist
            if set_actor == "protagonist"
            else self.samples[idx].perpetrator
        )
        container = self.samples[idx].containers[set_container]
        container_states = (
            self.samples[idx].protagonist_belief
            if set_actor == "protagonist"
            else self.samples[idx].true_state
        )
        obj_yes = container_states[container]
        obj_no = (
            self.samples[idx].objects[0]
            if self.samples[idx].objects[1] == obj_yes
            else self.samples[idx].objects[1]
        )
        assert obj_yes != obj_no

        obj = obj_yes if ans == "yes" else obj_no
        prompt += f"Question: Does {actor} believe the {container} contains {obj}?\n"
        prompt += f"Answer:"
        return prompt, ans


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
