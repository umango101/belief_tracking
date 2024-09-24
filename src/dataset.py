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


STORY_TEMPLATE = "<protagonist> is working in a busy restaurant. A customer asks <protagonist> for <obj_1>. <protagonist> grabs an opaque <container_1> and fills it with <obj_1>. Then <protagonist> grabs another opaque <container_2> and fills it with <obj_2>. A coworker named <perpetrator> observes <protagonist> pouring the contents in the <container_1> and the <container_2>. But <perpetrator> didn't hear the customer's request and swaps the <obj_event> in the <container_event> with <obj_swap> while <protagonist> was attending to another task. <protagonist> <saw/didn't see> <perpetrator> swapping the the contents of <container_event>."
NEW_STORY_TEMPLATE = "<protagonist> is working in a busy restaurant. A customer asks <protagonist> for <obj_1>. <protagonist> grabs an opaque <container_1> and fills it with <obj_1>. Then <protagonist> grabs another opaque <container_2> and fills it with <obj_1>. A coworker named <perpetrator> observes <protagonist> pouring the contents in the <container_1> and the <container_2>. But <perpetrator> didn't hear the customer's request and swaps the <obj_event> in the <container_event> with <obj_swap> while <protagonist> was attending to another task. <protagonist> <saw/didn't see> <perpetrator> swapping the the contents of <container_event>."


def swap_entities(story, entity_1, entity_2):
    story = story.replace(entity_1, "<<placeholder>>")
    story = story.replace(entity_2, entity_1)
    story = story.replace("<<placeholder>>", entity_2)
    return story


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


@dataclass(frozen=False)
class SampleV3(DataClassJsonMixin):
    story: str
    protagonist: str
    perpetrator: str
    states: list[str]
    containers: list[str]
    event_idx: Literal[0, 1]
    obsr_event: str
    event_noticed: bool = False
    diff_template: bool = False

    protagonist_belief: dict[str, str] = None
    true_state: dict[str, str] = None

    def __post_init__(self):
        assert len(self.states) == 2 and len(self.containers) == 2
        assert self.states[0] != self.states[1]
        assert self.containers[0] != self.containers[1]

        self.set_story()

    def __eq__(self, other) -> bool:
        assert isinstance(other, SampleV3)
        return (
            self.story == other.story
            and self.protagonist == other.protagonist
            and self.perpetrator == other.perpetrator
            and self.states == other.states
            and self.containers == other.containers
        )

    def set_story(self):
        # protagonist observation
        self.story = self.story + " " + self.obsr_event
        self.story = self.story.replace("<character1>", self.protagonist)
        self.story = self.story.replace("<character2>", self.perpetrator)
        self.story = self.story.replace("<state1>", self.states[0])
        self.story = self.story.replace("<state2>", 'NONE') if self.diff_template else self.story.replace("<state2>", self.states[1])
        self.story = self.story.replace("<container1>", self.containers[0])
        self.story = self.story.replace("<container2>", self.containers[1])

        # event
        self.story = self.story.replace("<state_event>", self.states[self.event_idx])
        self.story = self.story.replace(
            "<container_event>", self.containers[self.event_idx]
        )
        state_swap = self.states[1] if self.diff_template else self.states[1 ^ self.event_idx]
        self.story = self.story.replace("<state_swap>", state_swap)


        # true state (after container swap)
        if self.diff_template:
            self.true_state = {
                self.containers[0]: self.states[0],
                self.containers[1]: self.states[0],
            }
        else:
            self.true_state = {
                self.containers[0]: self.states[0],
                self.containers[1]: self.states[1],
            }
        self.true_state[self.containers[self.event_idx]] = state_swap

        # protagonist belief
        if self.event_noticed == False:
            self.protagonist_belief = {
                self.containers[0]: self.states[0],
                self.containers[1]: self.states[1],
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
        set_container: Literal[0, 1] | None = None,
        set_obj: Literal[0, 1] | None = None,
        set_actor: Literal["protagonist", "perpetrator"] = "protagonist",
        question_type: Literal["belief", "true_state"] = "belief",
    ) -> tuple[str, Literal["yes", "no"]]:
        prompt = f"Instruction: {self.instruction.strip()}\n\n"
        prompt += f"Story: {self.samples[idx].story.strip()}\n"

        q_actor = (
            self.samples[idx].protagonist
            if set_actor == "protagonist"
            else self.samples[idx].perpetrator
        )
        belief_states = (
            self.samples[idx].protagonist_belief
            if set_actor == "protagonist"
            else self.samples[idx].true_state
        )
        if set_ans is not None:
            ans = set_ans
            assert (
                set_container is None or set_obj is None
            ), "if both the container and the obj is set true, then the answer is determined"

            if set_container is None and set_obj is None:
                set_container = random.choice([0, 1])

            if set_container is not None:
                q_container = self.samples[idx].containers[set_container]
                if question_type == "belief":
                    obj_yes = belief_states[q_container]
                else:
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
                if question_type == "belief":
                    container_yes = c1 if belief_states[c1] == q_obj else c2
                else:
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

            if question_type == "belief":
                ans = "yes" if belief_states[q_container] == q_obj else "no"
            else:
                ans = "yes" if self.samples[idx].true_state[q_container] == q_obj else "no"

        if question_type == "belief":
            prompt += (
                f"Question: Does {q_actor} believe the {q_container} contains {q_obj}?\n"
            )
        else:
            prompt += f"Question: Does the {q_container} contain {q_obj}?\n"
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