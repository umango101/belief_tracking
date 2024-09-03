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
        """Keep track of people's knowledge defined in the story. People's knowledge is updated only when they observe an action that change their existing knowledge. To answer the question following the story, choose the correct option by predicting the answer option (either <op1> or <op2>) after the "Answer:" tag."""
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
        question += f"Answer:"

        return question, tags[correct_ans_idx]


def load_worldstate_dataset():
    ws_csv = pd.read_csv(
        os.path.join(
            env_utils.DEFAULT_DATA_DIR,
            # "bigtom_worldstate.csv",
            "bigtom/0_forward_belief_false_belief/stories.csv",
        ),
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
