import re
import random
import base64
import numpy as np
import matplotlib.pyplot as plt
from html2image import Html2Image
from typing import List, Dict, Set
from src.dataset import SampleV3, DatasetV3
from io import BytesIO

random.seed(10)

token_pos_coords = {
    "e1_last": (95, 170),
    "e2_last": (95, 325),
    "e1_query_obj_real": (310, 145),
    "e1_query_obj_belief": (450, 145),
    "e2_query_obj_real": (310, 300),
    "e2_query_obj_belief": (450, 300),
    "e1_query_charac": (260, 145),
    "e2_query_charac": (260, 300),
    "e1_obj1": (360, 105),
    "e1_obj2": (280, 125),
    "e2_obj1": (360, 260),
    "e2_obj2": (280, 280),
    "e1_state1": (610, 105),
    "e1_state2": (550, 125),
    "e2_state1": (610, 260),
    "e2_state2": (550, 280),
    "e1_charac1": (120, 105),
    "e1_charac2": (730, 105),
    "e2_charac1": (120, 255),
    "e2_charac2": (720, 250),
}


class StoryGenerator:
    def __init__(
        self,
        characters: Set,
        objects: Set,
        states: Set,
        stories: List[Dict],
        target: str,
        arrows: List[Dict],
        plot_data: Dict,
    ):
        self.characters = characters
        self.objects = objects
        self.states = states
        self.arrows = arrows
        self.stories = stories
        self.target = target
        self.plot_data = plot_data

    def color_text(self, text: str) -> str:
        """Apply color formatting to specific words in the text."""
        words = text.split()
        colored_words = []

        for word in words:
            clean_word = re.sub(r"[^\w\s]", "", word)  # Remove punctuation

            if clean_word in self.characters:
                colored_word = (
                    f'<span style="color:skyblue; font-weight:bold;">{word}</span>'
                )
            elif clean_word in self.objects:
                colored_word = (
                    f'<span style="color:maroon; font-weight:bold;">{word}</span>'
                )
            elif clean_word in self.states:
                colored_word = (
                    f'<span style="color:darkgreen; font-weight:bold;">{word}</span>'
                )
            else:
                colored_word = word

            colored_words.append(colored_word)

        return " ".join(colored_words)

    def generate_html(self) -> str:
        """Generate HTML content with stories, arrows, and a right-aligned plot."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: monospace;
            white-space: pre-wrap;
            line-height: 1.2;
            margin: 0 5px;
            background-color: white;
            position: relative;
            font-size: 18px;
            display: flex;
            height: 400px;
            width: 1300px;
        }
        .left-container {
            flex: 1;
            display: flex;
            max-width: 60%;
            flex-direction: column;
            justify-content: space-around;
        }
        .right-container {
            display: flex;
            flex-direction: column;
            flex: 0 0 40%;
        }
        .story-container {
            position: relative;
            border: 1px solid black;
            padding: 5px;
        }
        .target-container {
            border: 1px solid black;
            display: inline-block;
            padding: 5px;
        }
        .svg-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .label {
            background-color: brown;
            color: white;
            padding: 2px 5px;
            margin-bottom: 2px;
            display: inline-block;
            font-weight: bold;
        }
        img {
            height: 90%;
            width: -webkit-fill-available;
            margin: auto;
        }
    </style>
</head>
<body>
<div class="left-container">
"""

        # Add story blocks with labels
        for i, story in enumerate(self.stories):
            colored_story = self.color_text(story["story"])
            colored_question = self.color_text(story["question"])
            colored_answer = self.color_text(story["answer"])

            if i == 0:
                html_content += f"""<div class="box-wrapper"><div class="label">Corrupt Example</div><div class="story-container" id="story-{i}">Story: {colored_story}<br>Question: {colored_question}<br>Answer: {colored_answer}</div></div>"""
            else:
                html_content += f"""<div class="box-wrapper"><div class="label">Clean Example</div><div class="story-container" id="story-{i}">Story: {colored_story}<br>Question: {colored_question}<br>Answer: {colored_answer}</div></div>"""

        # Add target with label
        html_content += (
            f"""<div class="target-container">{self.color_text(self.target)}</div>"""
        )

        # Add SVG overlay for arrows
        for i, arrow in enumerate(self.arrows):
            try:
                start_x, start_y = arrow["start"]
                end_x, end_y = arrow["end"]
                color = arrow.get("color", "black")

                if start_x == end_x and start_y == end_y:
                    html_content += f"""
<svg class="svg-container">
    <path d="M {start_x},{start_y} C {start_x + 75},{start_y - 75} {end_x - 75},{end_y - 75} {end_x - 10},{end_y}"
          fill="none"
          stroke="{color}"
          stroke-width="4"
          style="paint-order: stroke fill;"
          filter="drop-shadow(0 0 2px black)"
          marker-end="url(#arrowhead_{i})"/>
    <defs>
        <marker id="arrowhead_{i}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>
        </marker>
    </defs>
</svg>
"""
                else:
                    html_content += f"""
<svg class="svg-container">
    <path d="M {start_x},{start_y} {end_x},{end_y}"
          fill="none"
          stroke="{color}"
          stroke-width="4"
          style="paint-order: stroke fill;"
          filter="drop-shadow(0 0 2px black)"
          marker-end="url(#arrowhead_{i})"/>
    <defs>
        <marker id="arrowhead_{i}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>
        </marker>
    </defs>
</svg>
"""
            except KeyError as e:
                print(f"Arrow configuration error: {e}")

        html_content += """
</div>
<div class="right-container">
"""

        # Generate the line plot
        x = np.arange(len(self.plot_data["labels"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, self.plot_data["values"], marker="o", color="b", label="Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(self.plot_data["labels"])
        ax.set_title(self.plot_data["title"])
        ax.set_xlabel(self.plot_data["x_label"])
        ax.set_ylabel(self.plot_data["y_label"])
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
        # Change the font family to times new roman
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(18)
            item.set_fontname("Times New Roman")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plot_data = buf.getvalue()
        buf.close()

        # Encode the plot as a base64 URI
        plot_data_uri = (
            f'data:image/png;base64,{base64.b64encode(plot_data).decode("utf-8")}'
        )
        html_content += f'<img src="{plot_data_uri}" alt="Plot"/>'

        html_content += """
</div>
</body>
</html>"""

        return html_content

    def save_html(self, filename: str = "../plots/experiments/output.html"):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.generate_html())

    def save_image(self, filename: str = "output.png"):
        hti = Html2Image(output_path="../plots/experiments/")
        hti.screenshot(html_str=self.generate_html(), save_as=filename)


def get_value_fetcher_exps(
    STORY_TEMPLATES,
    all_characters,
    all_containers,
    all_states,
    n_samples,
    question_type="state_question",
):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES["templates"])
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

        template = random.choice(STORY_TEMPLATES["templates"])
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
            clean = clean_dataset.__getitem__(
                idx,
                question_type=question_type,
                set_character=set_character,
                set_container=set_character,
            )
            corrupt = corrupt_dataset.__getitem__(
                idx,
                question_type=question_type,
                set_character=1 ^ set_character,
                set_container=1 ^ set_character,
            )
        else:
            clean = clean_dataset.__getitem__(idx, question_type=question_type)
            corrupt = corrupt_dataset.__getitem__(idx, question_type=question_type)

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_target": clean["target"],
                "clean_prompt": clean["prompt"],
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_target": corrupt["target"],
                "corrupt_prompt": corrupt["prompt"],
            }
        )

    return samples


def get_pos_trans_exps(
    STORY_TEMPLATES,
    all_characters,
    all_containers,
    all_states,
    n_samples,
    question_type="state_question",
):
    clean_configs, corrupt_configs, intervention_pos = [], [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES["templates"])
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
            characters=(
                characters
                if question_type == "state_question"
                else list(reversed(characters))
            ),
            containers=list(reversed(containers)),
            states=new_states,
            event_idx=None,
            event_noticed=False,
        )
        corrupt_configs.append(sample)

    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        set_container = random.choice([0, 1])

        if question_type == "belief_question":
            clean = clean_dataset.__getitem__(
                idx,
                set_container=set_container,
                set_character=set_container,
                question_type=question_type,
            )
            corrupt = corrupt_dataset.__getitem__(
                idx,
                set_container=1 ^ set_container,
                set_character=1 ^ set_container,
                question_type=question_type,
            )

        else:
            clean = clean_dataset.__getitem__(
                idx, set_container=set_container, question_type=question_type
            )
            corrupt = corrupt_dataset.__getitem__(
                idx, set_container=1 ^ set_container, question_type=question_type
            )

        samples.append(
            {
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": clean_configs[idx].states[1 ^ set_container],
            }
        )

    return samples


def get_obj_tracing_exps(
    STORY_TEMPLATES, all_characters, all_containers, all_states, n_samples
):
    clean_configs, corrupt_configs, random_container_indices = [], [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES["templates"])
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
            event_noticed=False,
        )
        clean_configs.append(sample)

    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        clean = clean_dataset.__getitem__(idx, set_container=-1)
        corrupt = corrupt_dataset.__getitem__(
            idx, set_container=random_container_indices[idx]
        )
        samples.append(
            {
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": clean_configs[idx].states[random_container_indices[idx]],
            }
        )

    return samples


def get_state_tracing_exps(
    STORY_TEMPLATES, all_characters, all_containers, all_states, n_samples
):
    clean_configs, corrupt_configs, random_state_indices = [], [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES["templates"])
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
            event_noticed=False,
        )
        clean_configs.append(sample)

    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        clean = clean_dataset.__getitem__(idx, set_container=random_state_indices[idx])
        corrupt = corrupt_dataset.__getitem__(
            idx, set_container=random_state_indices[idx]
        )
        samples.append(
            {
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": corrupt_configs[idx].states[random_state_indices[idx]],
            }
        )

    return samples


def get_state_pos_exps(
    STORY_TEMPLATES,
    all_characters,
    all_containers,
    all_states,
    n_samples,
    question_type="state_question",
):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES["templates"])
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
            event_noticed=False,
        )
        corrupt_configs.append(sample)

    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        random_container_idx = random.choice([0, 1])

        if question_type == "belief_question":
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_container_idx,
                set_character=random_container_idx,
                question_type=question_type,
            )
            corrupt = corrupt_dataset.__getitem__(
                idx,
                set_container=random_container_idx,
                set_character=random_container_idx,
                question_type=question_type,
            )
        else:
            clean = clean_dataset.__getitem__(
                idx, set_container=1 ^ random_container_idx, question_type=question_type
            )
            corrupt = corrupt_dataset.__getitem__(
                idx, set_container=random_container_idx, question_type=question_type
            )
        samples.append(
            {
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": clean_configs[idx].states[1 ^ random_container_idx],
            }
        )

    return samples


def get_character_tracing_exps(
    STORY_TEMPLATES, all_characters, all_containers, all_states, n_samples
):

    clean_configs, corrupt_configs, random_character_indices = [], [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES["templates"])
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
            event_noticed=False,
        )
        clean_configs.append(sample)

    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        clean = clean_dataset.__getitem__(
            idx, set_character=-1, set_container=0, question_type="belief_question"
        )
        corrupt = corrupt_dataset.__getitem__(
            idx, set_character=0, set_container=0, question_type="belief_question"
        )
        samples.append(
            {
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": clean_configs[idx].states[0],
            }
        )

    return samples


def get_character_pos_exps(
    STORY_TEMPLATES,
    all_characters,
    all_containers,
    all_states,
    n_samples,
    question_type="state_question",
):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template = random.choice(STORY_TEMPLATES["templates"])
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
            event_noticed=False,
        )
        corrupt_configs.append(sample)

    clean_dataset = DatasetV3(clean_configs)
    corrupt_dataset = DatasetV3(corrupt_configs)

    for idx in range(n_samples):
        random_character_idx = random.choice([0, 1])
        clean = clean_dataset.__getitem__(
            idx,
            set_character=1 ^ random_character_idx,
            set_container=random_character_idx,
            question_type=question_type,
        )
        corrupt = corrupt_dataset.__getitem__(
            idx,
            set_character=random_character_idx,
            set_container=random_character_idx,
            question_type=question_type,
        )
        samples.append(
            {
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": clean_configs[idx].states[random_character_idx],
            }
        )

    return samples
