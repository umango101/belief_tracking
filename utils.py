import random

from src.dataset import Dataset, Sample


def get_charac_pos_exp(
    STORY_TEMPLATES,
    all_characters,
    all_objects,
    all_states,
    n_samples,
    question_type="belief_question",
):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        template = STORY_TEMPLATES["templates"][template_idx]
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_objects[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        new_states = random.sample(all_states[template["state_type"]], 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states[template["state_type"]], 2)

        sample = Sample(
            template_idx=template_idx,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=new_states,
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = random.choice([0, 1])

        if question_type == "belief_question":
            clean = clean_dataset.__getitem__(
                idx,
                set_container=1 ^ random_choice,
                set_character=random_choice,
                question_type=question_type,
            )
            corrupt = corrupt_dataset.__getitem__(
                idx,
                set_container=1 ^ random_choice,
                set_character=1 ^ random_choice,
                question_type=question_type,
            )

        else:
            clean = clean_dataset.__getitem__(
                idx, set_container=random_choice, question_type=question_type
            )
            corrupt = corrupt_dataset.__getitem__(
                idx, set_container=1 ^ random_choice, question_type=question_type
            )

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1 ^ random_choice],
            }
        )

    return samples


def get_visibility_align_exps(
    STORY_TEMPLATES,
    all_characters,
    all_containers,
    all_states,
    n_samples,
    diff_visibility=False,
    both_directions=False,
):
    clean_configs, corrupt_configs, orders, samples = [], [], [], []

    for idx in range(n_samples):
        template = STORY_TEMPLATES["templates"][0]
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        no_vis_sample = Sample(
            template=template,
            characters=characters,
            containers=containers,
            states=states,
            visibility=False,
            event_idx=None,
            event_noticed=False,
        )

        new_states = random.sample(all_states[template["state_type"]], 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states[template["state_type"]], 2)

        vis_sample = Sample(
            template=(
                STORY_TEMPLATES["templates"][1]
                if diff_visibility
                else STORY_TEMPLATES["templates"][0]
            ),
            characters=characters,
            containers=containers,
            states=new_states,
            visibility=True,
            event_idx=None,
            event_noticed=False,
        )

        order = random.choice([0, 1]) if both_directions else 0
        if order == 0:
            clean_configs.append(no_vis_sample)
            corrupt_configs.append(vis_sample)
        else:
            clean_configs.append(vis_sample)
            corrupt_configs.append(no_vis_sample)
        orders.append(order)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = 0

        if question_type == "belief_question":
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=1 ^ random_choice if diff_visibility else random_choice,
                question_type=question_type,
            )
            corrupt = corrupt_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=1 ^ random_choice if diff_visibility else random_choice,
                question_type=question_type,
            )

        else:
            clean = clean_dataset.__getitem__(
                idx, set_container=random_choice, question_type=question_type
            )
            corrupt = corrupt_dataset.__getitem__(
                idx, set_container=1 ^ random_choice, question_type=question_type
            )

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": (
                    " " + clean_configs[idx].states[random_choice]
                    if orders[idx] == 0
                    else " unknown"
                ),
            }
        )

    return samples


def get_obj_pos_exps(
    STORY_TEMPLATES,
    all_characters,
    all_containers,
    all_states,
    n_samples,
    question_type="belief_question",
):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        template = STORY_TEMPLATES["templates"][template_idx]
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        new_states = random.sample(all_states[template["state_type"]], 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states[template["state_type"]], 2)

        sample = Sample(
            template_idx=template_idx,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=new_states,
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = random.choice([0, 1])

        if question_type == "belief_question":
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=1 ^ random_choice,
                question_type=question_type,
            )
            corrupt = corrupt_dataset.__getitem__(
                idx,
                set_container=1 ^ random_choice,
                set_character=1 ^ random_choice,
                question_type=question_type,
            )

        else:
            clean = clean_dataset.__getitem__(
                idx, set_container=random_choice, question_type=question_type
            )
            corrupt = corrupt_dataset.__getitem__(
                idx, set_container=1 ^ random_choice, question_type=question_type
            )

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1 ^ random_choice],
            }
        )

    return samples


def get_source_info(
    STORY_TEMPLATES,
    all_characters,
    all_containers,
    all_states,
    n_samples,
    question_type="belief_question",
):
    clean_configs, corrupt_configs = [], []
    samples = []

    for idx in range(n_samples):
        template_idx = 2
        template = STORY_TEMPLATES["templates"][template_idx]
        characters = random.sample(all_characters, 2)
        containers = random.sample(all_containers[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        sample = Sample(
            template_idx=template_idx,
            characters=characters,
            containers=containers,
            states=states,
        )
        clean_configs.append(sample)

        new_states = random.sample(all_states[template["state_type"]], 2)
        new_characters = random.sample(all_characters, 2)
        new_containers = random.sample(all_containers[template["container_type"]], 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states[template["state_type"]], 2)
        while new_characters[0] in characters or new_characters[1] in characters:
            new_characters = random.sample(all_characters, 2)
        while new_containers[0] in containers or new_containers[1] in containers:
            new_containers = random.sample(
                all_containers[template["container_type"]], 2
            )

        sample = Sample(
            template_idx=template_idx,
            characters=list(reversed(characters)),
            containers=list(reversed(containers)),
            states=list(reversed(states)),
        )
        corrupt_configs.append(sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        random_choice = random.choice([0, 1])

        if question_type == "belief_question":
            clean = clean_dataset.__getitem__(
                idx,
                set_container=random_choice,
                set_character=random_choice,
                question_type=question_type,
            )
            corrupt = corrupt_dataset.__getitem__(
                idx,
                set_container=1 ^ random_choice,
                set_character=1 ^ random_choice,
                question_type=question_type,
            )

        else:
            clean = clean_dataset.__getitem__(
                idx, set_container=random_choice, question_type=question_type
            )
            corrupt = corrupt_dataset.__getitem__(
                idx, set_container=1 ^ random_choice, question_type=question_type
            )

        samples.append(
            {
                "clean_characters": clean["characters"],
                "clean_objects": clean["objects"],
                "clean_states": clean["states"],
                "clean_story": clean["story"],
                "clean_question": clean["question"],
                "clean_prompt": clean["prompt"],
                "clean_ans": clean["target"],
                "corrupt_characters": corrupt["characters"],
                "corrupt_objects": corrupt["objects"],
                "corrupt_states": corrupt["states"],
                "corrupt_story": corrupt["story"],
                "corrupt_question": corrupt["question"],
                "corrupt_prompt": corrupt["prompt"],
                "corrupt_ans": corrupt["target"],
                "target": " " + clean_configs[idx].states[1 ^ random_choice],
            }
        )

    return samples
