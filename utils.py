import random

from src.dataset import Dataset, Sample

random.seed(10)


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


def get_unidirectional_visibility_exps(
    STORY_TEMPLATES,
    all_characters,
    all_objects,
    all_states,
    n_samples,
    additional_characs=False,
):
    clean_configs, corrupt_configs, samples = [], [], []

    for idx in range(n_samples):
        template = STORY_TEMPLATES["templates"][0]
        characters = (
            random.sample(all_characters, 2)
            if not additional_characs
            else random.sample(all_characters, 4)
        )
        containers = random.sample(all_objects[template["container_type"]], 2)
        states = random.sample(all_states[template["state_type"]], 2)

        no_vis_sample = Sample(
            template_idx=0 if not additional_characs else 3,
            characters=characters,
            containers=containers,
            states=states,
        )

        new_states = random.sample(all_states[template["state_type"]], 2)
        new_characters = random.sample(all_characters, 2)
        new_containers = random.sample(all_objects[template["container_type"]], 2)
        while new_states[0] in states or new_states[1] in states:
            new_states = random.sample(all_states[template["state_type"]], 2)
        while new_characters[0] in characters or new_characters[1] in characters:
            new_characters = random.sample(all_characters, 2)
        while new_containers[0] in containers or new_containers[1] in containers:
            new_containers = random.sample(all_objects[template["container_type"]], 2)

        vis_sample = Sample(
            template_idx=1,
            characters=new_characters,
            containers=new_containers,
            states=new_states,
        )

        clean_configs.append(no_vis_sample)
        corrupt_configs.append(vis_sample)

    clean_dataset = Dataset(clean_configs)
    corrupt_dataset = Dataset(corrupt_configs)

    for idx in range(n_samples):
        clean = clean_dataset.__getitem__(
            idx,
            set_character=0,
            set_container=1,
            question_type="belief_question",
        )
        corrupt = corrupt_dataset.__getitem__(
            idx,
            set_character=0,
            set_container=1,
            question_type="belief_question",
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
                "target": " " + clean_configs[idx].states[1],
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


def get_tb_fb_answer(tb_ans, fb_ans):
    diff_idx = 0
    for i, (v, j) in enumerate(zip(tb_ans, fb_ans)):
        if v != j:
            diff_idx = i
            break

    tb_ans = " ".join(tb_ans[diff_idx:])[:-1]
    fb_ans = " ".join(fb_ans[diff_idx:])[:-1]

    return tb_ans, fb_ans


def get_bigtom_samples(df_false, df_true, n_samples, belief_type="false_belief"):
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    samples = []
    instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

    for idx in range(n_samples):
        tb_ans = true_stories[idx]["answer"].split()
        fb_ans = false_stories[idx]["answer"].split()

        # Find the index of first word which is different in both answers
        diff_idx = 0
        for i, (v, j) in enumerate(zip(tb_ans, fb_ans)):
            if v != j:
                diff_idx = i
                break

        tb_ans = " ".join(tb_ans[diff_idx:])[:-1]
        fb_ans = " ".join(fb_ans[diff_idx:])[:-1]

        if belief_type == "true_belief":
            question = true_stories[idx]["question"]
            tb_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[idx]['story']}\nQuestion: {question}\nAnswer:"
            samples.append({"prompt": tb_prompt, "answer": tb_ans})
        else:
            question = false_stories[idx]["question"]
            fb_prompt = f"Instructions: {instruction}\n\nStory: {false_stories[idx]['story']}\nQuestion: {question}\nAnswer:"
            samples.append({"prompt": fb_prompt, "answer": fb_ans})

    return samples


def get_bigtom_value_fetcher_exps(
    df_false,
    df_true,
    n_samples,
):
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    samples = []
    instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

    for idx in range(n_samples):
        tb_ans = true_stories[idx]["answer"].split()
        fb_ans = false_stories[idx]["answer"].split()
        tb_ans, fb_ans = get_tb_fb_answer(tb_ans, fb_ans)

        question = true_stories[idx]["question"]
        org_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[idx]['story']}\nQuestion: {question}\nAnswer:"
        org_ans = tb_ans

        # Sample a different story
        random_sample_idx = random.choice(range(len(true_stories)))
        while random_sample_idx == idx:
            random_sample_idx = random.choice(range(len(true_stories)))

        tb_ans = true_stories[random_sample_idx]["answer"].split()
        fb_ans = false_stories[random_sample_idx]["answer"].split()
        tb_ans, fb_ans = get_tb_fb_answer(tb_ans, fb_ans)

        question = true_stories[random_sample_idx]["question"]
        alt_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[random_sample_idx]['story']}\nQuestion: {question}\nAnswer:"
        alt_ans = tb_ans

        samples.append(
            {
                "org_story": true_stories[idx]["story"],
                "org_question": true_stories[idx]["question"],
                "org_prompt": org_prompt,
                "org_ans": org_ans,
                "alt_story": true_stories[random_sample_idx]["story"],
                "alt_question": true_stories[random_sample_idx]["question"],
                "alt_prompt": alt_prompt,
                "alt_ans": alt_ans,
                "target": alt_ans,
            }
        )

    return samples


def get_bigtom_answer_state_exps(
    df_false,
    df_true,
    n_samples,
):
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    samples = []
    instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

    for idx in range(n_samples):
        org_tb_ans = true_stories[idx]["answer"].split()
        org_fb_ans = false_stories[idx]["answer"].split()
        org_tb_ans, org_fb_ans = get_tb_fb_answer(org_tb_ans, org_fb_ans)

        question = true_stories[idx]["question"]
        org_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[idx]['story']}\nQuestion: {question}\nAnswer:"
        org_ans = org_tb_ans

        # Sample a different story
        random_sample_idx = random.choice(range(len(false_stories)))
        while random_sample_idx == idx:
            random_sample_idx = random.choice(range(len(false_stories)))

        alt_tb_ans = true_stories[random_sample_idx]["answer"].split()
        alt_fb_ans = false_stories[random_sample_idx]["answer"].split()
        alt_tb_ans, alt_fb_ans = get_tb_fb_answer(alt_tb_ans, alt_fb_ans)

        question = false_stories[random_sample_idx]["question"]
        alt_prompt = f"Instructions: {instruction}\n\nStory: {false_stories[random_sample_idx]['story']}\nQuestion: {question}\nAnswer:"
        alt_ans = alt_fb_ans

        samples.append(
            {
                "org_story": false_stories[idx]["story"],
                "org_question": false_stories[idx]["question"],
                "org_prompt": org_prompt,
                "org_ans": org_ans,
                "alt_story": true_stories[random_sample_idx]["story"],
                "alt_question": true_stories[random_sample_idx]["question"],
                "alt_prompt": alt_prompt,
                "alt_ans": alt_ans,
                "target": org_fb_ans,
            }
        )

    return samples


def get_bigtom_query_charac(df_false, df_true, n_samples):
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    samples = []
    instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

    for idx in range(n_samples):
        alt_tb_ans = true_stories[idx]["answer"].split()
        alt_fb_ans = false_stories[idx]["answer"].split()
        alt_tb_ans, alt_fb_ans = get_tb_fb_answer(alt_tb_ans, alt_fb_ans)

        alt_question = true_stories[idx]["question"]
        alt_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[idx]['story']}\nQuestion: {alt_question}\nAnswer:"
        alt_ans = alt_tb_ans

        # Sample a different story
        random_sample_idx = random.choice(range(len(true_stories)))
        while random_sample_idx == idx:
            random_sample_idx = random.choice(range(len(true_stories)))

        org_tb_ans = true_stories[random_sample_idx]["answer"].split()
        org_fb_ans = false_stories[random_sample_idx]["answer"].split()
        org_tb_ans, org_fb_ans = get_tb_fb_answer(org_tb_ans, org_fb_ans)

        org_question = true_stories[random_sample_idx]["question"]
        # Replace the second word in org_question with the second word in org_question
        org_question = org_question.split()
        org_question[1] = alt_question.split()[1]
        org_question = " ".join(org_question)
        org_prompt = f"Instructions: {instruction}\n\nStory: {true_stories[random_sample_idx]['story']}\nQuestion: {org_question}\nAnswer:"
        org_ans = "unknown"

        samples.append(
            {
                "org_story": true_stories[random_sample_idx]["story"],
                "org_question": org_question,
                "org_prompt": org_prompt,
                "org_ans": org_ans,
                "alt_story": true_stories[idx]["story"],
                "alt_question": alt_question,
                "alt_prompt": alt_prompt,
                "alt_ans": alt_ans,
                "target": org_tb_ans,
            }
        )

    return samples


def get_bigtom_visibility_exps(
    df_false,
    df_true,
    n_samples,
):
    true_stories, false_stories = [], []
    for i in range(len(df_true)):
        story = df_true.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_true.iloc[i]["answer"]
        distractor = df_true.iloc[i]["distractor"]
        true_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    for i in range(len(df_false)):
        story = df_false.iloc[i]["story"]
        question = df_true.iloc[i]["question"]
        answer = df_false.iloc[i]["answer"]
        distractor = df_false.iloc[i]["distractor"]
        false_stories.append(
            {
                "story": story,
                "question": question,
                "answer": answer,
                "distractor": distractor,
            }
        )

    samples = []
    instruction = "1. Track the belief of each character as described in the story. 2. A character's belief is formed only when they perform an action themselves or can observe the action taking place. 3. A character does not have any belief about the container or its content which they cannot observe directly. 4. To answer the question, predict only the final state of the queried container in fewest tokens possible, strictly based on the belief of the character, mentioned in the question. 5. Do not predict the entire sentence with character or container as the final output."

    for idx in range(n_samples):
        tb_ans = true_stories[idx]["answer"].split()
        fb_ans = false_stories[idx]["answer"].split()
        tb_ans, fb_ans = get_tb_fb_answer(tb_ans, fb_ans)

        alt_story = true_stories[idx]["story"]
        alt_question = true_stories[idx]["question"]
        alt_prompt = f"Instructions: {instruction}\n\nStory: {alt_story}\nQuestion: {alt_question}\nAnswer:"
        alt_ans = tb_ans

        org_story = false_stories[idx]["story"]
        org_question = false_stories[idx]["question"]
        org_prompt = f"Instructions: {instruction}\n\nStory: {org_story}\nQuestion: {org_question}\nAnswer:"
        org_ans = fb_ans

        samples.append(
            {
                "org_story": org_story,
                "org_question": org_question,
                "org_prompt": org_prompt,
                "org_ans": org_ans,
                "alt_story": alt_story,
                "alt_question": alt_question,
                "alt_prompt": alt_prompt,
                "alt_ans": alt_ans,
                "target": alt_ans,
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
