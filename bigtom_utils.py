import torch


def get_ques_start_token_idx(tokenizer, prompt):
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze()
    corrolary_token = tokenizer.encode(":", return_tensors="pt").squeeze()[-1].item()
    ques_start_idx = (input_tokens == corrolary_token).nonzero()[2].item()

    return ques_start_idx - 1


def get_visitibility_sent_start_idx(tokenizer, prompt):
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze()

    story_start_idx = (input_tokens == 18422).nonzero()[0].item()
    period_idx = (input_tokens == 13).nonzero(as_tuple=True)[0]
    period_idx = period_idx[period_idx > story_start_idx]

    return period_idx[-1] + 1


def get_prompt_token_len(tokenizer, prompt):
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze()
    return len(input_tokens)


def check_pred(model, pred, target, verbose=False):
    prompt = f"Instruction: Check if the following ground truth and prediction are same or different. If they are the same, then predict 'Yes', else 'No' \n\nGround truth: {target}\nPrediction: {pred}\nAnswer:"

    if verbose:
        print(prompt)

    with torch.no_grad():
        with model.generate(
            prompt,
            max_new_tokens=2,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=model.tokenizer.pad_token_id,
        ):
            out = model.generator.output.save()

    prompt_len = get_prompt_token_len(model.tokenizer, prompt)

    return model.tokenizer.decode(out[0][prompt_len:-1]).strip()
