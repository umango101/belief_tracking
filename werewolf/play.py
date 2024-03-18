import random
import os
import sys
import torch
import fire

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import (
    load_model_tokenzier,
    compute_role_description,
    NewLineStoppingCriteria,
    compute_final_roles,
    ask_mental_state_questions,
)

seed = 10
random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
player_names = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
]
player_roles = ["Villager", "Werewolf", "Troublemaker"]

with open("/data/nikhil_prakash/mind/werewolf/game_description.txt", "r") as f:
    game_description = f.read()

# Delete all .txt and .cdv files in the conversations folder
for file in os.listdir("/data/nikhil_prakash/mind/werewolf/conversations"):
    if file.endswith(".txt") or file.endswith(".csv"):
        os.remove(os.path.join("/data/nikhil_prakash/mind/werewolf/conversations", file))


def play_werewolf(
    model_name: str = "daryl149/llama-2-70b-chat-hf",
    precision: str = "int4",
    n_games: int = 10,
    n_players: int = 3,
    n_rounds: int = 5,
):
    model, tokenizer = load_model_tokenzier(model_name, precision, device)

    for game_idx in range(1):
        players = [
            {"name": name, "role": role}
            for name, role in zip(random.sample(player_names, n_players), player_roles * n_players)
        ]
        players = compute_final_roles(players)
        players = compute_role_description(players)

        stopping_criteria = NewLineStoppingCriteria(tokenizer)

        with torch.no_grad():
            for round_idx in range(n_rounds):
                for player_idx in range(n_players):
                    player = players[player_idx]
                    prompt = f"{game_description}\n\n{player['role_description']}\n\n"

                    if os.path.exists(
                        f"/data/nikhil_prakash/mind/werewolf/conversations/{game_idx}.txt"
                    ):
                        with open(
                            f"/data/nikhil_prakash/mind/werewolf/conversations/{game_idx}.txt", "r"
                        ) as f:
                            conversation = f.read()
                    else:
                        conversation = ""

                    prompt += f"DAY PHASE:\n{conversation}\n{player['name']} says:"
                    print(prompt)

                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        num_return_sequences=1,
                        temperature=0.00,
                        do_sample=False,
                        stopping_criteria=[stopping_criteria],
                        early_stopping=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    with open(
                        f"/data/nikhil_prakash/mind/werewolf/conversations/{game_idx}.txt", "a+"
                    ) as f:
                        f.write(f"{player['name']} says: {response[len(prompt) :]}\n")

            own_vote, other_votes = ask_mental_state_questions(
                model, tokenizer, device, players, game_description, game_idx, conversation
            )
            # Concatenate the votes of the other players
            other_votes = [f"{player['name']}:{vote}" for player, vote in zip(players, other_votes)]

            # Create a csv to store the game results
            with open(f"/data/nikhil_prakash/mind/werewolf/conversations/{game_idx}.csv", "w") as f:
                f.write("Name,Role,Own Vote,Other Votes\n")
                for player in players:
                    f.write(f"{player['name']},{player['role']},{own_vote},{other_votes}\n")


if __name__ == "__main__":
    fire.Fire(play_werewolf)
