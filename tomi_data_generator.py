import json
import argparse
import random
from collections import defaultdict

random.seed(10)


class World:
    def __init__(self, world_file) -> None:
        with open(world_file, "r") as f:
            self.world = json.load(f)
        self.pointers = {k: -1 for k in self.world.keys()}

    def get_agent(self):
        self.pointers["agents"] += 1
        self.pointers["agents"] %= len(self.world["agents"])
        return self.world["agents"][self.pointers["agents"]]

    def get_location(self):
        self.pointers["locations"] += 1
        self.pointers["locations"] %= len(self.world["locations"])
        return self.world["locations"][self.pointers["locations"]]

    def get_object(self):
        self.pointers["objects"] += 1
        self.pointers["objects"] %= len(self.world["objects"])
        return self.world["objects"][self.pointers["objects"]]

    def get_containter(self):
        self.pointers["containers"] += 1
        self.pointers["containers"] %= len(self.world["containers"])
        return self.world["containers"][self.pointers["containers"]]

    def generate_sample(self, n_agents, m_agents, n_objects, m_objects):
        samples = []
        agents = [self.get_agent() for _ in range(n_agents)]
        objects = [self.get_object() for _ in range(n_objects)]
        containers = [self.get_containter() for _ in range(n_objects)]
        primary_location = self.get_location()
        redundant_location = self.get_location()
        world_state = defaultdict(dict)
        ground_truth = {}

        sample = ""

        # Defining agent's location
        num_redundant_location_agent = 0
        for idx in range(n_agents):
            if n_agents > 2 and num_redundant_location_agent <= 0:
                if random.random() < 0.25:
                    sample += f"{agents[idx]} entered the {redundant_location}.\n"
                    world_state[agents[idx]] = {
                        "location": redundant_location,
                        "beliefs": {},
                    }
                    num_redundant_location_agent += 1
                    continue

            sample += f"{agents[idx]} entered the {primary_location}.\n"
            world_state[agents[idx]] = {"location": primary_location, "beliefs": {}}

        # Defining object's location
        for idx in range(n_objects):
            sample += f"The {objects[idx]} is in the {containers[idx]}.\n"
            sample += f"The {containers[idx]} is in the {primary_location}.\n"

            # Define each agent's own belief as well as other agents' beliefs
            for agent_1 in world_state.keys():
                for agent_2 in world_state.keys():
                    if agent_2 not in world_state[agent_1]["beliefs"]:
                        world_state[agent_1]["beliefs"][agent_2] = {}

                    if (
                        world_state[agent_1]["location"] == primary_location
                        and world_state[agent_2]["location"] == primary_location
                    ):
                        world_state[agent_1]["beliefs"][agent_2][objects[idx]] = (
                            containers[idx]
                        )
                    else:
                        world_state[agent_1]["beliefs"][agent_2][
                            objects[idx]
                        ] = "Unknown"

            ground_truth[objects[idx]] = {
                "original": containers[idx],
                "current": containers[idx],
            }

        # Agents leave the primary location
        agents_in_primary_location = []
        for agent in world_state.keys():
            if world_state[agent]["location"] == primary_location:
                agents_in_primary_location.append(agent)

        # Randomly select m_agents to leave the primary location
        if m_agents <= len(agents_in_primary_location):
            agents_to_leave_primary_location = random.sample(
                agents_in_primary_location, m_agents
            )
        else:
            agents_to_leave_primary_location = random.sample(
                agents_in_primary_location, len(agents_in_primary_location) - 1
            )

        for agent in agents_to_leave_primary_location:
            sample += f"{agent} left the {primary_location}.\n"
            world_state[agent]["location"] = redundant_location

        # Agents move the objects to different containers
        for idx in range(m_objects):
            # Randomly select an agent whose "location" is primary location
            try:
                agent_in_location = random.choice(
                    [
                        agent
                        for agent in world_state.keys()
                        if world_state[agent]["location"] == primary_location
                    ]
                )
            except:
                print("Exception occurred!")
                continue

            sample += f"{agent_in_location} moved the {objects[idx]} to the {containers[(idx+1)%len(containers)]}.\n"
            ground_truth[objects[idx]]["current"] = containers[
                (idx + 1) % len(containers)
            ]

            # Update agent_in_location's own belief as well as other agents' beliefs.
            # An agent's belief about other agents' beliefs is updated only if both are present in the primary location.
            for agent_1 in world_state.keys():
                for agent_2 in world_state.keys():
                    if (
                        world_state[agent_1]["location"] == primary_location
                        and world_state[agent_2]["location"] == primary_location
                    ):
                        world_state[agent_1]["beliefs"][agent_2][objects[idx]] = (
                            containers[(idx + 1) % len(containers)]
                        )

        # Add redundant text to the sample
        sentences = sample.split("\n")
        redundant_sentence = (
            f"{random.choice(agents)} likes the {random.choice(objects)}."
        )
        sentences.insert(random.randint(0, len(sentences)), redundant_sentence)
        sample = "\n".join(sentences).replace("\n\n", "\n")

        # Generate questions
        for idx in range(len(objects)):
            samples.append(
                {
                    "context": sample,
                    "question": f"Where was the {objects[idx]} at the beginning?",
                    "answer": ground_truth[objects[idx]]["original"],
                }
            )
            samples.append(
                {
                    "context": sample,
                    "question": f"Where is the {objects[idx]} really?",
                    "answer": ground_truth[objects[idx]]["current"],
                }
            )

        for agent_1 in agents:
            for object in objects:
                samples.append(
                    {
                        "context": sample,
                        "question": f"Where will {agent_1} look for the {object}?",
                        "answer": world_state[agent_1]["beliefs"][agent_1][object],
                    }
                )

        for agent_1 in agents:
            for agent_2 in agents:
                if agent_1 != agent_2:
                    for object in objects:
                        samples.append(
                            {
                                "context": sample,
                                "question": f"Where does {agent_1} think that {agent_2} searches for the {object}?",
                                "answer": world_state[agent_1]["beliefs"][agent_2][
                                    object
                                ],
                            }
                        )

        return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_file", type=str, default="world.json")
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--n_objects", type=int, default=5)
    args = parser.parse_args()

    samples = []
    world = World(args.world_file)
    for n_agents in range(2, args.n_agents + 1):
        for m_agents in range(0, n_agents):
            for n_objects in range(1, args.n_objects + 1):
                for m_objects in range(0, n_objects):
                    samples += world.generate_sample(
                        n_agents=n_agents,
                        m_agents=m_agents,
                        n_objects=n_objects,
                        m_objects=m_objects,
                    )
                    print(
                        f"Generated {len(samples)} samples for n_agents={n_agents}, m_agents={m_agents}, n_objects={n_objects}, m_objects={m_objects}"
                    )

    with open("generated_tomi.json", "a") as f:
        json.dump(samples, f, indent=4)


if __name__ == "__main__":
    main()
