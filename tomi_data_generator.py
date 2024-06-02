import json
import argparse
import random
from collections import defaultdict

random.seed(10)


class World:
    def __init__(self, world_file, add_redundant_sentence=True) -> None:
        with open(world_file, "r") as f:
            self.world = json.load(f)
        self.pointers = {k: -1 for k in self.world.keys()}
        self.add_redundant_sentence = add_redundant_sentence

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

    def agent_entry(self, sample, agent, primary_location, world_state, ground_truth):
        sample += f"{agent} entered the {primary_location}.\n"
        world_state[agent] = {
            "location": defaultdict(dict),
            "beliefs": defaultdict(dict),
        }
        world_state[agent]["location"][agent] = primary_location

        for agent_2 in world_state.keys():
            # If both agent and agent_2 are in the same location,
            # then their beliefs about each other's location is the same
            if world_state[agent_2]["location"][agent_2] == primary_location:
                world_state[agent]["location"][agent_2] = primary_location
                world_state[agent_2]["location"][agent] = primary_location
            else:
                world_state[agent]["location"][agent_2] = "Unknown"
                world_state[agent_2]["location"][agent] = "Unknown"

        # Define each agent's own belief as well as other agents' beliefs
        # based on objects current containers in the primary location
        for agent_2 in world_state.keys():
            if world_state[agent_2]["location"][agent_2] == primary_location:
                for object in ground_truth.keys():
                    # All objects are in the primary location
                    world_state[agent]["beliefs"][agent_2][object] = ground_truth[
                        object
                    ]["current"]
                    world_state[agent_2]["beliefs"][agent][object] = ground_truth[
                        object
                    ]["current"]

            else:
                for object in ground_truth.keys():
                    world_state[agent]["beliefs"][agent_2][object] = "Unknown"
                    world_state[agent_2]["beliefs"][agent][object] = "Unknown"

        return sample

    def object_definition(
        self, sample, object, container, primary_location, world_state, ground_truth
    ):
        sample += f"The {object} is in the {container}.\n"
        sample += f"The {container} is in the {primary_location}.\n"

        # Define each agent's own belief as well as other agents' beliefs
        for agent_1 in world_state.keys():
            for agent_2 in world_state.keys():
                # If an agent is defined in world_state, then they have entered the
                # primary location and hence know the initial location of all objects
                if agent_1 == agent_2:
                    world_state[agent_1]["beliefs"][agent_2][object] = container
                # An agent's belief about other agents' beliefs is updated only
                # if both are present in the primary location.
                elif (
                    world_state[agent_1]["location"][agent_1] == primary_location
                    and world_state[agent_2]["location"][agent_2] == primary_location
                ):
                    world_state[agent_1]["beliefs"][agent_2][object] = container
                else:
                    world_state[agent_1]["beliefs"][agent_2][object] = "Unknown"

        ground_truth[object] = {
            "original": container,
            "current": container,
        }

        return sample

    def agent_exit(self, sample, agent, primary_location, world_state):
        sample += f"{agent} left the {primary_location}.\n"
        world_state[agent]["location"][agent] = "Unknown"

        # If anyone is present in primary location, then their beliefs about
        # the exiting agent's location is updated to "Unknown"
        for agent_2 in world_state.keys():
            if world_state[agent_2]["location"][agent_2] == primary_location:
                world_state[agent_2]["location"][agent] = "Unknown"

        return sample

    def object_movement(
        self,
        sample,
        agent,
        object,
        new_container,
        primary_location,
        world_state,
        ground_truth,
    ):
        sentence = f"{agent} moved the {object} to the {new_container}."
        if sentence not in sample:
            sample += f"{agent} moved the {object} to the {new_container}.\n"
            ground_truth[object]["current"] = new_container

            # Update agent's own belief as well as other agents' beliefs.
            # An agent's belief about other agents' beliefs is updated only
            # if both are present in the primary location.
            for agent_1 in world_state.keys():
                for agent_2 in world_state.keys():
                    if (
                        world_state[agent_1]["location"][agent_1] == primary_location
                        and world_state[agent_2]["location"][agent_2]
                        == primary_location
                    ):
                        world_state[agent_1]["beliefs"][agent_2][object] = new_container
                        world_state[agent_2]["beliefs"][agent_1][object] = new_container

        return sample

    def generate_sample(self, n_agents, m_agents, n_objects, m_objects, max_movements):
        AGENTS = [self.get_agent() for _ in range(n_agents)]
        OBJECTS = [self.get_object() for _ in range(n_objects)]
        CONTAINERS = [self.get_containter() for _ in range(n_objects)]
        PRIMARY_LOC = self.get_location()

        world_state = defaultdict(dict)
        ground_truth = {}
        samples = []
        sample = ""

        while (
            (n_agents != 0) or (m_agents != 0) or (n_objects != 0) or (m_objects != 0)
        ):
            # 0 - 0.25: Agent entry
            # 0.25 - 0.5: Object definition
            # 0.5 - 0.75: Object movement
            # 0.75 - 1: Agent Exit
            if random.random() < 0.25:
                if n_agents != 0:
                    sample = self.agent_entry(
                        sample,
                        AGENTS[n_agents - 1],
                        PRIMARY_LOC,
                        world_state,
                        ground_truth,
                    )
                    n_agents -= 1

            elif 0.25 <= random.random() < 0.5:
                if n_objects != 0:
                    sample = self.object_definition(
                        sample,
                        OBJECTS[n_objects - 1],
                        CONTAINERS[n_objects - 1],
                        PRIMARY_LOC,
                        world_state,
                        ground_truth,
                    )
                    n_objects -= 1

            elif 0.5 <= random.random() < 0.75:
                if m_objects != 0:
                    # Agents in primary location
                    agents_in_primary_location = [
                        agent
                        for agent in world_state.keys()
                        if world_state[agent]["location"][agent] == PRIMARY_LOC
                    ]

                    # If there is atleast one agent in the primary location and
                    # atleast one object is defined, then move the object
                    if len(agents_in_primary_location) != 0 and len(ground_truth) != 0:
                        # Randomly select an agent whose "location" is primary location to move the objects
                        agent_to_move_obj = random.choice(agents_in_primary_location)

                        # Randomly select an already defined object to move
                        object_to_move = random.choice(list(ground_truth.keys()))

                        # Select n_movements for the object
                        n_movements = random.randint(1, max_movements)

                        # Randomly select a container to move the object which is not the current container
                        new_container = ground_truth[object_to_move]["current"]
                        while new_container == ground_truth[object_to_move]["current"]:
                            new_container = random.choice(CONTAINERS)

                        for _ in range(n_movements):
                            sample = self.object_movement(
                                sample,
                                agent_to_move_obj,
                                object_to_move,
                                new_container,
                                PRIMARY_LOC,
                                world_state,
                                ground_truth,
                            )

                        m_objects -= 1

            else:
                if m_agents != 0:
                    # Agents in primary location
                    agents_in_primary_location = [
                        agent
                        for agent in world_state.keys()
                        if world_state[agent]["location"][agent] == PRIMARY_LOC
                    ]

                    if len(agents_in_primary_location) != 0:
                        agent_to_exit = random.choice(agents_in_primary_location)

                        sample = self.agent_exit(
                            sample, agent_to_exit, PRIMARY_LOC, world_state
                        )

                        m_agents -= 1

        if self.add_redundant_sentence:
            sentences = sample.split("\n")
            redundant_sentence = (
                f"{random.choice(AGENTS)} likes the {random.choice(OBJECTS)}."
            )
            sentences.insert(random.randint(0, len(sentences)), redundant_sentence)
            sample = "\n".join(sentences).replace("\n\n", "\n")

        # Generate questions
        for idx in range(len(OBJECTS)):
            samples.append(
                {
                    "context": sample,
                    "question": f"Where was the {OBJECTS[idx]} at the beginning?",
                    "answer": ground_truth[OBJECTS[idx]]["original"],
                }
            )
            samples.append(
                {
                    "context": sample,
                    "question": f"Where is the {OBJECTS[idx]} really?",
                    "answer": ground_truth[OBJECTS[idx]]["current"],
                }
            )

        for agent_1 in AGENTS:
            for object in OBJECTS:
                samples.append(
                    {
                        "context": sample,
                        "question": f"Where will {agent_1} look for the {object}?",
                        "answer": world_state[agent_1]["beliefs"][agent_1][object],
                    }
                )

        for agent_1 in AGENTS:
            for agent_2 in AGENTS:
                if agent_1 != agent_2:
                    for object in OBJECTS:
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
    parser.add_argument("--n_iterations", type=int, default=1)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--n_objects", type=int, default=5)
    parser.add_argument("--max_movements", type=int, default=1)
    parser.add_argument("--add_redundant_sentence", action="store_false")
    args = parser.parse_args()

    print(args)

    samples = []
    world = World(args.world_file)
    for _ in range(args.n_iterations):
        for n_agents in range(2, args.n_agents + 1):
            for m_agents in range(0, n_agents):
                for n_objects in range(1, args.n_objects + 1):
                    for m_objects in range(0, n_objects):
                        samples += world.generate_sample(
                            n_agents=n_agents,
                            m_agents=m_agents,
                            n_objects=n_objects,
                            m_objects=m_objects,
                            max_movements=args.max_movements,
                        )
                        print(
                            f"Generated {len(samples)} samples for n_agents={n_agents}, m_agents={m_agents}, n_objects={n_objects}, m_objects={m_objects}"
                        )

    with open("generated_tomi.json", "a") as f:
        json.dump(samples, f, indent=4)


if __name__ == "__main__":
    main()
