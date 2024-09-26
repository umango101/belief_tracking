@dataclass
class Event:
    container: str
    state: str


@dataclass
class Question:
    character: str
    container: str
    state: str


@dataclass
class Story:
    prefix: str
    context: str
    initial_event: list[Event]
    causal_event: Optional[Event]
    observe_event: bool
    questoin: Question


def solve_TOM(
    story: Story,
) -> Literal["yes", "no"]:

    marker_assign = {}
    # marker_assign[e1] = i1 is equivant to rep(e1) = assign(rep(e1), i1)

    true_belief = {}
    # true_belief[k] = s is equivant to rep(k) = bind(rep(k), id(s))
    # where id(s) = marker_assign[s]

    # prefix
    marker_assign["<character_1>"] = c1

    # context
    marker_assign["<state_1>"] = o1

    # initial event
    marker_assign["<container_1>"] = k1
    marker_assign["<container_2>"] = k2
    marker_assign["<state_2>"] = o2

    # This experiment assumes that the containers binding ID is getting changed based on the object
    # But an equivant code could be written where the objects binding ID is getting changed based on the containers ID
    # If both are happening, then it will be very meesy
    true_belief = {
        c1: o1,
        c2: o2,
    }

    # causal event
    if story.causal_event is None:
        character_belief = true_belief.clone()

    else:
        # causal event happnes
        true_belief[k_event] = o_event
        if story.obser_event:
            character_belief[k_event] = o_event
        # else: => character belief does not change

    answer = (
        "yes"
        if character_belief[marker_assign[question.container]]
        == marker_assign[question.state]
        else "no"
    )

    return answer


# Experiemnts
