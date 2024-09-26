<prefix>
Noor is a barista at a coffee shop. She wants to make a coffee for a customer. 
</prefix>

<context> 
The custormer asks for oat milk. 
</context>

Initial Event: 
<character_1> changes that state of <container_1> to <state_1>. And <character_1> changes that state of <container_2> to <state_2>

Causal Event:
<character_2> (unaware of the <context/>) changes the state of (<container1> to <state_2> or <container_2> to <state_1>)

Observe Event:
<character_1> either notices the Causal Event or not.

Question: 
Does <character> belive that <container> is in <state>?

story = prefix + context + Initial Event + Causal Event + Observe Event + Question