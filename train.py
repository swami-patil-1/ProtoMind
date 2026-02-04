import torch
import torch.nn as nn
import random

from env import SimpleWorld
from brain import NewbornBrain

env = SimpleWorld()
brain = NewbornBrain()

optimizer = torch.optim.Adam(brain.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

state = torch.tensor(env.reset())

for step in range(10000):

    # Preference-biased exploration
    bias = brain.action_bias.detach().numpy()
    probs = bias - bias.min() + 0.01  # make positive
    probs = probs / probs.sum()

    if random.random() < brain.curiosity:
        action = random.randint(0, 4)
    else:
        action = random.choices(range(5), weights=probs)[0]

    next_state_np, reward, pain = env.step(action)
    next_state = torch.tensor(next_state_np)

    pred_next_state = brain(state, action)
    loss = loss_fn(pred_next_state, next_state)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    brain.update_internal_state(
        reward, pain, loss, state, action
    )

    state = next_state

    if step % 300 == 0:
        print(
            f"Step {step} | "
            f"Loss {loss.item():.4f} | "
            f"Pleasure {brain.pleasure:.2f} | "
            f"Stress {brain.stress:.2f} | "
            f"Curiosity {brain.curiosity:.2f} | "
            f"Memories {len(brain.salient_memories)}"
        )
