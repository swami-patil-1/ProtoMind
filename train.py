import torch
import torch.nn as nn
import random

from env import SimpleWorld
from brain import NewbornBrain


env = SimpleWorld()
brain = NewbornBrain()

optimizer = torch.optim.Adam(brain.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

state = torch.tensor(env.reset(), dtype=torch.float32)

for step in range(10000):

    action = random.randint(0, brain.action_dim - 1)

    next_state_np, pleasure, stress = env.step(action)
    next_state = torch.tensor(next_state_np, dtype=torch.float32)

    pred_next_state = brain(state, action)
    loss = loss_fn(pred_next_state, next_state)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    brain.update_internal_state(
        pleasure=pleasure,
        stress=stress,
        prediction_error=loss,
        state=state,
        action=action,
        step=step
    )

    state = next_state

    if step % 50 == 0:
        print(
            f"Step {step:05d} | "
            f"Loss {loss.item():.4f} | "
            f"P {brain.pleasure:.2f} | "
            f"S {brain.stress:.2f} | "
            f"C {brain.curiosity:.2f} | "
            f"ExpP {brain.expected_pleasure:.2f} | "
            f"ExpS {brain.expected_stress:.2f} | "
            f"LTM {len(brain.long_term_memory)}"
        )
