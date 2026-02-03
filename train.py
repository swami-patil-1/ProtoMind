import torch
import torch.nn as nn
import random

from env import SimpleWorld
from brain import NewbornBrain

env = SimpleWorld()
brain = NewbornBrain()

optimizer = torch.optim.Adam(brain.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

state = env.reset()
state = torch.tensor(state)

for step in range(3000):
    action = random.randint(0, 4)

    next_state = env.step(action)
    next_state = torch.tensor(next_state)

    pred_next_state = brain(state, action)

    loss = loss_fn(pred_next_state, next_state)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    state = next_state

    if step % 200 == 0:
        print(f"Step {step} | Prediction error: {loss.item():.4f}")
