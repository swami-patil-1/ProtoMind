import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

from env import SimpleWorld
from brain import NewbornBrain


env = SimpleWorld()
brain = NewbornBrain()

optimizer = torch.optim.Adam(brain.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

state = torch.tensor(env.reset(), dtype=torch.float32)

loss_hist = []
pleasure_hist = []
stress_hist = []
curiosity_hist = []
memory_hist = []

for step in range(2000):

    # exploration vs prediction
    if random.random() < 0.2:
        action = brain.get_reflex_action()
    else:
        action = brain.choose_action_with_prediction(state)

    next_state_np, pleasure, stress, source = env.step(action)
    next_state = torch.tensor(next_state_np, dtype=torch.float32)

    pred_next_state = brain(state, action)
    loss = loss_fn(pred_next_state, next_state)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    brain.update_internal_state(
        pleasure=pleasure,
        stress=stress,
        source=source,
        prediction_error=loss,
        state=state,
        action=action,
        step=step
    )

    state = next_state

    loss_hist.append(loss.item())
    pleasure_hist.append(brain.pleasure)
    stress_hist.append(brain.stress)
    curiosity_hist.append(brain.curiosity)
    memory_hist.append(len(brain.long_term_memory))

    if step % 50 == 0:
        print(
            f"Step {step:05d} | "
            f"Loss {loss.item():.4f} | "
            f"P {brain.pleasure:.2f} | "
            f"S {brain.stress:.2f} | "
            f"C {brain.curiosity:.2f} | "
            f"LTM {len(brain.long_term_memory)}"
        )


# -------- Graphs --------

plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
plt.plot(pleasure_hist,label="Pleasure")
plt.plot(stress_hist,label="Stress")
plt.title("Emotion Dynamics")
plt.legend()

plt.subplot(1,3,2)
plt.plot(curiosity_hist,label="Curiosity")
plt.title("Curiosity")
plt.legend()

plt.subplot(1,3,3)
plt.plot(loss_hist,label="Prediction Loss")
plt.title("World Model Learning")
plt.legend()

plt.tight_layout()
plt.show()


# -------- Spatial Heatmaps --------

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(brain.pleasure_map, cmap="Greens")
plt.colorbar()
plt.title("Pleasure Map (Comfort Zones)")

plt.subplot(1,2,2)
plt.imshow(brain.stress_map, cmap="Reds")
plt.colorbar()
plt.title("Stress Map (Danger Zones)")

plt.tight_layout()
plt.show()