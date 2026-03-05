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

# Metrics
loss_hist = []
pleasure_hist = []
stress_hist = []
curiosity_hist = []
exp_p_hist = []
exp_s_hist = []
memory_hist = []

for step in range(2000):

    # Exploration vs prediction
    if random.random() < 0.2:
        action = brain.get_reflex_action()
    else:
        action = brain.choose_action_with_prediction(state)

    next_state_np, pleasure, stress, source = env.step(action)
    next_state = torch.tensor(next_state_np, dtype=torch.float32)

    # World model prediction
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

    # Logging
    loss_hist.append(loss.item())
    pleasure_hist.append(brain.pleasure)
    stress_hist.append(brain.stress)
    curiosity_hist.append(brain.curiosity)
    exp_p_hist.append(brain.expected_pleasure)
    exp_s_hist.append(brain.expected_stress)
    memory_hist.append(len(brain.long_term_memory))

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

# -------- Visualization --------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(pleasure_hist, label="Pleasure")
plt.plot(stress_hist, label="Stress")
plt.plot(curiosity_hist, label="Curiosity")
plt.title("Emotion Dynamics")
plt.legend()

plt.subplot(1,3,2)
plt.plot(loss_hist, label="Prediction Loss")
plt.plot(curiosity_hist, label="Curiosity")
plt.title("Learning")

plt.legend()

plt.subplot(1,3,3)
plt.plot(exp_p_hist, label="Expected Pleasure")
plt.plot(exp_s_hist, label="Expected Stress")
plt.plot(memory_hist, label="Memory Size")
plt.title("Memory & Anticipation")
plt.legend()

plt.tight_layout()
plt.show()