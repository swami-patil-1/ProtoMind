import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from env import RichWorld
from brain import NewbornBrain

NUM_STEPS = 2000
SEEDS     = [42, 43, 44, 45, 46]
HALF      = NUM_STEPS // 2

ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right", 4: "stay", 5: "signal"}


def run_episode(policy, seed, verbose=False):
    """Run one episode. policy ∈ {"predictive", "reflex"}.

    "predictive" = current 20/80 reflex/world-model mix.
    "reflex"     = always reflex (no learned-policy actions). World model still
                   trains and emotions still update — only action selection
                   ignores the world model. This is the no-learning baseline.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env   = RichWorld()
    brain = NewbornBrain()
    grid  = env.size
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.01)
    loss_fn   = nn.MSELoss()

    state     = torch.tensor(env.reset(), dtype=torch.float32)
    proximity = {}

    H = {
        "loss": [], "pleasure": [], "stress": [], "mood": [], "curiosity": [],
        "energy": [], "separation": [], "memory": [], "attention": [], "salience": [],
        "hab_mother": [], "hab_father": [], "hab_gmom": [],
        "rpe": [], "trust_mother": [], "trust_stranger": [],
        "action": [], "env": [], "source": [], "pos": [],
    }
    visit_early = np.zeros((grid, grid))
    visit_late  = np.zeros((grid, grid))

    for step in range(NUM_STEPS):
        if policy == "reflex" or random.random() < 0.2:
            action = brain.get_reflex_action()
        else:
            action = brain.choose_action_with_prediction(state, proximity=proximity)

        signal_active = (action == 5) and (brain.stress > 0.5)

        next_state_np, pleasure, stress, source, proximity, present_entities, time_of_day = \
            env.step(action, signal_active=signal_active)

        next_state = torch.tensor(next_state_np, dtype=torch.float32)

        pred_next_state = brain(state, action)
        loss = loss_fn(pred_next_state, next_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        brain.update_internal_state(
            pleasure=pleasure, stress=stress, source=source,
            prediction_error=loss, state=next_state, action=action, step=step,
            proximity=proximity, present_entities=present_entities, time_of_day=time_of_day,
        )

        state = next_state

        H["loss"].append(loss.item())
        H["pleasure"].append(brain.pleasure)
        H["stress"].append(brain.stress)
        H["mood"].append(brain.mood)
        H["curiosity"].append(brain.curiosity)
        H["energy"].append(brain.energy)
        H["separation"].append(brain.separation_anxiety)
        H["memory"].append(len(brain.long_term_memory))
        H["attention"].append(brain.attention)
        H["salience"].append(brain.salience)
        H["hab_mother"].append(brain.get_habituation_factor("mother"))
        H["hab_father"].append(brain.get_habituation_factor("father"))
        H["hab_gmom"].append(brain.get_habituation_factor("grandmother"))
        H["rpe"].append(brain.rpe_pleasure)
        H["trust_mother"].append(brain.entity_trust["mother"])
        H["trust_stranger"].append(brain.entity_trust["stranger"])
        H["action"].append(action)
        H["env"].append(env.environment)
        H["source"].append(source)
        r, c = int(next_state_np[0] * (grid - 1)), int(next_state_np[1] * (grid - 1))
        H["pos"].append((r, c))
        (visit_early if step < HALF else visit_late)[r, c] += 1

        if verbose and step % 100 == 0:
            print(f"[{policy:10s}] step {step:05d} | P {brain.pleasure:6.2f} | "
                  f"S {brain.stress:5.2f} | mood {brain.mood:+.2f} | "
                  f"cur {brain.curiosity:.2f} | LTM {len(brain.long_term_memory)}")

    return {"H": H, "brain": brain, "visit_early": visit_early, "visit_late": visit_late, "grid": grid}


def compute_metrics(H, grid, lo, hi):
    danger_pos = (grid - 1, grid - 1)
    near_danger = sum(1 for i in range(lo, hi)
                      if abs(H["pos"][i][0] - danger_pos[0]) + abs(H["pos"][i][1] - danger_pos[1]) <= 1)
    near_mother = sum(1 for i in range(lo, hi) if H["source"][i] == "mother")
    return {
        "pleasure":      float(np.mean(H["pleasure"][lo:hi])),
        "stress":        float(np.mean(H["stress"][lo:hi])),
        "mood":          float(np.mean(H["mood"][lo:hi])),
        "curiosity":     float(np.mean(H["curiosity"][lo:hi])),
        "mother_pct":    100 * near_mother / (hi - lo),
        "danger_pct":    100 * near_danger / (hi - lo),
        "loss":          float(np.mean(H["loss"][lo:hi])),
    }


# -------- Run both policies across all seeds --------

print(f"Running {len(SEEDS)} seeds × 2 policies × {NUM_STEPS} steps...")
runs = {"predictive": [], "reflex": []}
for seed in SEEDS:
    for policy in ("predictive", "reflex"):
        print(f"  seed={seed}  policy={policy}")
        runs[policy].append(run_episode(policy, seed=seed, verbose=False))

# Representative single-seed runs for plotting (first seed)
exp, base = runs["predictive"][0], runs["reflex"][0]


# -------- Aggregate comparison (mean ± std across seeds) --------

def aggregate(runs_list, lo, hi):
    per_seed = [compute_metrics(r["H"], r["grid"], lo, hi) for r in runs_list]
    keys = per_seed[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std()))
    return out

def print_comparison(exp_agg, base_agg, label):
    print(f"\n--- {label}  (mean ± std over {len(SEEDS)} seeds) ---")
    print(f"{'metric':<14s} {'predictive':>22s} {'reflex-only':>22s} {'delta':>10s}")
    for k in ["pleasure", "stress", "mood", "curiosity", "mother_pct", "danger_pct", "loss"]:
        em, es = exp_agg[k]
        bm, bs = base_agg[k]
        unit = "%" if k.endswith("_pct") else ""
        print(f"{k:<14s} {em:>14.3f}{unit} ± {es:5.3f} "
              f"{bm:>13.3f}{unit} ± {bs:5.3f} {em - bm:>+10.3f}")

print("\n" + "=" * 78)
print(f"PREDICTIVE vs REFLEX-ONLY BASELINE — {len(SEEDS)} seeds")
print("=" * 78)
print_comparison(aggregate(runs["predictive"], 0, NUM_STEPS),
                 aggregate(runs["reflex"],     0, NUM_STEPS),
                 "FULL RUN (0–2000)")
print_comparison(aggregate(runs["predictive"], 0, HALF),
                 aggregate(runs["reflex"],     0, HALF),
                 "EARLY (0–1000)")
print_comparison(aggregate(runs["predictive"], HALF, NUM_STEPS),
                 aggregate(runs["reflex"],     HALF, NUM_STEPS),
                 "LATE  (1000–2000)")

# Action distribution averaged across seeds
def action_dist(runs_list):
    pct = np.zeros(len(ACTION_NAMES))
    for r in runs_list:
        c = Counter(r["H"]["action"])
        for a in ACTION_NAMES:
            pct[a] += 100 * c.get(a, 0) / NUM_STEPS
    return pct / len(runs_list)

ep_pct  = action_dist(runs["predictive"])
ba_pct  = action_dist(runs["reflex"])
print(f"\n--- Action distribution (mean over {len(SEEDS)} seeds) ---")
print(f"{'action':<8s} {'predictive':>12s} {'reflex-only':>12s}")
for a, name in ACTION_NAMES.items():
    print(f"{name:<8s} {ep_pct[a]:>11.1f}% {ba_pct[a]:>11.1f}%")


# -------- Time-Series Dashboard (predictive run) --------

H = exp["H"]
fig1, axes = plt.subplots(2, 4, figsize=(22, 8))
fig1.suptitle("ProtoMind — Developmental Dynamics (predictive run)", fontsize=14)

axes[0, 0].plot(H["pleasure"], label="Pleasure", color="green")
axes[0, 0].plot(H["stress"],   label="Stress",   color="red")
axes[0, 0].plot(H["mood"],     label="Mood",     color="darkgreen", linewidth=2, linestyle="--")
axes[0, 0].set_title("Emotion & Mood"); axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(H["rpe"], label="RPE (Pleasure)", color="gold")
axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
axes[0, 1].set_title("Reward Prediction Error"); axes[0, 1].legend(fontsize=8)

axes[0, 2].plot(H["energy"],     label="Energy",             color="darkorange")
axes[0, 2].plot(H["separation"], label="Separation Anxiety", color="purple", linestyle="--")
axes[0, 2].set_ylim(0, 1.05)
axes[0, 2].set_title("Energy & Separation Anxiety"); axes[0, 2].legend(fontsize=8)

axes[0, 3].plot(H["trust_mother"],   label="Mother Trust",   color="green")
axes[0, 3].plot(H["trust_stranger"], label="Stranger Trust", color="saddlebrown")
axes[0, 3].axhline(y=0.5, color="black", linestyle="--", alpha=0.3)
axes[0, 3].set_ylim(-0.1, 1.1)
axes[0, 3].set_title("Social Trust / Reputation"); axes[0, 3].legend(fontsize=8)

axes[1, 0].plot(H["curiosity"], label="Curiosity", color="purple")
axes[1, 0].set_title("Curiosity"); axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(H["attention"], label="Attention", color="blue")
axes[1, 1].plot(H["salience"],  label="Salience",  color="orange")
axes[1, 1].set_title("Attention & Salience"); axes[1, 1].legend(fontsize=8)

axes[1, 2].plot(H["hab_mother"], label="Mother",      color="green")
axes[1, 2].plot(H["hab_father"], label="Father",      color="steelblue")
axes[1, 2].plot(H["hab_gmom"],   label="Grandmother", color="purple")
axes[1, 2].set_ylim(0, 1.05)
axes[1, 2].axhline(y=1.0, color="black", linestyle="--", alpha=0.2)
axes[1, 2].set_title("Habituation Factors (1=fresh, 0=bored)"); axes[1, 2].legend(fontsize=8)

axes[1, 3].plot(H["loss"], label="World Model Loss", color="gray")
axes[1, 3].set_title("World Model Loss"); axes[1, 3].legend(fontsize=8)

plt.tight_layout()
plt.savefig("dynamics.png", dpi=120)
plt.show()


# -------- Predictive vs Baseline: emotion + occupancy comparison --------

fig_cmp, ax_cmp = plt.subplots(2, 2, figsize=(14, 8))
fig_cmp.suptitle("Predictive vs Reflex-Only Baseline", fontsize=14)

# rolling mean for readability
def rolling(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

ax_cmp[0, 0].plot(rolling(exp["H"]["pleasure"]),  label="predictive", color="green")
ax_cmp[0, 0].plot(rolling(base["H"]["pleasure"]), label="reflex-only", color="gray", linestyle="--")
ax_cmp[0, 0].set_title("Pleasure (rolling mean)"); ax_cmp[0, 0].legend(fontsize=8)

ax_cmp[0, 1].plot(rolling(exp["H"]["stress"]),  label="predictive", color="red")
ax_cmp[0, 1].plot(rolling(base["H"]["stress"]), label="reflex-only", color="gray", linestyle="--")
ax_cmp[0, 1].set_title("Stress (rolling mean)"); ax_cmp[0, 1].legend(fontsize=8)

vmax = max(exp["visit_late"].max(), base["visit_late"].max())
g = exp["grid"]
im_p = ax_cmp[1, 0].imshow(exp["visit_late"], cmap="viridis", vmin=0, vmax=vmax)
ax_cmp[1, 0].set_title("Predictive — late occupancy"); plt.colorbar(im_p, ax=ax_cmp[1, 0])
im_b = ax_cmp[1, 1].imshow(base["visit_late"], cmap="viridis", vmin=0, vmax=vmax)
ax_cmp[1, 1].set_title("Reflex-only — late occupancy"); plt.colorbar(im_b, ax=ax_cmp[1, 1])
for ax in (ax_cmp[1, 0], ax_cmp[1, 1]):
    ax.scatter([0], [0], marker="*", s=200, c="white", edgecolors="black", label="mother (home)")
    ax.scatter([g - 1], [g - 1], marker="X", s=150, c="red", edgecolors="black", label="danger")
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig("predictive_vs_baseline.png", dpi=120)
plt.show()


# -------- Spatial Memory (predictive run) --------

brain = exp["brain"]
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle("ProtoMind — Spatial Memory (predictive run)", fontsize=14)
im0 = axes2[0].imshow(brain.pleasure_map, cmap="Greens", vmin=0); plt.colorbar(im0, ax=axes2[0])
axes2[0].set_title("Pleasure Map (Comfort Zones)")
im1 = axes2[1].imshow(brain.stress_map,   cmap="Reds",   vmin=0); plt.colorbar(im1, ax=axes2[1])
axes2[1].set_title("Stress Map (Danger Zones)")
im2 = axes2[2].imshow(brain.map_counts,   cmap="Blues",  vmin=0); plt.colorbar(im2, ax=axes2[2])
axes2[2].set_title("Visit Count Map (Exploration)")
plt.tight_layout()
plt.savefig("spatial_memory.png", dpi=120)
plt.show()
