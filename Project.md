# ProtoMind — Technical Architecture

## Core Philosophy

ProtoMind is built on one thesis:

> Intelligence, emotions, and morality should **emerge** from developmental processes — not be engineered top-down.

The system rejects:
- Pre-trained knowledge
- Labeled reward signals
- Hardcoded emotion rules
- Static datasets

It accepts:
- A wired learning mechanism (architecture + reflexes)
- Continuous parameter updates from experience
- Emotions as global state variables (not labels)
- Memory as selective weight change, not storage

---

## System Architecture (5 Layers)

### 1. Sensory Layer
Converts world state into a normalized feature vector.
Currently: 2D position on a 5x5 grid → `[x/4, y/4]` ∈ [0, 1]²

The world (`RichWorld`) provides rich contextual signals beyond position:
- **Environment**: home (safe, predictable) or outside (uncertain, novel)
- **Social entities**: mother, father, grandmother, stranger — each with own emotional profile and stochastic presence (arrive/depart probabilities per step per environment)
- **Ambient sensory events**: loud sound, music, new toy, rain, animal — position-independent, environment-weighted probabilities
- All signals are delivered via the `source` string which the brain uses for habituation and expectation tracking

### 2. World Model
A 2-layer neural network (MLP) trained self-supervised to predict the next state from the current state + action.
- Input: `state (2) + action_onehot (5)` → 7-dim
- Hidden: 32 units, ReLU
- Output: predicted next state (2-dim)
- Training signal: MSE prediction error (no external labels)
- Curiosity = prediction error magnitude (surprise drives exploration)

### 3. Memory System
**Two-tier architecture:**

A. Working Memory (explicit, short-term)
- List of recent `(state, pleasure, stress, source, timestamp)` tuples
- Cleared at each sleep cycle (every 500 steps)

B. Long-Term Memory — `SleepMemory` (explicit, consolidated)
- Capacity: 1000 entries
- Consolidation: recency × (0.2 + emotional intensity) scoring
- Provides state-based and source-based expectation queries
- Nearest-neighbor lookup (distance-weighted averaging)

Memory stores **raw** (un-habituated) pleasure/stress values — it remembers what actually happened, not the agent's subjective response at the time.

### 4. Emotion / Hormone System
Global state variables that bias learning, memory, and action:

| Variable | Role |
|---|---|
| `pleasure` | Accumulated positive experience (exponential smoothing) |
| `stress` | Accumulated negative experience |
| `curiosity` | Driven by prediction error + expectation violation + RPE |
| `expected_pleasure` | Running average of memory-based pleasure expectations |
| `expected_stress` | Running average of memory-based stress expectations |
| `rpe_pleasure` | Reward Prediction Error: `actual_pleasure - expected_pleasure` |
| `rpe_stress` | Reward Prediction Error for stress |
| `curiosity_drive` | 1 + curiosity — scales exploration bonus in decisions |
| `safety_drive` | 1 + stress — scales safety penalty in decisions |
| `attachment_drive` | 1 + pleasure — scales pleasure bonus in decisions |
| `attention` | Smoothed salience — amplifies net emotional signal |
| `salience` | 0.4 × novelty + 0.3 × emotion strength + 0.3 × RPE magnitude |

**Habituation (Day-9):**
Each source ("parent", "danger", "event", "none") has a fatigue value ∈ [0, 1].
- Build-up: `fatigue += ALPHA * (1 - fatigue)` per visit (ALPHA = 0.05)
- Recovery: `fatigue *= (1 - DECAY)` every step (DECAY = 0.005)
- Sleep recovery: `fatigue *= 0.5` at each sleep cycle
- Habituation factor: `1 - fatigue` scales incoming pleasure/stress before emotional accumulation
- Spatial maps store **raw** pleasure/stress (true environment value); emotional state accumulates habituated values; memory stores raw values

**Curiosity update (Day-19 fix):**
The secure-base multiplier `(1 + 0.5 × mother_proximity)` is applied only to the new contribution, not to the smoothed carry-over (the earlier formulation caused exponential runaway when `0.85 × multiplier > 1`):
```
curiosity_input = (prediction_error + |es − stress| + 0.5 × |rpe_pleasure|) × (1 + 0.5 × secure_base)
curiosity       = min(0.85 × curiosity + curiosity_input, 10.0)
```

**Reward Prediction Error (Day-11):**
A dopamine-like surprise signal computed each step before expectations update:
- `rpe_pleasure = actual_pleasure - expected_pleasure`
- `rpe_stress   = actual_stress   - expected_stress`
- Positive RPE (better than expected): `pleasure += 0.2 * max(0, rpe_pleasure)` — excitement
- Negative RPE (disappointment): `stress += 0.1 * max(0, -rpe_pleasure)` — mild distress
- Curiosity: `+= 0.5 * abs(rpe_pleasure)` — surprise drives exploration regardless of sign
- Salience: `0.3 × RPE magnitude` added — surprising events are more salient, better remembered
- First encounters produce high RPE (nothing expected → everything is a surprise)
- Familiar routines produce RPE ≈ 0 (expected = actual)

### 5. Action System
Two modes selected per step (80/20 split):

**Reflex (20%):** Probability-weighted random action biased by expected stress (stay) and expected pleasure (move).

**Predictive (80%):** Look-ahead over all 5 actions using the world model. Value function per action:
```
value = attachment_drive × ep
      - safety_drive × es
      + curiosity_drive × curiosity
      + attention × (ep - es)
```
Where `ep`, `es` are blended 50/50 from long-term memory and spatial map expectations for the predicted next state.

---

## Environment

`RichWorld` — 5×5 grid with dynamic social and sensory context:

**Environments** (transition every ~300 steps home / ~100 steps outside):
- Home: safe, mother always present, lower ambient event rates
- Outside: higher unpredictability, strangers more likely, richer ambient events, position resets on transition

**Social Entities** (stochastic presence per step):
| Entity | Pleasure | Stress | Home presence | Outside presence | Home pos | Outside pos |
|---|---|---|---|---|---|---|
| Mother | 1.0 | 0.0 | Always | Often (80%) | (0,0) | (2,2) |
| Father | 0.8 | 0.0 | Frequent (~60%) | Occasional | (0,1) | (2,3) |
| Grandmother | 0.9 | 0.0 | Rare (~1-2 visits) | Very rare | (1,0) | (1,2) |
| Stranger | 0.1 | 0.5 | Very rare | Occasional (2%) | (3,3) | (4,3) |

**Danger zones**: (4,4) home and outside

**Ambient Sensory Events** (position-independent):
| Event | Pleasure | Stress | Home prob/step | Outside prob/step |
|---|---|---|---|---|
| loud_sound | 0.0 | 0.8 | 2% | 5% |
| music | 0.6 | 0.0 | 5% | 1% |
| new_toy | 0.7 | 0.0 | 3% | 1% |
| rain | 0.0 | 0.3 | 1% | 6% |
| animal | 0.5 | 0.2 | 1% | 8% |

State: normalized position `[row/4, col/4]`
Actions: up (0), down (1), left (2), right (3), stay (4)

---

## Training Loop

2000 steps, Adam optimizer (lr=0.01), MSELoss:

```
for each step:
    1. Select action (20% reflex, 80% predictive)
    2. Step environment → next_state, pleasure, stress, source
    3. Predict next_state with world model
    4. Compute MSE loss, backprop, update world model
    5. Update internal state (habituation → emotions → memory → drives)
    6. Every 500 steps: sleep consolidation
```

Sleep consolidation:
- Flush working memory → long-term memory
- Score and prune LTM to capacity
- Reduce stress × 0.5, curiosity × 0.7, attention × 0.7
- Restore all source fatigues × 0.5

---

## Developmental Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1: World Model | Done | Predict next state from action |
| Phase 2: Emotions | Done | Pleasure, stress, curiosity as global state |
| Phase 3: Memory | Done | Working + long-term, sleep consolidation |
| Phase 4: Attachment + Drives | Done | Source-weighted expectations, spatial maps |
| Phase 4b: Habituation | Done | Boredom forces exploration |
| Phase 4c: RPE / Dopamine | Done | Surprise signal — first encounters feel special |
| Phase 4d: Rich World | Done | Social entities, environments, sensory events |
| Phase 4e: Validation & Stability (Day-19) | Done | Behavioral instrumentation, reflex-only baseline, curiosity stabilization |
| Phase 5: Language | Future | Compression of world model into symbols |
| Phase 6: Self-Model | Future | Model of "me" enabling guilt, pride, reflection |

---

## Validation Protocol (Day-19)

Every behavioral claim about emergence is now evaluated against a **reflex-only baseline**: the same world, same seed, same brain updates, but action selection always uses `get_reflex_action()` (no world-model planning). The world model still trains, emotions still update — only the policy is disabled. Differences between the two runs isolate what is genuinely *learned* from what is a statistical property of the environment.

`train.py` exposes `run_episode(policy, seed)` and produces a side-by-side comparison table for every new phase.

Baseline-validated claims (seed = 42, 2000 steps):
- Mother-seeking: 27.0% predictive vs 7.5% reflex (≈ 3.5× lift).
- Danger-zone avoidance: 0.0% predictive vs 8.3% reflex.
- Mean pleasure / mood: substantially higher in predictive run.
- Action distribution: predictive policy is directionally biased (`up`/`left` dominant); reflex is uniform.

Instrumentation produced per run: `dynamics.png` (8-panel time-series), `predictive_vs_baseline.png` (rolling means + occupancy comparison), `occupancy_early_vs_late.png`, `spatial_memory.png`.

---

## Key Design Decisions

1. **No reward function** — learning is driven purely by world model prediction error
2. **Emotions are not labels** — they are continuous variables that drift from experience
3. **Memory is selective** — only emotionally salient experiences persist explicitly; most learning is implicit weight change
4. **Sleep is functional** — not just a metaphor; it consolidates memory and resets emotional overload
5. **Habituation is biological** — repeated stimuli lose impact; novelty drives exploration naturally
6. **Morality is downstream** — when a social agent (parent) has higher priority weight, behavioral correction emerges without hardcoding right/wrong
