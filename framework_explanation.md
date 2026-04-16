# Risk-Aware Generative Active Learning

**Anchored Contrastive Scoring · Semantic Neighborhood Disagreement · Risk Penalty**

> Complete Framework: Theory, Mathematics & Worked Example

---

## Part 1 — Problem Setup and Notation

We consider a sequential recommendation problem. At each timestep `t`, a user has accumulated an interaction history:

```
h_t = {(a1, r1), (a2, r2), ..., (a_{t-1}, r_{t-1})}
```

Where:
- `a_i` is the item shown at step i
- `r_i ∈ {0, 1}` is the binary reward — `1` = click/engage, `0` = skip/ignore

We define two sub-histories:

```
h_t+ = positive history {a_i ∈ h_t : r_i = 1}   (liked items)
h_t- = negative history {a_i ∈ h_t : r_i = 0}   (skipped items)
```

At `t = 1`, we have `h_1 = ∅` — this is the **cold-start moment**: zero history, zero signal.

The system must select the next item:

```
a*_t = argmax Q(a, h_t)   over candidate pool C_t
```

Our task is to design the scoring function `Q` that balances exploitation, exploration, and safety.

---

## Part 2 — Why Classical Bandits Fail

LinUCB, the dominant cold-start bandit algorithm, defines:

```
Q_LinUCB(a) = θ̂_t^T * x_a + α * sqrt(x_a^T * A_t^{-1} * x_a)
```

Where `A_t = Σ x_{a_i} x_{a_i}^T + λI` is the covariance matrix, updated after every interaction.

| Failure | Description |
|---------|-------------|
| **Computational** | Matrix inversion `A_t^{-1}` costs `O(d³)`. For `d = 768` (standard embedding), that is `768³ ≈ 453 million` operations per timestep. |
| **Psychological Blindness** | The uncertainty term is purely geometric. It treats showing a stranger a niche, alienating item identically to a safe mainstream item — both reduce uncertainty in the same way. |
| **Wrong Uncertainty** | The covariance matrix measures uncertainty about a global reward parameter `θ`, not about this specific user's preferences in this specific semantic region. |

---

## Part 3 — The Unified Objective Function

We replace all three LinUCB terms with a new scoring function composed of three interpretable components:

```
Q(a, h_t) = P_ACS(a, h_t)  +  α · SND(a, t)  −  λ · R(a)
              Exploitation      Exploration       Safety Penalty
```

**Parameters:**
- `α > 0` controls exploration aggressiveness
- `λ > 0` controls the strength of the safety constraint
- Both are tuned on a validation set

| Term | What it measures | How computed | LLM needed? |
|------|-----------------|--------------|-------------|
| `P_ACS(a, h_t)` | Will this user like this item? | Batched LLM call on shortlist | Yes (once) |
| `SND(a, t)` | How uncertain about this user here? | Neighbor lookup + count | No |
| `R(a)` | How risky is this item for a stranger? | Pre-computed offline | No (cached) |

---

## Part 4 — Offline Preprocessing

> Performed **once per catalog**. Never repeated unless the catalog changes.

### Step 1 — Semantic Embeddings

A frozen LLM (or sentence encoder) maps every item to a dense vector:

```
f_LLM : a ↦ e_a ∈ ℝ^d
```

These vectors are indexed into a FAISS structure for fast approximate nearest neighbor (ANN) retrieval in `O(log |I|)`.

### Step 2 — Risk Weight Assignment

The LLM is prompted once per item:

> *"On a scale from 0 to 1, how risky is it to recommend this item to a complete stranger with no established taste profile? Consider: how niche it is, how much prior context it assumes, how polarizing its content is. Output only a decimal number."*

This produces a static risk function `R : I → [0, 1]` stored as a lookup table.

- `R(a) ≈ 0` → safe for anyone
- `R(a) ≈ 1` → high risk to show a stranger

### Stage 1 — ANN Retrieval `O(log |I|)`

Compute the user's current preference centroid from positive interactions:

```
ē_t = Σ r_i · e_{a_i} / (Σ r_i + ε)
```

Where `ε = 10⁻⁸` prevents division by zero. At cold-start (`h_t = ∅`), `ē_t ≈ 0` and ANN returns globally popular items — a safe default.

Query FAISS to retrieve the top-M (`M ≈ 50`) unseen semantically relevant candidates:

```
C_t = ANN(ē_t, k=M) \ {a_i : (a_i, ·) ∈ h_t}
```

---

## Part 5 — Term 1: Anchored Contrastive Scoring (ACS)

### The Problem with Flat P(YES)

Standard LLM rankers ask:

```
"Will this user like item a? [YES] / [NO]"

P_abs(YES | a, h_t) = softmax(log p(YES), log p(NO))_YES
```

This produces **miscalibrated probabilities** because the LLM must infer preferences from a flat item list without a concrete reference point. LLMs are also systematically overconfident on cold absolute binary judgments.

### The ACS Mechanism

For each candidate `a ∈ C_t`, retrieve the most personally relevant evidence from the user's history:

```
x+(a) = argmin ‖e_a − e_{a_i}‖₂   (closest liked item)
x-(a) = argmin ‖e_a − e_{a_i}‖₂   (closest skipped item)
```

Then prompt the LLM contrastively:

> *"A user engaged with [description of x+] but skipped [description of x−]. Is [description of a] more similar to what they engaged with or what they skipped? Answer only: [ENGAGED] or [SKIPPED]."*

Extract the calibrated probability from raw logprobs:

```
P_ACS(a, h_t) = exp(log p(ENGAGED)) / [ exp(log p(ENGAGED)) + exp(log p(SKIPPED)) ]
```

**Cold-start fallback:** When `|h_t+| = 0` or `|h_t-| = 0`, one anchor is unavailable. The system falls back to absolute `P(YES)` scoring. After the first positive AND first negative interaction, ACS activates fully.

---

## Part 6 — Term 2: Semantic Neighborhood Disagreement (SND)

### Why Not Entropy?

Shannon entropy `H(a,t) = −P·log P − (1−P)·log(1−P)` is maximized when `P = 0.5`. It seems to measure uncertainty — but it measures the LLM's **population-level prior**, not evidence about this specific user.

> **The Core Flaw:** A niche polarizing item will always give `P ≈ 0.5` for everyone, regardless of how much we know about the current user. Entropy never decreases with interaction count. This is *aleatoric* uncertainty — irreducible, item-specific, wrong for active learning.

What active learning requires is **epistemic uncertainty** — uncertainty that is:

1. Specific to this user — not the population average
2. Decreasing as the user provides more evidence
3. High in regions where the user's reactions have been inconsistent
4. Low in regions where the user's preferences are already clear

SND satisfies all four. Entropy satisfies none.

### Definition

For candidate `a ∈ C_t`, retrieve the `k` nearest already-rated items from the user's history:

```
N_k(a, t) = top-k items in h_t closest to a by ‖e_a − e_{a_i}‖₂

n+(a,t) = |{a_i ∈ N_k(a,t) : r_i = 1}|
n-(a,t) = |{a_i ∈ N_k(a,t) : r_i = 0}|

SND(a, t) = min(n+, n-) / (max(n+, n-) + 1)
```

### Behavior Analysis

| n+ | n- | SND | Interpretation |
|----|----|-----|----------------|
| 0 | 0 | 0.00 | Cold-start — no rated neighbors, no signal |
| 5 | 0 | 0.00 | All neighbors liked — certain positive region |
| 0 | 5 | 0.00 | All neighbors skipped — certain negative region |
| 3 | 3 | 0.75 | Perfectly split — maximum uncertainty |
| 4 | 1 | 0.20 | Leaning positive but one surprise skip |
| 7 | 2 | 0.25 | Mostly positive, mild residual doubt |

### The ε Cold-Start Floor

When `|N_k(a,t)| < 2` for most candidates, SND ≈ 0 everywhere. To prevent clustering around the first liked item:

```
SND_ε(a, t) = SND(a, t) + ε · 1[|N_k(a,t)| < 2]
```

Where `ε = 0.05`. This ensures the system occasionally probes new semantic regions even before SND has neighborhood evidence.

### Convergence Property

> **Claim:** For any user with consistent preferences near item `a`:
>
> `lim E[SND(a,t)] = 0` as `t → ∞`
>
> **Proof sketch:** If the user consistently likes items near `a`, all `k` neighbors will have `r_i = 1`, giving `n- = 0` and `SND = 0` for all `t ≥ k`. Symmetrically for consistent skips. Only genuine inconsistency maintains non-zero SND, and that either reflects true aleatoric randomness (stabilises) or resolves over time (decreases). SND never increases with `t`. □

---

## Part 7 — Term 3: Risk Penalty R(a)

The subtraction `−λ · R(a)` creates an asymmetric exploration policy:

| Item Type | Effect |
|-----------|--------|
| **Safe items** `R(a) ≈ 0` | Penalty is negligible. These can be freely explored even when SND is high — the system learns aggressively from safe content. |
| **Risky items** `R(a) ≈ 1` | Penalty is large. For a risky item to be selected, `P_ACS` must be very high to overcome the deduction. The system cannot use risky content as an exploration vehicle. |

> **Key safety property:** A high-SND, high-R(a) item — uncertain AND risky — has its exploration bonus cancelled by the risk penalty. **Exploration only proceeds through safe items.**

---

## Part 8 — Full Worked Example with Numbers

> Generic item catalog. User has completed 5 interactions. We score candidate Item-F.

### User History h5

| Step | Item | Embedding (2D simplified) | Reward |
|------|------|--------------------------|--------|
| a1 | Item-A | [0.9, 0.1] | 1 (liked) |
| a2 | Item-B | [0.1, 0.9] | 0 (skipped) |
| a3 | Item-C | [0.8, 0.2] | 1 (liked) |
| a4 | Item-D | [0.2, 0.8] | 0 (skipped) |
| a5 | Item-E | [0.7, 0.3] | 1 (liked) |

**Candidate:** Item-F with embedding `e_F = [0.75, 0.25]`, `R(F) = 0.15`

**Parameters:** `α = 0.5`, `λ = 0.3`, `k = 3`

---

### Computing Term 1 — ACS

**Step 1: Find x+ (closest liked item to Item-F)**

```
‖e_F − e_A‖ = ‖[−0.15,  0.15]‖ = 0.212
‖e_F − e_C‖ = ‖[−0.05,  0.05]‖ = 0.071  ← minimum
‖e_F − e_E‖ = ‖[ 0.05, −0.05]‖ = 0.071

→ x+ = Item-C (tied with E; take C)
```

**Step 2: Find x− (closest skipped item to Item-F)**

```
‖e_F − e_B‖ = ‖[0.65, −0.65]‖ = 0.919
‖e_F − e_D‖ = ‖[0.55, −0.55]‖ = 0.778  ← minimum

→ x- = Item-D
```

**Step 3: Contrastive LLM Prompt**

> *"User engaged with Item-C but skipped Item-D. Is Item-F more like Item-C or Item-D?"*

```
LLM logprobs:
  log p([ENGAGED]) = −0.22
  log p([SKIPPED]) = −2.43
```

**Step 4: Extract Probability**

```
P_ACS = exp(−0.22) / [ exp(−0.22) + exp(−2.43) ]
      = 0.802 / [ 0.802 + 0.088 ]
      = 0.802 / 0.890
      = 0.901
```

---

### Computing Term 2 — SND (k = 3)

Retrieve 3 nearest rated neighbors of Item-F:

| Item | Distance to F | Reward |
|------|--------------|--------|
| Item-E | 0.071 | 1 (liked) |
| Item-C | 0.071 | 1 (liked) |
| Item-A | 0.212 | 1 (liked) |

```
n+ = 3, n- = 0

SND(F, 5) = min(3, 0) / (max(3, 0) + 1) = 0 / 4 = 0.0
```

> All 3 nearest neighbors were liked. The neighborhood is unanimous — **no exploration bonus needed**. We already know this user likes this region.

---

### Computing Term 3 — Risk

```
R(F) = 0.15   (pre-computed offline)

λ · R(F) = 0.3 × 0.15 = 0.045
```

---

### Final Score for Item-F

```
Q(F, h5) = 0.901 + (0.5 × 0.0) − 0.045
         = 0.901 + 0.000 − 0.045
         = 0.856
```

---

### Comparison: A Risky Alternative — Item-G

Item-G: `e_G = [0.15, 0.85]` (close to skipped region), `R(G) = 0.8` (high risk)

```
x+ = Item-A (closest liked, distance = 1.06)
x- = Item-B (closest skipped, distance = 0.071)

log p([ENGAGED]) = −3.1
log p([SKIPPED]) = −0.09

P_ACS(G) = exp(−3.1) / [exp(−3.1) + exp(−0.09)]
         = 0.045 / [0.045 + 0.914]
         = 0.047
```

SND for Item-G — neighbors: Item-B (r=0), Item-D (r=0), Item-A (r=1):

```
n+ = 1, n- = 2

SND(G, 5) = min(1, 2) / (max(1, 2) + 1) = 1 / 3 = 0.333
```

```
Q(G, h5) = 0.047 + (0.5 × 0.333) − (0.3 × 0.8)
         = 0.047 + 0.167 − 0.240
         = −0.026
```

---

### Decision

```
Q(Item-F) = 0.856  >>  Q(Item-G) = −0.026
```

**Item-F is selected.** The system correctly:

- Exploits a high-confidence item via ACS (`0.901`)
- Adds no unnecessary exploration bonus since `SND = 0`
- Rejects Item-G despite mild exploration value — low ACS + high risk = negative total score

---

## Summary — What Each Component Does

| Component | Asks | Answers | Key Property |
|-----------|------|---------|--------------|
| `P_ACS` | Does this user like this? | Calibrated exploitation score | Contrastive, history-anchored |
| `SND` | How uncertain are we here? | User-specific exploration bonus | Provably → 0 as t → ∞ |
| `R(a)` | Is this safe to explore? | Psychological safety constraint | Blocks risky exploration |

---

*Risk-Aware Generative Active Learning Framework — Complete Technical Reference*
