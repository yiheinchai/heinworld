# Iteration 6: Action Conditioning with FiLM

## Goal

Add **controllable actions** to the dynamics model so you can steer what happens in the generated world. Use FiLM (Feature-wise Linear Modulation) to condition the transformer on action inputs.

## Why This Iteration Exists

In Iteration 5, the dynamics model generated future frames but you had no control. The ball moved on its own. For an interactive world model, you need to say "I pressed UP" and see the paddle move up.

This iteration uses **hardcoded action labels** for Pong (we know the actions: up, down, stay). The next iteration will remove this requirement by learning actions from video alone.

## Concepts You Need

### 1. Action Conditioning — The Design Space

There are several ways to inject action information into a transformer:

**Option A: Concatenate as extra tokens**
```
input: [frame_tokens, action_token, frame_tokens, action_token, ...]
```
Problem: changes sequence length, mixes modalities, the model must learn to ignore action tokens during spatial attention.

**Option B: Add to embeddings**
```
token_embeds = token_embeds + action_embed
```
Problem: weak conditioning — the action information gets diluted through many layers.

**Option C: FiLM (Feature-wise Linear Modulation)** ← what we use
```
After FFN in each block: output = γ * output + β
where (γ, β) are learned functions of the action
```
Best of both worlds: strong conditioning at every layer without changing the sequence.

### 2. FiLM — How It Works

FiLM modulates intermediate features by learning a **scale** (γ) and **shift** (β) from the conditioning signal:

```python
class FiLM(nn.Module):
    def __init__(self, action_dim, feature_dim):
        super().__init__()
        self.scale = nn.Linear(action_dim, feature_dim)
        self.shift = nn.Linear(action_dim, feature_dim)

    def forward(self, x, action_embed):
        """
        x: (B, T*P, d_model) — features from the FFN
        action_embed: (B, d_model) — action embedding
        """
        gamma = self.scale(action_embed)    # (B, d_model)
        beta = self.shift(action_embed)     # (B, d_model)
        # Broadcast: gamma and beta are (B, 1, d_model), x is (B, T*P, d_model)
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)
```

**Intuition**: the action tells each layer which features to amplify and which to suppress. If the action is "up," the FiLM layer might amplify features related to upward motion and suppress downward motion features. It's like a volume knob per feature.

### 3. Where FiLM Goes in the Block

Add FiLM after the FFN in each space-time block:

```python
class SpaceTimeBlockWithFiLM(nn.Module):
    def forward(self, x, T, P, action_embed, causal_mask=None):
        # Spatial attention (unchanged)
        x = x + self.spatial_attn(self.ln1(x))

        # Temporal attention (unchanged)
        x = x + self.temporal_attn(self.ln2(x), causal_mask)

        # FFN + FiLM conditioning
        ffn_out = self.ffn(self.ln3(x))
        ffn_out = self.film(ffn_out, action_embed)  # ← action modulates here
        x = x + ffn_out
        return x
```

### 4. Action Representation for Pong

For this iteration, we hardcode 3 actions for Pong:
```
0 = stay
1 = up
2 = down
```

Embed them with `nn.Embedding(3, d_model)`:
```python
action_embed = nn.Embedding(num_actions, d_model)
# action_id: (B,) integers in {0, 1, 2}
# → (B, d_model) embedding vector
```

Each transition between frames has one action. For a clip of T frames, you have T-1 actions:
```
frame0 --[action0]--> frame1 --[action1]--> frame2 --[action2]--> frame3
```

### 5. Getting Action Labels

For real Pong, you'd record gameplay with controller inputs. For this exercise, you can **infer approximate actions** from the frame data:

```python
def infer_pong_actions(frames):
    """Heuristic: track the left paddle's vertical position change."""
    actions = []
    for i in range(len(frames) - 1):
        # Find paddle position (left side of frame, bright pixels)
        curr_paddle_y = find_paddle_y(frames[i])
        next_paddle_y = find_paddle_y(frames[i + 1])
        dy = next_paddle_y - curr_paddle_y
        if dy < -1:
            actions.append(1)   # up
        elif dy > 1:
            actions.append(2)   # down
        else:
            actions.append(0)   # stay
    return actions
```

Or simply use **random actions** during training — the model will learn that different actions lead to different outcomes even if the labels are noisy. The point of this iteration is to learn the FiLM architecture, not to get perfect action labels.

### 6. Training with Actions

The dataset now returns `(context_tokens, target_tokens, actions)`:

```python
# context_tokens: (T-1, P) — token indices for frames 0..T-2
# target_tokens:  (T-1, P) — token indices for frames 1..T-1
# actions:        (T-1,)   — action for each transition
```

Loss is the same cross-entropy, but the dynamics model now receives actions as additional input.

---

## Your Task

### Data
- Use the frozen tokenizer to get video tokens (same as Iteration 5)
- For each consecutive pair of frames, assign an action label (use heuristic paddle tracking or random)
- Dataset returns `(context_tokens, target_tokens, actions)`

### Model

Modify the Dynamics Model from Iteration 5:

1. Add `nn.Embedding(num_actions, d_model)` for action embeddings
2. Add a `FiLM` module to each `SpaceTimeBlock`
3. In the forward pass, embed the action and pass it to each block

```
Input: token indices (B, T, P) + actions (B, T)
 → Token embedding + positional encoding → (B, T, P, d_model)
 → Action embedding → (B, T, d_model)
 → N × SpaceTimeBlock with FiLM(action_embed)
 → Classification head → (B, T, P, vocab_size)
```

### Training
- Same as Iteration 5: cross-entropy loss on predicted tokens
- Adam, lr=1e-4, ~50 epochs

### Visualization — Interactive Generation

1. Encode seed frames → tokens
2. For each generation step, **you choose the action** (or cycle through them):
   - Generate with action=UP for 3 frames
   - Generate with action=DOWN for 3 frames
   - Generate with action=STAY for 3 frames
3. Verify that different actions produce **different** generated frames
4. Display the full generated sequence, annotating which action was used

### What to Observe

1. **Different actions → different outcomes** — the same seed frame produces different next frames depending on the action. This is controllability!
2. **FiLM is lightweight** — it adds very few parameters (just 2 linear layers per block) but provides strong conditioning
3. **The limitation**: we **cheated** by using known action labels. Real gameplay videos on the internet don't come with "I pressed UP" labels. You'd need millions of labeled videos to scale this approach, which is impractical
4. This motivates the key insight: *can we figure out what action happened just by comparing two consecutive frames?*
