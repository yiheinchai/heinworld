# Iteration 8: MaskGIT — Parallel Iterative Decoding

## Goal

Replace slow autoregressive decoding (one token at a time) with MaskGIT's parallel decoding: predict all tokens at once, keep the most confident, re-predict the rest. This is Stage 3 of TinyWorlds.

## Why This Iteration Exists

In Iteration 5, the dynamics model predicted 64 tokens per frame one-by-one — 64 sequential forward passes. MaskGIT reduces this to ~8 forward passes while maintaining quality, by predicting many tokens in parallel and iteratively refining.

## Concepts You Need

### 1. The Problem with Autoregressive Decoding for Images

In GPT, generating text one word at a time makes sense — each word heavily depends on the previous one ("The cat sat on the ___"). But image patches have **more spatial independence** — knowing patch (0,0) is blue sky doesn't strongly constrain what patch (7,7) looks like.

Autoregressive: 64 tokens × 1 forward pass each = **64 forward passes**
Desired: predict all 64 at once, somehow

### 2. MaskGIT — The Key Idea

Train with **masked prediction** (like BERT), generate with **iterative unmasking**:

**Training** (like BERT):
1. Take all 64 tokens of a frame
2. Randomly mask 50-100% of them (replace with a [MASK] token)
3. Predict the masked tokens given the unmasked ones
4. Cross-entropy loss only on masked positions

**Inference** (iterative):
1. Start with all 64 positions masked
2. Forward pass → get logits for all positions
3. Sample/argmax all 64 tokens
4. Keep only the **k most confident** predictions (highest softmax probability)
5. Re-mask the rest
6. Repeat from step 2, increasing k each iteration

After ~8 iterations, all tokens are unmasked.

### 3. The Masking Schedule

During training, the mask ratio is sampled uniformly from [0.5, 1.0]:
```python
mask_ratio = torch.rand(1) * 0.5 + 0.5  # between 50% and 100%
num_masked = int(mask_ratio * num_tokens)
```

During inference, tokens are revealed following an **exponential schedule**:
```python
# At step m out of T total steps:
# Fraction revealed so far: 1 - exp(-k * m / T) / (1 - exp(-k))
# where k controls the curve shape (k ≈ 1-3)
```

Early steps reveal few tokens (hard, sparse context), later steps reveal many (easy, rich context). Like solving a jigsaw puzzle — the first pieces are the hardest.

### 4. Temporal Anchors

For video, you don't mask the **context frames** (past frames that are already known). You only mask the **future frame** being generated. Additionally, guarantee at least one unmasked "anchor" token per patch position across time:

```python
# For each of the 64 patch positions, ensure at least one frame's token
# at that position is unmasked — this provides temporal continuity
```

### 5. The [MASK] Token

Add a special learnable embedding for masked positions:
```python
mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
# When a position is masked, replace its embedding with mask_token
```

### 6. Confidence-Based Token Selection

At each inference step, after predicting all tokens:
```python
logits = model(tokens_with_masks)           # (B, P, vocab)
probs = F.softmax(logits, dim=-1)            # (B, P, vocab)
max_probs, predicted = probs.max(dim=-1)     # (B, P) — confidence per position

# Sort by confidence, keep top-k
k = num_to_reveal_this_step
_, top_k_indices = max_probs.topk(k, dim=-1)

# Unmask the top-k positions
tokens[masked_positions[top_k_indices]] = predicted[top_k_indices]
```

High-confidence predictions are likely correct → lock them in.
Low-confidence predictions get another chance with more context next iteration.

### 7. Training Changes vs Iteration 5

| | Iteration 5 (Autoregressive) | This Iteration (MaskGIT) |
|---|---|---|
| Training | Causal: predict next token given all previous | Masked: predict masked tokens given unmasked |
| Attention | Causal temporal mask | Bidirectional (future frame patches can see each other!) |
| Loss | All positions | Masked positions only |
| Inference | Sequential: 1 token per step | Parallel: many tokens per step |

The decoder temporal attention for the **current frame** is now **bidirectional** — unmasked patches in the frame being generated can attend to each other. Only attention to future frames is blocked.

---

## Your Task

### Model

Modify the Dynamics Model from Iteration 5/6:

1. **Add a [MASK] token** embedding
2. **Change training to masked prediction**:
   - Take a clip of T frames with their tokens
   - Context frames (0..T-2): keep all tokens unmasked
   - Target frame (T-1): randomly mask 50-100% of tokens
   - Replace masked tokens with the [MASK] embedding
   - Predict and compute loss only on masked positions
3. **Add FiLM action conditioning** (from Iteration 6/7)

```
Input: token indices (B, T, P) with some masked + actions (B, T)
 → Embed tokens (use mask_token for masked positions)
 → Add positional encodings
 → N × SpaceTimeBlock with FiLM
 → Classification head → logits at masked positions
 → Cross-entropy loss on masked positions only
```

### Inference — Iterative Unmasking

```python
def generate_frame_maskgit(model, context_tokens, action, num_steps=8):
    P = 64  # patches per frame
    # Start fully masked
    current = torch.full((1, P), MASK_TOKEN_ID, device=device)

    for step in range(num_steps):
        # Forward pass with context + current (partially unmasked) frame
        all_tokens = torch.cat([context_tokens, current.unsqueeze(1)], dim=1)
        logits = model(all_tokens, action)  # (1, T, P, vocab)
        next_logits = logits[:, -1]          # (1, P, vocab)

        # Predict all positions
        probs = F.softmax(next_logits, dim=-1)
        max_probs, predicted = probs.max(dim=-1)  # (1, P)

        # Determine how many to reveal this step (exponential schedule)
        frac = (step + 1) / num_steps
        k = max(1, int(P * frac))  # simple linear schedule, or use exponential

        # Find still-masked positions
        masked_pos = (current == MASK_TOKEN_ID)
        # Among masked positions, pick the k most confident
        masked_probs = max_probs.clone()
        masked_probs[~masked_pos] = -1  # don't re-select already unmasked
        _, topk = masked_probs.topk(min(k, masked_pos.sum()), dim=-1)

        # Unmask them
        current[0, topk[0]] = predicted[0, topk[0]]

    return current  # (1, P) — fully unmasked predicted tokens
```

### Training
- Loss: cross-entropy on masked positions only
- Adam, lr=1e-4, ~50 epochs
- Use the frozen tokenizer + frozen action model for tokenization

### Visualization

1. Show the iterative unmasking process: at each of the 8 steps, decode the partially-revealed tokens to pixels. Watch the frame "crystallize" from noise
2. Compare generation quality to autoregressive (Iteration 5)
3. Compare speed: time 64 autoregressive steps vs 8 MaskGIT steps

### What to Observe

1. **~8 forward passes instead of 64** — roughly 8x faster generation
2. **Quality is comparable** — iterative refinement produces coherent frames
3. **The crystallization effect** — early steps get the broad structure, late steps fill in details
4. **Bidirectional attention in the decoder** is key — unlike autoregressive, each predicted token benefits from ALL other unmasked tokens, not just the ones before it
