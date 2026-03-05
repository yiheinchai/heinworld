# Iteration 2: Discrete Bottleneck with Finite Scalar Quantization (FSQ)

## Goal

Add a quantization layer to your autoencoder so the latent space is **discrete** — each spatial position maps to one of a fixed number of codes, like words in a vocabulary.

## Why This Iteration Exists

In Iteration 1, you built a conv autoencoder with continuous latents. The latent values were arbitrary floats with no structure. You couldn't sample from it, and you couldn't treat latents like a vocabulary for next-token prediction.

For the world model, we need to predict future frames like predicting next words. That requires **discrete tokens** and **cross-entropy loss** — not continuous vectors and regression loss. This iteration creates that discrete vocabulary.

## Concepts You Need

### 1. Why Discrete Representations?

In GPT, you predict the next word from a vocabulary of ~50,000 tokens using cross-entropy. You can't do cross-entropy on continuous floats — you need to pick from a finite set of options.

The plan:
- **Iteration 1**: frame → continuous latent (floats) → frame
- **This iteration**: frame → continuous latent → **round to discrete codes** → frame
- **Later iterations**: predict discrete codes with a transformer (like GPT predicts word tokens)

Each spatial position in the latent grid gets assigned a code from a vocabulary. For example, with vocab size 1024, each 8×8 position holds one of 1024 possible tokens — just like each position in a sentence holds one of 50,000 possible words.

### 2. Finite Scalar Quantization (FSQ)

FSQ is a simple, elegant quantization method. The idea:

**Step 1**: Encoder outputs L values per spatial position (e.g., L=5)
```
encoder output: (B, L, 8, 8) where L=5
```

**Step 2**: Squash each value to [-1, 1] with tanh, then scale to [0, num_bins-1]
```python
z = tanh(z)                          # [-1, 1]
z = (z + 1) / 2 * (num_bins - 1)    # [0, num_bins-1], e.g. [0, 3]
```

**Step 3**: Round to nearest integer
```python
z_q = round(z)   # each value is now 0, 1, 2, or 3
```

**Vocabulary size** = `num_bins^L`. With `num_bins=4, L=5`: each position is one of `4^5 = 1024` possible codes (like having a 1024-word vocabulary).

### 3. The Straight-Through Estimator

Here's the problem: `round()` has zero gradient everywhere (it's a step function). If you use it directly, no gradients flow back to the encoder, and it can't learn.

**Solution — straight-through estimator (STE)**:
- **Forward pass**: actually round the values (discrete output)
- **Backward pass**: pretend the rounding didn't happen (copy gradients through)

In PyTorch:
```python
z_q = z + (round(z) - z).detach()
```

This is a clever trick. Let's break it down:
- `round(z) - z` is the rounding "error"
- `.detach()` removes it from the computation graph
- So in the **forward** pass: `z_q = z + (round(z) - z) = round(z)` ✓
- In the **backward** pass: gradient of `z_q` w.r.t. `z` = 1 (the detached part has no gradient) ✓

The encoder learns to push values close to integer boundaries, and gradients flow as if the rounding wasn't there.

### 4. Converting Latents to Token Indices

Each spatial position has L quantized values, each in `{0, 1, ..., num_bins-1}`. To get a single index (like a word ID), treat it as a mixed-radix number:

```python
# L=5, num_bins=4
# quantized values at one position: [2, 0, 3, 1, 2]
# index = 2*4^4 + 0*4^3 + 3*4^2 + 1*4^1 + 2*4^0
#       = 512 + 0 + 48 + 4 + 2 = 566

# Vectorized:
basis = num_bins ** torch.arange(L)  # [1, 4, 16, 64, 256]
indices = (quantized * basis).sum(dim=-1)  # single integer per position
```

This gives you `(B, 8, 8)` indices where each value is in `[0, 1023]` — your image tokens.

### 5. FSQ vs VQ-VAE (Why FSQ Is Better)

You may have heard of VQ-VAE, which uses a learnable codebook. FSQ is simpler and avoids VQ-VAE's problems:

| | VQ-VAE | FSQ |
|---|--------|-----|
| Codebook | Learned embeddings | Implicit (all combinations of rounded scalars) |
| Collapse | Common — many codes unused | Impossible — all codes reachable by construction |
| Extra losses | Needs commitment loss + EMA updates | None |
| Implementation | Complex | ~10 lines of code |

---

## Your Task

### Model

Take your conv autoencoder from Iteration 1 and add FSQ quantization:

**Encoder** (same as before, but output L channels instead of 16):
```
(B, 3, 64, 64)
 → Conv2d(3, 32, 4, stride=2, padding=1) + ReLU     → (B, 32, 32, 32)
 → Conv2d(32, 64, 4, stride=2, padding=1) + ReLU     → (B, 64, 16, 16)
 → Conv2d(64, 128, 4, stride=2, padding=1) + ReLU    → (B, 128, 8, 8)
 → Conv2d(128, L, 3, stride=1, padding=1)             → (B, L, 8, 8)  ← L=5
```

**FSQ layer** (new):
```python
def quantize(z, num_bins=4):
    z = torch.tanh(z)                              # [-1, 1]
    z = (z + 1) / 2 * (num_bins - 1)              # [0, num_bins-1]
    z_q = z + (z.round() - z).detach()             # straight-through round
    return z_q
```

**Decoder** (takes L channels as input instead of 16):
```
(B, L, 8, 8)
 → ConvTranspose2d(L, 128, 3, stride=1, padding=1) + ReLU    → (B, 128, 8, 8)
 → ConvTranspose2d(128, 64, 4, stride=2, padding=1) + ReLU   → (B, 64, 16, 16)
 → ConvTranspose2d(64, 32, 4, stride=2, padding=1) + ReLU    → (B, 32, 32, 32)
 → ConvTranspose2d(32, 3, 4, stride=2, padding=1) + Sigmoid  → (B, 3, 64, 64)
```

### Training
- Same as Iteration 1: reconstruct the input frame
- Loss: `F.smooth_l1_loss(reconstruction, original)`
- Adam, lr=1e-3, ~20 epochs
- The straight-through estimator means training works exactly the same — gradients flow through the quantization

### Visualization & Analysis

1. Show original vs reconstruction (should be similar quality to Iteration 1)
2. **Inspect quantized values**: encode a frame, look at the quantized latent — all values should be integers in {0, 1, 2, 3}
3. **Compute token indices**: convert the L-dimensional quantized vectors to single indices. Print the 8×8 grid of token IDs for a frame
4. **Codebook utilization**: across a batch of frames, how many unique token IDs appear out of the possible 1024? (Should be high with FSQ)

### What to Observe

1. Reconstruction quality is comparable to Iteration 1 despite the discrete bottleneck — FSQ preserves information well
2. The latent values are now **exactly** integers {0, 1, 2, 3} — discrete!
3. Each frame is now represented as 64 tokens (8×8 grid) from a vocabulary of 1024. This is directly analogous to a 64-word sentence from a 1024-word vocabulary
4. But... you're still using a CNN. Transformers are better at learning global patterns. And you can't handle video (multiple frames) yet
