# Iteration 7: Latent Action Model — Unsupervised Action Discovery

## Goal

Build a model that **discovers discrete actions** from consecutive frames without any labels. This is Stage 2 of TinyWorlds and the key innovation from DeepMind's Genie paper.

## Why This Iteration Exists

In Iteration 6, you used handcrafted heuristics to extract Pong actions. This doesn't scale — you can't write heuristics for every game, and internet videos don't come with controller inputs.

The insight: if frame_t shows the paddle at position A and frame_t+1 shows it at position B, *something* caused that change. We don't need to know the button press — we just need a compressed representation of "what changed." If we force this representation through a tiny discrete bottleneck, it naturally becomes an "action code."

## Concepts You Need

### 1. The Information Bottleneck Argument

Consider two frames. The **difference** between them contains:
- Player actions (paddle moved up)
- Game physics (ball bounced)
- Environmental changes (score updated)

If you force this difference through a tiny bottleneck (e.g., 3 bits = 8 possible codes), only the **most important factor of variation** survives. In Pong, the dominant factor is the paddle direction — so the 8 codes naturally correspond to action-like concepts.

### 2. Latent Action Model Architecture

It's an autoencoder, but asymmetric:

```
ACTION ENCODER:
  [frame_t, frame_t+1] → encode → FSQ quantize → action_code (tiny: 3 bits)

ACTION DECODER:
  [frame_t, action_code] → predict frame_t+1
```

The encoder sees BOTH frames and compresses "what changed" into a discrete code.
The decoder sees only frame_t plus the code, and must reconstruct frame_t+1.

### 3. Why the Decoder Design Matters

The decoder gets `frame_t` as context. If it also got `frame_t+1` or even strong temporal context, it could just **ignore the action** and copy/interpolate. The action code would carry no information (mode collapse).

**Solution — masking trick**: in the decoder, mask out most frames. Only give it:
- The first frame (distant past)
- The action code

This forces the decoder to actually **use** the action code to predict what comes next, because it can't just copy from recent context.

### 4. Variance Penalty — Preventing Action Collapse

Even with the masking trick, the model might collapse: encode every frame pair to the same action code. This is a local minimum where the decoder learns to predict "the average next frame" regardless of action.

**Fix**: add a penalty that encourages high variance in the action latents:

```python
# action_latents: (B, action_dim) — pre-quantization continuous values
variance = action_latents.var(dim=0).mean()  # variance across the batch
variance_penalty = 1.0 / (variance + 1e-6)  # high penalty when variance is low
```

This pushes the encoder to spread actions across the code space.

### 5. Binary FSQ for Actions

For the action tokenizer, use **num_bins=2** (binary) instead of 4:
- Each latent dimension is 0 or 1
- With `action_dim=3`: vocabulary = 2³ = **8 possible actions**
- This is intentionally tiny — actions are simple (direction choices), not complex visual patterns

Compare:
- Video tokenizer: `num_bins=4, latent_dim=5` → 1024 visual tokens (complex)
- Action tokenizer: `num_bins=2, latent_dim=3` → 8 action tokens (simple)

### 6. Training the Latent Action Model

```python
# Forward pass:
action_latent = action_encoder(frame_t, frame_t_plus_1)  # sees both frames
action_code, action_idx = fsq_binary(action_latent)       # quantize to 0/1
predicted_frame = action_decoder(frame_t, action_code)     # predicts from frame_t + code

# Losses:
reconstruction_loss = F.l1_loss(predicted_frame, frame_t_plus_1)
variance_loss = 1.0 / (action_latent.var(dim=0).mean() + 1e-6)

total_loss = reconstruction_loss + 0.1 * variance_loss
```

### 7. How This Connects to the Full Pipeline

After training:
1. **During dynamics training** (Iteration 8): use the frozen action encoder to infer actions between consecutive frames. These replace the handcoded actions from Iteration 6.
2. **During inference**: the user selects an action index (0-7). Convert it to the corresponding action code and feed it to the dynamics model via FiLM.

---

## Your Task

### Model

**Action Encoder** (sees two frames, outputs action code):
```
[frame_t, frame_t+1] concatenated along channel dim → (B, 6, 64, 64)
 → PatchEmbed → (B, P, d_model)
 → Transformer blocks (2 layers, bidirectional)
 → Mean pool across patches → (B, d_model)
 → Linear(d_model, action_dim=3) → (B, 3)
 → Binary FSQ (num_bins=2) → (B, 3) discrete {0, 1}
```

**Action Decoder** (sees frame_t + action code, predicts frame_t+1):
```
frame_t → PatchEmbed → (B, P, d_model)
action_code (B, 3) → Linear(3, d_model) → broadcast to all patches
 → Add action embedding to patch embeddings
 → Transformer blocks (2 layers)
 → Pixel head → (B, 3, 64, 64)
 → Sigmoid
```

Or use the same FiLM approach from Iteration 6 to inject the action code.

### Data
- Dataset returns `(frame_t, frame_t+1)` pairs
- No action labels needed!

### Training
- Loss: `L1 reconstruction + 0.1 * variance penalty`
- Adam, lr=1e-4, ~30 epochs

### Visualization & Analysis

1. Reconstruct frame_t+1 from frame_t + inferred action. How good is it?
2. **Action distribution**: encode 1000 frame pairs, plot a histogram of the 8 action codes. Are all codes used? (High utilization = variance penalty working)
3. **Action semantics**: group frame pairs by their inferred action code. For each code, show 5 example pairs. Do frames within a group share a common "action" (e.g., all show paddle moving up)?
4. Compare inferred actions to the heuristic actions from Iteration 6. Do they correlate?

### What to Observe

1. **The model discovers meaningful actions** — different codes capture different types of frame transitions. Some codes mean "ball moving left," others mean "paddle up"
2. **No labels were used** — the discrete actions emerged purely from the information bottleneck
3. **8 codes is enough for Pong** — the game has simple dynamics. More complex games might need more bits
4. The remaining problem: the dynamics model from Iteration 5 decodes tokens one-at-a-time (autoregressive). With 64 tokens per frame, that's 64 sequential steps. Can we do it faster?
