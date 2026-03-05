# Iteration 4: Space-Time Transformer

## Goal

Extend the single-frame ViT tokenizer to handle **video clips** by factoring attention into spatial (within each frame) and temporal (across frames) components. This completes the Video Tokenizer — Stage 1 of TinyWorlds.

## Why This Iteration Exists

In Iteration 3, you built a ViT that processes one frame at a time. But video frames are heavily correlated — the ball in Pong moves a few pixels between frames, and the background doesn't change at all. Processing frames independently wastes this structure.

The naive approach (concatenate all patches from all frames into one long sequence) doesn't scale:
- 4 frames × 64 patches = 256 tokens → attention is O(256²) = 65,536 operations
- 16 frames × 64 patches = 1024 tokens → O(1024²) = 1,048,576 operations

Factored space-time attention is much cheaper and captures both spatial and temporal patterns.

## Concepts You Need

### 1. The Space-Time Factorization

Instead of one big attention over all tokens, alternate between two smaller attentions:

**Spatial attention** — "What's happening in this frame?"
```
Input: (B, T, P, E) → reshape to (B*T, P, E)
Each frame's P patches attend to each other independently
Same frame only — no cross-frame information
```

**Temporal attention** — "What happened at this position over time?"
```
Input: (B, T, P, E) → transpose to (B, P, T, E) → reshape to (B*P, T, E)
Each patch position attends to the same position across all frames
Apply causal mask — can only look at current and past frames
```

**Computational cost**:
- Full attention: O(T²P² × E) — quadratic in total tokens
- Factored: O(TP² × E) + O(PT² × E) = O(TP(P+T) × E) — much smaller when T and P are both moderate

### 2. Causal Masking in Temporal Attention

For generation (predicting future frames), the temporal attention must be **causal** — frame 3 can only attend to frames 0, 1, 2, 3, not frame 4.

```python
# Causal mask for T timesteps
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
# True means "block this attention"
#  [[False,  True,  True,  True],
#   [False, False,  True,  True],
#   [False, False, False,  True],
#   [False, False, False, False]]
```

Spatial attention is **bidirectional** (all patches in a frame can see each other). Only temporal attention is causal.

### 3. Space-Time Transformer Block

Each block has three sub-layers instead of two:

```python
class SpaceTimeBlock(nn.Module):
    def forward(self, x, T, P):
        # x: (B, T*P, E)
        B_orig = x.shape[0]

        # 1. Spatial attention (within each frame)
        x = x.reshape(B_orig * T, P, E)
        x = x + self.spatial_attn(self.ln1(x))  # no mask — bidirectional
        x = x.reshape(B_orig, T * P, E)

        # 2. Temporal attention (across frames, per patch position)
        x = x.reshape(B_orig, T, P, E)
        x = x.permute(0, 2, 1, 3).reshape(B_orig * P, T, E)
        x = x + self.temporal_attn(self.ln2(x), causal_mask)  # causal!
        x = x.reshape(B_orig, P, T, E).permute(0, 2, 1, 3).reshape(B_orig, T * P, E)

        # 3. FFN
        x = x + self.ffn(self.ln3(x))
        return x
```

The reshaping is the trickiest part. The key insight:
- **Spatial**: merge B and T so each "batch item" is one frame → `(B*T, P, E)`
- **Temporal**: merge B and P so each "batch item" is one patch position → `(B*P, T, E)`

### 4. Positional Encoding for Video

For video, you need both spatial and temporal position information:

```python
# Spatial: which patch position? (same for all frames)
spatial_pos = nn.Parameter(torch.randn(1, 1, P, d_model) * 0.02)  # (1, 1, P, E)

# Temporal: which frame? (same for all patches)
temporal_pos = nn.Parameter(torch.randn(1, T, 1, d_model) * 0.02)  # (1, T, 1, E)

# Combined: broadcast addition
pos = spatial_pos + temporal_pos  # (1, T, P, E)
tokens = tokens + pos  # add to patch embeddings
```

### 5. Video Dataset

Now each sample is a **clip** of T consecutive frames, not a single frame:

```python
class VideoClipDataset(Dataset):
    def __init__(self, frames, num_frames=4, frame_skip=2):
        self.frames = frames
        self.num_frames = num_frames
        self.frame_skip = frame_skip

    def __len__(self):
        return len(self.frames) - (self.num_frames * self.frame_skip)

    def __getitem__(self, idx):
        clip = self.frames[idx:idx + self.num_frames * self.frame_skip:self.frame_skip]
        return clip, clip  # (T, 3, 64, 64), autoencoder target
```

`frame_skip` controls temporal resolution — skip every N frames to get more visual change between consecutive clip frames.

### 6. Decoder for Video

The decoder takes quantized tokens `(B, T, P, L)` and reconstructs all T frames. It uses the same space-time transformer blocks with a pixel head that outputs `(B, T, 3, 64, 64)`.

---

## Your Task

### Data
- Load Pong frames
- Build a `VideoClipDataset` with `num_frames=4, frame_skip=2`
- Each sample: `(4, 3, 64, 64)` clip → target is the same clip

### Model

**Space-Time Transformer Autoencoder**:

Config:
- `patch_size=8`, `d_model=128`, `n_heads=4`, `n_layers=2`
- `latent_dim=5`, `num_bins=4`
- `num_frames=4`

Architecture:
```
ENCODER:
  clip (B, T, 3, 64, 64)
  → PatchEmbed each frame → (B, T, P, d_model)    [P=64]
  → Add spatial + temporal positional embeddings
  → Flatten to (B, T*P, d_model)
  → N × SpaceTimeBlock (spatial attn + causal temporal attn + FFN)
  → Reshape to (B, T, P, d_model)
  → LatentHead: Linear → (B, T, P, L=5)
  → FSQ → (B, T, P, L=5) discrete, indices (B, T, P)

DECODER:
  quantized (B, T, P, L=5)
  → LatentEmbed: Linear → (B, T, P, d_model)
  → Add spatial + temporal positional embeddings
  → Flatten to (B, T*P, d_model)
  → N × SpaceTimeBlock
  → Reshape to (B, T, P, d_model)
  → PixelHead → (B, T, 3, 64, 64)
  → Sigmoid
```

### Training
- Loss: `F.smooth_l1_loss(reconstruction, original)` over all T frames
- Adam, lr=1e-4
- ~30 epochs

### Visualization

1. For one clip, show all 4 original frames in a row, then all 4 reconstructions below
2. Print the token grid for each frame — notice how similar consecutive frames have similar token patterns
3. Compare reconstruction quality to Iteration 3 (single frame)

### What to Observe

1. **Video reconstruction works** — the model reconstructs all 4 frames, not just one
2. **Temporal coherence** — consecutive frames have similar token grids, showing the model understands temporal continuity
3. **You now have TinyWorlds' complete Video Tokenizer** — it compresses video clips into discrete tokens
4. **The limitation**: this is still just compression (autoencoding). You can reconstruct, but you can't **generate** new frames. For that, you need a second model that predicts future tokens — like GPT predicts next words
