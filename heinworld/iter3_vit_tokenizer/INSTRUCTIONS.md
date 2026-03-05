# Iteration 3: Vision Transformer Tokenizer

## Goal

Replace the CNN encoder-decoder with a **transformer-based** architecture using patch embeddings. This is the single-frame version of TinyWorlds' Video Tokenizer (Stage 1).

## Why This Iteration Exists

CNNs work but have limited receptive fields — a 3×3 filter only sees 3×3 pixels. Even stacked, the receptive field grows slowly. Transformers with self-attention see **everything at once** — every patch attends to every other patch. This matters for capturing global patterns (e.g., the ball and paddle positions in Pong are far apart but related).

More importantly, you already know transformers from building GPT. By replacing the CNN with a transformer, you're applying your existing knowledge to vision and setting up for the space-time transformer in the next iteration.

## Concepts You Need

### 1. Patch Embedding — Tokenizing an Image

Instead of processing individual pixels, divide the image into non-overlapping patches and treat each patch as a "token":

```
64×64 image with patch_size=8:
→ 8×8 = 64 patches, each is 8×8×3 = 192 pixel values
→ Linear projection: 192 → d_model (e.g., 128)
→ Result: 64 tokens, each a 128-dim vector
```

This is identical to how GPT converts word IDs into embedding vectors. Here, a patch of pixels is the "word" and the linear projection is the "embedding layer."

In PyTorch, a single Conv2d does this efficiently:
```python
# kernel_size=stride=patch_size means non-overlapping patches
patch_embed = nn.Conv2d(3, d_model, kernel_size=8, stride=8)
# Input:  (B, 3, 64, 64)
# Output: (B, d_model, 8, 8)  → reshape to (B, 64, d_model)
```

### 2. Positional Encoding

After patch embedding, the transformer has 64 tokens but no idea where each patch came from spatially. You need positional encodings.

**Sinusoidal 2D positional encoding**: extend the 1D sinusoidal encoding from your GPT to 2D. For an 8×8 grid, each position (row, col) gets an encoding:

```python
def sinusoidal_2d(grid_h, grid_w, d_model):
    pe = torch.zeros(grid_h * grid_w, d_model)
    position_h = torch.arange(grid_h).unsqueeze(1).repeat(1, grid_w).flatten()
    position_w = torch.arange(grid_w).repeat(grid_h)

    div_term = torch.exp(torch.arange(0, d_model // 2, 2) * -(math.log(10000.0) / (d_model // 2)))

    # First half: height encoding
    pe[:, 0::4] = torch.sin(position_h.unsqueeze(1) * div_term)
    pe[:, 1::4] = torch.cos(position_h.unsqueeze(1) * div_term)
    # Second half: width encoding
    pe[:, 2::4] = torch.sin(position_w.unsqueeze(1) * div_term)
    pe[:, 3::4] = torch.cos(position_w.unsqueeze(1) * div_term)

    return pe  # (grid_h * grid_w, d_model)
```

**Simpler alternative for this iteration**: just use learned positional embeddings:
```python
self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
# Add to patch embeddings: tokens = patch_embeds + self.pos_embed
```

### 3. Transformer Encoder (You Know This)

Standard transformer blocks — identical to GPT but without causal masking (for single-frame encoding, each patch can see all other patches):

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x
```

Key difference from GPT: **no causal mask** in attention. For encoding a single image, every patch should see every other patch (bidirectional attention).

### 4. Latent Head — Projecting to FSQ Dimensions

After the transformer encoder, project each token from d_model down to the FSQ latent dimension L:

```python
latent_head = nn.Linear(d_model, L)  # d_model → 5
# (B, 64, d_model) → (B, 64, 5) → FSQ quantize → (B, 64, 5) discrete
```

### 5. PixelShuffle Decoder Head

The decoder transformer outputs `(B, 64, d_model)` — 64 tokens of d_model dims. You need to convert this back to pixels: `(B, 3, 64, 64)`.

**PixelShuffle** (also called sub-pixel convolution) is an efficient upsampling method:

```python
# Project each token to patch_size * patch_size * 3 values
pixel_head = nn.Linear(d_model, patch_size * patch_size * 3)
# (B, 64, d_model) → (B, 64, 192)

# Reshape: (B, 64, 192) → (B, 8, 8, 8, 8, 3) → (B, 3, 64, 64)
# Each token "unfolds" into its 8×8×3 pixel patch
output = output.reshape(B, 8, 8, 8, 8, 3)
output = output.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, 64, 64)
```

Or more simply, use `nn.PixelShuffle`:
```python
# Project to (3 * 8 * 8) channels per patch position
head = nn.Linear(d_model, 3 * patch_size * patch_size)
# Reshape to (B, 3*64, 8, 8) then PixelShuffle with upscale_factor=8
# → (B, 3, 64, 64)
```

### 6. Full Architecture Summary

```
ENCODER:
  frame (B, 3, 64, 64)
  → PatchEmbed: Conv2d(3, d_model, 8, stride=8) → reshape → (B, 64, d_model)
  → Add positional encoding
  → N × TransformerBlock (bidirectional attention)
  → LatentHead: Linear(d_model, L=5) → (B, 64, 5)
  → FSQ quantize → (B, 64, 5) discrete

DECODER:
  quantized (B, 64, 5)
  → LatentEmbed: Linear(L=5, d_model) → (B, 64, d_model)
  → Add positional encoding
  → N × TransformerBlock (bidirectional attention)
  → PixelHead: Linear(d_model, 3*8*8=192) → reshape → (B, 3, 64, 64)
  → Sigmoid
```

---

## Your Task

### Model

Build a Vision Transformer autoencoder with FSQ:

- `patch_size = 8` → 64 patches per frame
- `d_model = 128` (embedding dimension)
- `n_heads = 4`
- `n_layers = 2` (for both encoder and decoder — keep it small for M2)
- `latent_dim = 5, num_bins = 4` (reuse FSQ from Iteration 2)
- Learned positional embeddings (simpler than sinusoidal for now)

### Training
- Same setup: reconstruct single frames
- Loss: `F.smooth_l1_loss`
- Adam, lr=1e-4 (slightly lower — transformers can be less stable than CNNs)
- ~30 epochs

### Visualization

1. Reconstructions: original vs reconstructed (compare quality to the CNN version)
2. Token indices: print the 8×8 token grid (same as Iteration 2)
3. Attention weights (optional but insightful): extract attention weights from one transformer block and visualize as a heatmap. Do nearby patches attend more to each other?

### What to Observe

1. Reconstruction quality should be **similar to the CNN** — transformers aren't magic for single frames, but they set up the architecture for video
2. The token grid should look structured — nearby patches should have similar or smoothly varying tokens
3. You now have the same architecture as TinyWorlds' Video Tokenizer, just for single frames
4. **The limitation**: this processes one frame at a time. To model video, you need temporal attention. But naively attending over all patches across all frames is O((T×P)²) — expensive. How can you factor this?
