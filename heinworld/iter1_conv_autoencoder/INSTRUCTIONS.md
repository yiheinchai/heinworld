# Iteration 1: Convolutional Autoencoder

## Goal

Learn to **compress** a frame into a small latent representation and **reconstruct** it back. This is the foundation for everything that follows — before you can predict the future, you need an efficient way to represent frames.

## Why This Iteration Exists

In Iteration 0, the MLP had two major problems:
1. **No spatial structure** — it treated each pixel independently
2. **Huge parameter count** — 12,288 input dimensions, millions of parameters

Convolutions solve both: they process local patches of pixels and share weights across the image.

## Concepts You Need

### 1. Convolutions (Conv2d)

A convolution slides a small filter (e.g., 3×3) across the image, computing a dot product at each position. This means:

- **Local connectivity**: each output pixel depends only on a small neighborhood of input pixels
- **Weight sharing**: the same filter is used at every position (translation invariance)
- **Much fewer parameters**: a 3×3 filter with 16 output channels = `3×3×3×16 = 432` parameters, vs millions for an MLP

```python
# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
```

- `in_channels=3`: RGB input
- `out_channels=16`: produces 16 feature maps
- `kernel_size=3`: 3×3 filter
- `stride=1`: move filter 1 pixel at a time
- `padding=1`: pad input so output has same spatial size

**Stride for downsampling**: `stride=2` moves the filter 2 pixels at a time, halving spatial dimensions:
```
Input:  (B, 3, 64, 64)
Conv2d(3, 16, 4, stride=2, padding=1)
Output: (B, 16, 32, 32)   ← spatial size halved, channels increased
```

### 2. Transposed Convolutions (ConvTranspose2d)

The reverse of a strided convolution — **upsamples** spatial dimensions:

```python
nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
# Input:  (B, 16, 32, 32)
# Output: (B, 3, 64, 64)  ← spatial size doubled
```

This is used in the decoder to go from small latent back to full resolution.

### 3. Autoencoder Architecture

An autoencoder has two halves:

```
Input frame → [ENCODER] → small latent → [DECODER] → reconstructed frame
  (3,64,64)              (d,8,8)                      (3,64,64)
```

The **bottleneck** (small latent) forces the model to learn a compressed representation. It can't just memorize the input — it has to find the essential information.

**Encoder**: repeatedly downsample with strided convolutions
```
(3, 64, 64) → (32, 32, 32) → (64, 16, 16) → (128, 8, 8) → latent
```

**Decoder**: repeatedly upsample with transposed convolutions
```
latent → (128, 8, 8) → (64, 16, 16) → (32, 32, 32) → (3, 64, 64)
```

### 4. Reconstruction Loss

The training objective is simple: make the output match the input.

```python
loss = F.smooth_l1_loss(reconstructed_frame, original_frame)
```

There are no labels. The model learns to compress and decompress by being forced through the bottleneck. This is **self-supervised** learning.

### 5. Choosing the Bottleneck Size

The bottleneck determines the compression ratio:

```
Input:      3 × 64 × 64 = 12,288 values
Bottleneck: 16 × 8 × 8  = 1,024 values  → 12x compression
Bottleneck: 32 × 8 × 8  = 2,048 values  → 6x compression
```

Too small → reconstructions are blurry (too much information lost)
Too large → no compression, model just copies through

A good starting point: `16 channels × 8×8 spatial = 1,024 latent values`.

### 6. Activation Functions

Use **ReLU** (or GELU/SiLU) between conv layers in the encoder/decoder. But **no activation** on the final decoder output — you want the model to output any value in [0, 1], not just positive values.

```python
# Encoder block
nn.Conv2d(32, 64, 4, stride=2, padding=1),
nn.ReLU(),

# Last decoder layer — no activation
nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
nn.Sigmoid(),  # or just clamp output to [0,1]
```

`Sigmoid` on the final layer squashes output to [0, 1] which matches your normalized pixel range.

---

## Your Task

### Data
- Reuse the same Pong data loading from Iteration 0
- But this time, each sample is just **one frame** (not a pair) — the target is the same frame (autoencoder reconstructs its input)
- Keep frames as `(3, 64, 64)` — no flattening needed

### Model

Build a convolutional autoencoder:

**Encoder** (4 layers, each halves spatial dims):
```
(B, 3, 64, 64)
 → Conv2d(3, 32, 4, stride=2, padding=1) + ReLU     → (B, 32, 32, 32)
 → Conv2d(32, 64, 4, stride=2, padding=1) + ReLU     → (B, 64, 16, 16)
 → Conv2d(64, 128, 4, stride=2, padding=1) + ReLU    → (B, 128, 8, 8)
 → Conv2d(128, 16, 3, stride=1, padding=1)            → (B, 16, 8, 8)  ← bottleneck
```

**Decoder** (mirrors the encoder):
```
(B, 16, 8, 8)
 → ConvTranspose2d(16, 128, 3, stride=1, padding=1) + ReLU   → (B, 128, 8, 8)
 → ConvTranspose2d(128, 64, 4, stride=2, padding=1) + ReLU   → (B, 64, 16, 16)
 → ConvTranspose2d(64, 32, 4, stride=2, padding=1) + ReLU    → (B, 32, 32, 32)
 → ConvTranspose2d(32, 3, 4, stride=2, padding=1) + Sigmoid  → (B, 3, 64, 64)
```

### Training
- Optimizer: Adam, lr=1e-3
- Loss: `F.smooth_l1_loss(reconstruction, original)`
- Train for ~20 epochs
- Print loss every epoch

### Visualization
1. Show 5 random frames: original vs reconstruction side by side
2. **Inspect the latent space**: encode a frame, print the latent tensor's min, max, mean, std — notice the values are arbitrary continuous floats

### What to Observe

After training:
1. Are the reconstructions sharp? How do they compare to the MLP predictions from Iteration 0?
2. How many parameters does this model have vs the MLP? (should be much fewer)
3. Encode two very different frames — look at their latent vectors. Is there any structure? Could you interpolate between them?
4. Try generating a frame from a **random** latent vector (random noise at the bottleneck shape). What comes out?

The key realization: the latent space is continuous and unstructured. You can't easily sample from it or use it as a discrete vocabulary like words in a language model. That's the problem for the next iteration.
