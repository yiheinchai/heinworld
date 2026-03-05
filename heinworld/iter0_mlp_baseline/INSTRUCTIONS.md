# Iteration 0: Predict the Next Frame with an MLP

## Goal

Build the simplest possible model that predicts the next video frame from the current one. This is your "hello world" for video prediction.

## Concepts You Need

### 1. Video as Data

A video is a sequence of images (frames). Each frame is a 3D array:

```
frame.shape = (Height, Width, Channels) = (64, 64, 3)
```

- **Height × Width** = spatial resolution (64×64 pixels)
- **Channels** = 3 color channels (Red, Green, Blue)
- Each pixel value is an integer 0–255 (uint8)

A video with N frames is just a 4D array: `(N, 64, 64, 3)`.

### 2. HDF5 Files

HDF5 is a format for storing large arrays on disk. Think of it as a dictionary of numpy arrays saved to a single file.

```python
import h5py

# Open and read
with h5py.File("pong_frames.h5", "r") as f:
    # f["frames"] is a lazy reference — data stays on disk
    print(f["frames"].shape)   # (N, 64, 64, 3)
    print(f["frames"].dtype)   # uint8

    # Slicing loads only what you ask for into RAM
    first_frame = f["frames"][0]        # shape: (64, 64, 3)
    first_100 = f["frames"][:100]       # shape: (100, 64, 64, 3)

    # Load everything into memory
    all_frames = f["frames"][:]         # shape: (N, 64, 64, 3)
```

**Why HDF5 instead of a folder of PNGs?** One file operation to read any slice, rather than thousands of individual file opens. Also supports compression.

### 3. Preparing Image Data for PyTorch

Raw pixel data needs three transformations:

```
Step 1 — Normalize:     uint8 [0, 255]  →  float32 [0.0, 1.0]
                        data.astype(np.float32) / 255.0

Step 2 — Channels first: (H, W, C)  →  (C, H, W)
                        PyTorch convention. Use .permute() or np.transpose()

Step 3 — Batch dimension: (C, H, W)  →  (B, C, H, W)
                        DataLoader handles this automatically
```

Why channels first? PyTorch's conv layers and other operations expect `(Batch, Channels, Height, Width)`. Images are stored as `(Height, Width, Channels)`. You rearrange with:

```python
# numpy: (N, H, W, C) → (N, C, H, W)
frames = np.transpose(frames, (0, 3, 1, 2))

# torch: same thing
frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
```

### 4. PyTorch Dataset and DataLoader

A **Dataset** defines what one sample looks like. A **DataLoader** batches and shuffles samples.

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return one sample (input, target)
        return self.data[idx], self.data[idx + 1]

dataset = MyDataset(data)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch_input, batch_target in loader:
    # batch_input.shape = (64, ...)
    ...
```

For frame prediction, each sample is a pair: `(frame_t, frame_t+1)`. If you have N frames, you have N-1 valid pairs.

### 5. Flattening for MLP

An MLP (Multi-Layer Perceptron) takes a 1D vector as input. A frame is 3D: `(3, 64, 64)`. You flatten it:

```python
flat = frame.reshape(-1)  # shape: (3 * 64 * 64,) = (12288,)
```

This throws away all spatial structure — pixel (0,0) and pixel (63,63) are just numbers at different positions in a long vector. This is a major limitation we'll address in later iterations.

### 6. Training Loop Basics

If you've built GPT-1, this is familiar. The key pieces:

```python
model = MLP(...)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.SmoothL1Loss()  # or MSELoss

for epoch in range(num_epochs):
    for input_batch, target_batch in loader:
        prediction = model(input_batch)
        loss = loss_fn(prediction, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Why Smooth L1 instead of MSE?** MSE penalizes large errors quadratically, which makes training unstable when predictions are far off (early in training). Smooth L1 is linear for large errors and quadratic for small ones — more forgiving. Either works for this iteration.

### 7. MPS Backend (Apple Silicon)

On your M2 MacBook, use the MPS (Metal Performance Shaders) backend instead of CUDA:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

This runs on your GPU cores. Expect ~2-5x speedup over CPU for this small model.

---

## Your Task

### Data
- Load `pong_frames.h5` from `../data/pong_frames.h5`
- Build a Dataset that returns `(frame_t, frame_t+1)` pairs as flattened float tensors

### Model
- 2-layer MLP: `Linear(12288, 256) → ReLU → Linear(256, 12288)`
- Input: flattened frame_t `(12288,)`
- Output: predicted frame_t+1 `(12288,)`

### Training
- Optimizer: Adam, lr=1e-3
- Loss: SmoothL1Loss
- Train for ~10 epochs
- Print loss every epoch

### Visualization
- After training, pick 5 random frame pairs from the data
- For each, show side by side: input frame, predicted next frame, actual next frame
- Use matplotlib: `plt.imshow()` expects `(H, W, C)` with values in [0, 1]
- Remember to reshape from flat `(12288,)` back to `(3, 64, 64)` then permute to `(64, 64, 3)` for display

### What to Observe

After training, look at your predictions critically:
1. Are they sharp or blurry? Why?
2. How many parameters does your model have? (hint: count them with `sum(p.numel() for p in model.parameters())`)
3. What would happen if you tried 128×128 frames instead of 64×64?

These observations motivate the next iteration.
