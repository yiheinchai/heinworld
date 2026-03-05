# Iteration 9: Full TinyWorlds Assembly

## Goal

Combine all components into the complete three-stage TinyWorlds pipeline: train sequentially, then run interactive inference where you control a generated game world.

## Why This Iteration Exists

You've built every component individually. Now you put them together into a working system — the same architecture as DeepMind's Genie. This iteration is about **integration**, not new concepts.

## The Three-Stage Pipeline

### Stage 1: Video Tokenizer (Iterations 3-4)
- **Trains on**: raw video frames
- **Learns**: compress frames into discrete tokens and reconstruct
- **After training**: freeze weights, used by Stages 2 and 3
- **Architecture**: Space-Time Transformer encoder + FSQ + Space-Time Transformer decoder

### Stage 2: Latent Action Model (Iteration 7)
- **Trains on**: raw video frame pairs
- **Learns**: infer discrete action codes between consecutive frames
- **After training**: freeze weights, used by Stage 3 for action labels and by inference for action control
- **Architecture**: Transformer encoder (sees 2 frames) + Binary FSQ + Transformer decoder

### Stage 3: MaskGIT Dynamics (Iteration 8)
- **Trains on**: video tokens (from frozen Stage 1) + inferred actions (from frozen Stage 2)
- **Learns**: predict next frame tokens given context + action
- **Architecture**: Space-Time Transformer with FiLM action conditioning + MaskGIT masked prediction

### Training Order

```
Stage 1 → save checkpoint → freeze
Stage 2 → save checkpoint → freeze
Stage 3 (uses frozen Stage 1 + Stage 2) → save checkpoint
```

Each stage is independent in terms of training — you don't backpropagate through frozen stages.

## Full Inference Pipeline

```python
def interactive_generation(tokenizer, action_model, dynamics, seed_frames, num_generate=20):
    """
    1. Encode seed frames → video tokens
    2. For each step:
       a. User selects action (0-7) or use random
       b. Dynamics model predicts next frame tokens (MaskGIT)
       c. Decode tokens → pixels with tokenizer decoder
       d. Append to context, slide window
    3. Display generated video
    """
    # Encode seed
    with torch.no_grad():
        z_q, context_tokens = tokenizer.encode(seed_frames)  # (1, T_seed, P)

    generated_frames = []

    for step in range(num_generate):
        # Choose action (interactive or random)
        action_id = random.randint(0, 7)  # or get from keyboard input

        # Prepare action tensor
        T_ctx = context_tokens.shape[1]
        actions = torch.full((1, T_ctx + 1), action_id, device=device)

        # Predict next frame tokens via MaskGIT
        next_tokens, _ = generate_frame_maskgit(
            dynamics, context_tokens, actions, num_steps=8
        )  # (1, P)

        # Decode tokens → pixels
        next_pixels = tokens_to_pixels(next_tokens.unsqueeze(1))  # (1, 1, 3, H, W)
        generated_frames.append(next_pixels[0, 0].cpu())

        # Update context: append new tokens, slide window to keep fixed length
        context_tokens = torch.cat([
            context_tokens[:, 1:],           # drop oldest frame
            next_tokens.unsqueeze(1)          # add new frame
        ], dim=1)

    return generated_frames
```

## Your Task

### 1. Organize the Code

Create clean, importable modules:

```
heinworld/
├── data/
│   └── pong_frames.h5
├── models/
│   ├── fsq.py              — FSQ and BinaryFSQ
│   ├── transformer.py       — TransformerBlock, SpaceTimeBlock, FiLM
│   ├── video_tokenizer.py   — SpaceTimeTokenizer (Stage 1)
│   ├── latent_actions.py    — LatentActionModel (Stage 2)
│   └── dynamics.py          — MaskGITDynamics (Stage 3)
├── train_stage1.py          — Train video tokenizer
├── train_stage2.py          — Train latent action model
├── train_stage3.py          — Train dynamics (loads frozen stage 1 + 2)
├── inference.py             — Full interactive inference
└── checkpoints/             — Saved model weights
```

### 2. Train All Three Stages

Run them in order:
```bash
python train_stage1.py   # → saves checkpoints/tokenizer.pt
python train_stage2.py   # → saves checkpoints/action_model.pt
python train_stage3.py   # → saves checkpoints/dynamics.pt
```

### 3. Run Inference

```bash
python inference.py
```

This should:
1. Load all three frozen models
2. Load a seed clip from the dataset
3. Generate 20+ frames with random or user-selected actions
4. Save as a grid of images or animated GIF

### 4. Visualization

Create a figure showing:
- **Row 1**: seed frames (3-4 frames)
- **Row 2-4**: generated frames with different action sequences
- **Label** each generated frame with the action used

### What to Observe

1. **The full pipeline works end-to-end** — from raw pixels to generated game worlds
2. **Actions produce meaningful control** — different action codes lead to visibly different generated futures
3. **Quality degrades over long horizons** — after 10+ generated frames, the scene may drift. This is expected for a small model on limited data
4. **Every component was necessary** — you've felt each problem firsthand and understand why every design choice exists

## Congratulations

You've rebuilt TinyWorlds from scratch. You now understand:
- Image tokenization (patches → transformer → FSQ)
- Video modeling (space-time factored attention)
- World dynamics (next-token prediction on visual tokens)
- Action conditioning (FiLM)
- Unsupervised action discovery (information bottleneck)
- Fast parallel decoding (MaskGIT)

You can now read the [Genie paper](https://arxiv.org/abs/2402.15391) and understand every design decision. You can also explore the original tinyworlds codebase in `../tinyworlds/` and see how these same ideas scale with more engineering.

## Where to Go Next

- **Scale up**: try 128×128 resolution, more transformer layers, longer clips
- **New datasets**: try Sonic or Zelda instead of Pong
- **Improvements from the original tinyworlds README**:
  - RoPE (Rotary Positional Encoding) instead of learned positional embeddings
  - SwiGLU FFN instead of GELU
  - RMSNorm instead of LayerNorm
  - Mixture of Experts (MoE) for scaling
  - Better training: gradient accumulation, mixed precision, torch.compile
