# TinyWorlds: Project-Based Learning Curriculum

## What You're Building Toward

**TinyWorlds** is a world model that learns to generate interactive video game environments from unlabeled gameplay videos. It has three stages:
1. **Video Tokenizer** — compresses video frames into discrete tokens (like words)
2. **Latent Action Model** — infers what "action" happened between frames *without labels*
3. **Dynamics Model** — predicts the next frame given past frames + action

The key insight from DeepMind's Genie paper: by learning actions unsupervised, you can train on *any* video from the internet — no controller input needed.

## Your Starting Point

You've built GPT-1 from scratch, so you already know:
- Transformers (attention, positional encoding, layer norms, FFN)
- Autoregressive next-token prediction
- Training loops, loss functions, optimization
- Text tokenization

**The core question this curriculum answers**: *You know how to predict the next word. How do you predict the next video frame?*

## Hardware Note

All iterations are designed to train on your M2 MacBook (MPS backend). We use small resolutions (64x64), short clips, and compact models throughout.

---

## Iteration 0: Predict the Next Frame (Naive Baseline)

### Goal
Get data flowing. Predict the next video frame from the current one using the simplest possible model.

### What You Build
- Load Pong gameplay video → extract frames at 64x64
- Flatten each frame to a vector (64 × 64 × 3 = 12,288 values)
- Train a 2-layer MLP: `frame_t → frame_t+1`
- Visualize predictions vs ground truth

### New Concepts
- **Video as data**: frames as tensors, normalization to [-1, 1]
- **Frame prediction as regression**: smooth L1 / MSE loss on pixels
- **MPS backend**: training on Apple Silicon

### What You'll Notice (Problems)
- Predictions are **extremely blurry** — the model averages over all possible futures
- **12,288 input dimensions** is huge — the model has millions of parameters just for tiny 64x64 frames
- No spatial understanding — pixel (0,0) and pixel (63,63) are treated identically
- Scaling to 128x128 would quadruple parameters

### What This Motivates
→ *"I need a way to exploit the spatial structure of images and compress them into something smaller."*

---

## Iteration 1: Convolutional Autoencoder

### Goal
Learn to compress frames into a small latent representation, then reconstruct them. Exploit spatial structure with convolutions.

### What You Build
- **Encoder**: stack of Conv2d + ReLU + stride-2 downsampling (64x64 → 32x32 → 16x16 → 8x8)
- **Decoder**: stack of ConvTranspose2d to upsample back (8x8 → 16x16 → 32x32 → 64x64)
- Bottleneck: 8×8 spatial grid with `d` channels = 8×8×d latent (e.g., d=16 → 1024 values, 12x compression)
- Train with reconstruction loss: `L1(decoded_frame, original_frame)`

### New Concepts
- **Autoencoders**: learn to compress and decompress
- **Convolutional layers**: exploit spatial locality and translation invariance
- **Latent space**: the compressed representation in the middle
- **Reconstruction loss**: how good is the decompression?

### What You'll Notice (Problems)
- Reconstructions are decent but the latent space is **continuous** — values are arbitrary floats
- If you try to **sample random latents** and decode them, you get garbage
- The latent space has no structure — nearby points don't produce similar images
- You can't easily do "next token prediction" on continuous vectors like you did with GPT

### What This Motivates
→ *"For generation, I need discrete tokens — like a vocabulary for images. How do I force the latent space to be discrete?"*

---

## Iteration 2: Discrete Bottleneck with FSQ

### Goal
Make the autoencoder produce discrete tokens instead of continuous vectors — creating a "visual vocabulary."

### What You Build
- Same conv encoder-decoder from Iteration 1
- **Replace** the continuous bottleneck with **Finite Scalar Quantization (FSQ)**:
  - Encoder outputs L values per spatial position (e.g., L=5)
  - `tanh` squashes each to [-1, 1], then scale to [0, num_bins-1] and **round**
  - Each spatial position becomes a discrete code from a vocabulary of `num_bins^L` (e.g., 4^5 = 1024 tokens)
  - **Straight-through estimator**: copy gradients through the non-differentiable rounding

### New Concepts
- **Why discrete?** Language models work on discrete tokens. To reuse the "predict next token" paradigm from GPT for images, you need discrete visual tokens
- **Finite Scalar Quantization (FSQ)**: simpler and more stable than VQ-VAE's codebook approach — no codebook collapse, no EMA updates, no commitment loss
- **Straight-through estimator**: the key trick — during forward pass you round (non-differentiable), during backward pass you pretend the rounding didn't happen and pass gradients straight through
- **Codebook utilization**: with FSQ, every code is reachable by construction (just combinations of rounded scalars)

### What You'll Notice (Problems)
- Works well for single frames, but each frame is tokenized **independently**
- The model doesn't know that frame_t and frame_t+1 are related
- Conv encoder has a **fixed receptive field** — can't capture long-range dependencies within a frame
- You now have discrete tokens... but they come from a CNN, not a transformer

### What This Motivates
→ *"I have discrete image tokens now. Can I replace the CNN with a transformer — like a Vision Transformer — so I can leverage my GPT knowledge?"*

---

## Iteration 3: Vision Transformer Tokenizer (= Video Tokenizer, Stage 1 of TinyWorlds)

### Goal
Replace the conv encoder-decoder with a transformer-based architecture using patch embeddings. This is the **Video Tokenizer** from TinyWorlds.

### What You Build
- **Patch embedding**: split 64x64 frame into non-overlapping 8x8 patches → 8×8 = 64 patches per frame
- Each patch is linearly projected to an embedding vector (like token embedding in GPT)
- **Add positional encoding** (sinusoidal, 2D) so the transformer knows where each patch is
- **Encoder**: Transformer blocks (self-attention over all 64 patches)
- **FSQ quantization**: project each patch embedding to L dims → quantize → discrete token per patch
- **Decoder**: Transformer blocks + **PixelShuffle** head to reconstruct the frame
- Train with reconstruction loss

### New Concepts
- **Patch embedding**: the "tokenization" step for images — each patch becomes a token (like a word)
- **Vision Transformer (ViT)**: applying your GPT knowledge to images
- **PixelShuffle**: efficient sub-pixel upsampling for the decoder head
- **Connection to GPT**: encoder patches ≈ text tokens, transformer blocks ≈ GPT layers, FSQ codes ≈ vocabulary indices

### What You'll Notice (Problems)
- Works well for **single frames**, but you're still processing frames independently
- To predict the next frame, you'd need to somehow connect frames across time
- The transformer only does **spatial** attention (within one frame)
- You have 64 tokens per frame — for a 16-frame clip, that's 1024 tokens. Full attention over all of them is O(1024²) = expensive

### What This Motivates
→ *"I need temporal modeling. But full attention over space AND time is too expensive. How do I efficiently attend across both?"*

---

## Iteration 4: Space-Time Transformer

### Goal
Extend the single-frame transformer to handle video clips by factoring attention into spatial and temporal components.

### What You Build
- Input: clip of T frames, each with P patches → [B, T, P, E] tensor
- **Space-Time Transformer Block** (replaces standard transformer block):
  1. **Spatial attention**: reshape to [B×T, P, E] — each frame's patches attend to each other
  2. **Temporal attention**: reshape to [B×P, T, E] — each patch position attends across time (causal mask)
  3. **FFN**: standard feed-forward
- Build a video autoencoder: encode T frames → T×P discrete tokens → decode back to T frames
- Train with reconstruction loss on all frames

### New Concepts
- **Factored attention**: instead of O((T×P)²), you get O(T×P² + P×T²) — much cheaper
- **Spatial attention**: "what's happening in this frame?" (global within frame)
- **Temporal attention with causal mask**: "what happened at this position before?" (across time, can only look backward)
- **Why causal?**: for generation, you can't peek at future frames

### What You'll Notice (Problems)
- The video autoencoder reconstructs clips well — it understands both space and time
- But it's still just **compression** — it can't generate new frames
- You have discrete tokens for video... this looks a lot like a language modeling setup!
- *"I have a sequence of discrete tokens. I've built GPT before. Can I just... predict the next token?"*

### What This Motivates
→ *"I need a second model that looks at the token sequence and predicts what comes next — like GPT but for video tokens."*

---

## Iteration 5: Autoregressive Dynamics (Next-Token Prediction for Video)

### Goal
Train a transformer to predict next-frame tokens given previous-frame tokens. This is where your GPT experience directly applies.

### What You Build
- **Freeze** the video tokenizer from Iteration 4
- Encode training videos → sequences of discrete tokens [B, T, P]
- Train a new Space-Time Transformer:
  - Input: video tokens for frames 0..T-1
  - Output: predicted tokens for frames 1..T (shifted by 1)
  - Loss: **cross-entropy** over the token vocabulary (just like GPT!)
- **Naive autoregressive inference**:
  - Encode seed frame(s) → tokens
  - Predict next frame tokens one-by-one
  - Decode tokens → pixels using frozen tokenizer decoder

### New Concepts
- **Two-model pipeline**: tokenizer (frozen) + dynamics (trainable) — separation of concerns
- **Cross-entropy on visual tokens**: treating frame prediction exactly like language modeling
- **Autoregressive generation**: predict token by token, feed predictions back as input
- **The frame prediction = language modeling insight**: this is the conceptual bridge

### What You'll Notice (Problems)
- Generated frames are coherent short-term but **drift** over time — the ball in Pong might disappear
- There's **no control** — the model just hallucinates what happens next
- You can't say "move the paddle up" — there's no action input
- Autoregressive decoding of 64 tokens per frame is **slow** — each token depends on all previous ones

### What This Motivates
→ *"I need to condition the model on actions so I can control what happens. But where do I get action labels?"*

---

## Iteration 6: Action Conditioning with FiLM

### Goal
Add controllable actions to the dynamics model so you can steer what happens next.

### What You Build
- Start with **hardcoded actions** for Pong (up=0, down=1, stay=2) — we'll remove this requirement later
- **FiLM conditioning** (Feature-wise Linear Modulation):
  - Action embedding → linear layer → (scale γ, shift β)
  - After each FFN: `output = γ * output + β`
  - The action *modulates* the features rather than being concatenated as input
- Train dynamics model: given frames + action sequence → predict next frames
- Interactive inference: press keys to choose actions, see the world respond

### New Concepts
- **FiLM conditioning**: a powerful, lightweight way to inject conditioning information. Instead of concatenating actions as extra tokens (which changes sequence length), FiLM learns to *scale and shift* intermediate features. Think of it as: the action tells each layer "pay more attention to upward motion" by scaling those features up
- **Controllable generation**: the model generates *different* futures based on different actions
- **Action embedding**: discrete action → learned vector (just like token embeddings in GPT)

### What You'll Notice (Problems)
- Works great! But we **cheated** — we used ground-truth action labels
- Real gameplay videos on the internet don't come with action labels
- If we want to scale to arbitrary video data, we can't rely on labeled actions
- *"Can I somehow figure out what action happened just by looking at two consecutive frames?"*

### What This Motivates
→ *"I need a model that can infer actions from video alone — no labels needed."*

---

## Iteration 7: Latent Action Model (= Stage 2 of TinyWorlds)

### Goal
Learn to infer discrete actions between frames without any labels. This is the key innovation from the Genie paper.

### What You Build
- **Action Encoder**: takes two consecutive frames → encodes to action latent → FSQ quantization → discrete action token
- **Action Decoder**: takes frame_t + action token → predicts frame_t+1
- **Critical training trick**: in the decoder, mask out most frames except the first — forces the model to actually *use* the action token (otherwise it can ignore actions and just copy the previous frame)
- **Variance penalty**: auxiliary loss that penalizes low variance in action latents — prevents the model from collapsing all actions to a single code

### New Concepts
- **Unsupervised action discovery**: the model *invents* its own action vocabulary by finding the minimal information needed to explain frame-to-frame changes
- **Information bottleneck**: the FSQ quantization forces the model to compress "what changed" into a tiny discrete code (e.g., 8 possible actions with 3-bit binary FSQ)
- **Decoder masking trick**: without this, the decoder learns to just auto-complete from context and ignores the action entirely
- **Mode collapse prevention**: the variance penalty ensures the model uses all available action codes, not just one

### What You'll Notice (Problems)
- The model discovers meaningful actions! In Pong, different codes correspond to up/down/stay
- But the dynamics model from Iteration 5 decodes tokens **one at a time** — too slow
- For 64 tokens per frame, you need 64 sequential forward passes
- *"GPT generates one token at a time because each word depends on the previous. But image patches are more spatially independent — can I predict multiple tokens in parallel?"*

### What This Motivates
→ *"I need a faster decoding strategy that can predict many tokens at once."*

---

## Iteration 8: MaskGIT Parallel Decoding (= Stage 3 of TinyWorlds)

### Goal
Replace slow autoregressive token-by-token decoding with fast parallel decoding using masked prediction.

### What You Build
- **Training change** (MaskGIT-style):
  - Instead of causal next-token prediction, **randomly mask 50-100% of tokens** in each frame
  - Train the model to predict the masked tokens given unmasked ones (like BERT, not GPT!)
  - Guarantee at least one "anchor" token per patch position across time (temporal continuity)
- **Iterative parallel inference**:
  - Start with the future frame **fully masked**
  - Step 1: predict all 64 tokens in parallel, keep the ~2 most confident ones
  - Step 2: predict remaining 62, keep ~5 more most confident
  - Step 3: keep ~13 more...
  - After ~8 steps: all 64 tokens predicted (exponential schedule)
  - Decode the full frame at once

### New Concepts
- **MaskGIT**: a hybrid between autoregressive (sequential, slow) and non-autoregressive (parallel, fast but lower quality). You get most of the quality with much less compute
- **Confidence-based sampling**: at each step, predict everything but only *commit* to the highest-confidence predictions. Low-confidence predictions get re-masked and retried
- **Exponential unmasking schedule**: reveal tokens slowly at first (when context is sparse) then quickly (when most tokens are known). Like solving a jigsaw — the first pieces are hardest
- **BERT-style training for generation**: surprisingly, masked prediction (bidirectional) works for generation if you do iterative refinement

### What You'll Notice
- **8 forward passes** instead of 64 — roughly 8x faster generation!
- Quality is comparable to full autoregressive
- The iterative refinement produces coherent frames — early steps get the "gist," later steps fill in details

---

## Iteration 9: Full TinyWorlds Assembly

### Goal
Combine all components into the complete three-stage TinyWorlds pipeline.

### What You Build
- **Stage 1**: Video Tokenizer (Iteration 3-4) — trains first, then frozen
- **Stage 2**: Latent Action Model (Iteration 7) — trains on raw video, then frozen
- **Stage 3**: Dynamics Model with MaskGIT (Iteration 8) — trains on tokenized video + inferred actions
- **Full inference pipeline**:
  1. Encode seed frames → video tokens
  2. User picks action (or use inferred actions)
  3. Dynamics model predicts next frame tokens via MaskGIT
  4. Video tokenizer decodes tokens → pixels
  5. Append frame, repeat autoregressively

### New Concepts
- **Multi-stage training pipeline**: each stage produces a frozen checkpoint used by the next
- **System integration**: making three independent models work together
- **Interactive world modeling**: a playable, generative game environment

### What You'll Appreciate
- Every component exists for a reason — you've felt each problem firsthand
- The architecture is elegant: GPT-style prediction + BERT-style decoding + unsupervised action learning
- You could now read the Genie paper and understand every design choice

---

## Concept Map: How It All Connects

```
Iteration 0: MLP frame prediction
    │ Problem: no spatial structure
    ▼
Iteration 1: Conv Autoencoder
    │ Problem: continuous latents, can't generate
    ▼
Iteration 2: FSQ Discrete Bottleneck
    │ Problem: CNN limited, not transformer-based
    ▼
Iteration 3: ViT Tokenizer (= TinyWorlds Stage 1, single frame)
    │ Problem: no temporal modeling
    ▼
Iteration 4: Space-Time Transformer (= TinyWorlds Stage 1, video)
    │ Problem: compression only, can't generate
    ▼
Iteration 5: Autoregressive Dynamics
    │ Problem: no control, no actions
    ▼
Iteration 6: FiLM Action Conditioning
    │ Problem: needs action labels
    ▼
Iteration 7: Latent Action Model (= TinyWorlds Stage 2)
    │ Problem: slow autoregressive decoding
    ▼
Iteration 8: MaskGIT Decoding (= TinyWorlds Stage 3)
    │
    ▼
Iteration 9: Full TinyWorlds Assembly
```

## Key Papers to Read Along the Way

| Iteration | Paper | Why |
|-----------|-------|-----|
| 2 | [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) | Core quantization method |
| 3 | [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | Patch embeddings + transformers for vision |
| 4 | [Is Space-Time Attention All You Need for Video?](https://arxiv.org/abs/2102.05095) | Factored attention |
| 6 | [FiLM: Visual Reasoning with Conditioning](https://arxiv.org/abs/1709.07871) | Action conditioning |
| 7-9 | [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) | The paper TinyWorlds implements |
| 8 | [MaskGIT](https://arxiv.org/abs/2202.04200) | Parallel decoding |

## Dataset

Use **Pong** throughout all iterations (64x64, simple dynamics, clear actions). It trains fast on M2 and the visual patterns are easy to debug. You can use the download script in the original tinyworlds repo to get the data.

After completing the curriculum with Pong, try Sonic or Zelda at 128x128 to stress-test your understanding.
