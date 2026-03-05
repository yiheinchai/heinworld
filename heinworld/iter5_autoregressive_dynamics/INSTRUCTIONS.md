# Iteration 5: Autoregressive Dynamics — Next-Token Prediction for Video

## Goal

Train a transformer to predict the **next frame's tokens** given previous frames' tokens. This is where your GPT experience directly applies — but on video tokens instead of word tokens.

## Why This Iteration Exists

You now have a video tokenizer (Iteration 4) that converts frames into discrete tokens. This is exactly the setup for language modeling:

```
GPT:   [The, cat, sat, on] → predict [the]
This:  [frame0_tokens, frame1_tokens, frame2_tokens] → predict [frame3_tokens]
```

The dynamics model is a second, separate transformer. The video tokenizer is **frozen** — you don't change it anymore. This separation of concerns is important: the tokenizer handles compression, the dynamics model handles prediction.

## Concepts You Need

### 1. Two-Model Pipeline

```
Training:
  raw video → [FROZEN Video Tokenizer Encoder] → discrete tokens → [DYNAMICS MODEL] → predicted next tokens
                                                                           ↓
                                                              cross-entropy loss vs actual next tokens

Inference:
  seed frames → encode → tokens → dynamics predicts next tokens → [FROZEN Video Tokenizer Decoder] → pixels
```

The video tokenizer is trained in Iteration 4 and **never updated again**. You save its weights and load them here.

### 2. Cross-Entropy on Visual Tokens

In GPT, you predict the next word from a vocabulary of ~50K tokens using cross-entropy. Here, you predict the next **visual token** from a vocabulary of 1024.

```python
# Dynamics model output: logits over vocabulary for each patch position
logits = dynamics_model(input_tokens)  # (B, P, vocab_size=1024)

# Target: actual token indices from the tokenizer
target = tokenizer.encode(next_frame)  # (B, P)  values in [0, 1023]

loss = F.cross_entropy(logits.reshape(-1, 1024), target.reshape(-1))
```

This is identical to language model training — the only difference is the vocabulary is visual tokens instead of words.

### 3. Teacher Forcing with Frame Sequences

During training, you have ground-truth tokens for all frames. The dynamics model sees frames 0..T-2 and predicts frames 1..T-1 (shifted by one, like GPT).

```
Input tokens:  [frame0, frame1, frame2]  → each is (P,) = (64,) token IDs
Target tokens: [frame1, frame2, frame3]  → shifted by one timestep
```

The model uses the space-time transformer with causal temporal attention, so frame 3's prediction can only use information from frames 0, 1, 2.

### 4. Token Embedding

The dynamics model takes discrete token indices as input (not raw pixels). You need an embedding layer:

```python
token_embed = nn.Embedding(vocab_size, d_model)  # 1024 → d_model
# input_tokens: (B, T, P) integers → (B, T, P, d_model) embeddings
```

This is exactly like word embeddings in GPT.

### 5. Autoregressive Inference

At inference time, generate one frame at a time:

```
1. Encode seed frames (e.g., 3 frames) → token sequences
2. Feed tokens to dynamics model → get logits for next frame's 64 tokens
3. Sample or argmax each of the 64 token positions
4. Decode the predicted tokens → pixel frame using frozen tokenizer decoder
5. Encode the new frame → tokens, append to context
6. Repeat from step 2
```

For sampling, you can use:
- **Argmax** (greedy): always pick highest-probability token — deterministic but can be repetitive
- **Temperature sampling**: `probs = softmax(logits / temperature)` then sample — adds diversity

### 6. Saving and Loading Model Checkpoints

You need to save the trained tokenizer from Iteration 4 and load it here:

```python
# Save (after training tokenizer)
torch.save(tokenizer.state_dict(), "tokenizer.pt")

# Load (in this iteration)
tokenizer = SpaceTimeTokenizer(...)  # same architecture
tokenizer.load_state_dict(torch.load("tokenizer.pt"))
tokenizer.eval()
for p in tokenizer.parameters():
    p.requires_grad = False  # freeze — don't train it
```

---

## Your Task

### Prerequisites
- Train the Space-Time Tokenizer from Iteration 4 and save its weights
- Or, train it fresh at the beginning of this script

### Data Pipeline
1. Load Pong frames
2. Use the **frozen tokenizer encoder** to convert all frames to token indices
3. Create a dataset of `(context_tokens, target_tokens)` pairs:
   - context: tokens for frames [0, 1, 2] → shape `(3, 64)`
   - target: tokens for frames [1, 2, 3] → shape `(3, 64)`

### Dynamics Model

A space-time transformer that takes token indices and predicts logits:

```
Input: token indices (B, T, P) — integers in [0, 1023]
 → Token embedding: nn.Embedding(1024, d_model) → (B, T, P, d_model)
 → Add spatial + temporal positional embeddings
 → Flatten to (B, T*P, d_model)
 → N × SpaceTimeBlock (causal temporal attention)
 → Reshape to (B, T, P, d_model)
 → Classification head: Linear(d_model, vocab_size=1024) → (B, T, P, 1024)
```

Config: `d_model=128, n_heads=4, n_layers=4` (deeper than the tokenizer — prediction is harder than reconstruction)

### Training
- Loss: `F.cross_entropy` over the vocabulary at each position
- Adam, lr=1e-4
- ~50 epochs

### Visualization

1. Pick a clip from the dataset
2. Feed the first 3 frames as context
3. Autoregressively generate 4+ new frames
4. Decode all frames (context + generated) back to pixels using the frozen tokenizer decoder
5. Display: context frames, then generated frames — mark where generation starts

### What to Observe

1. **Short-term predictions are coherent** — the next 1-2 frames look reasonable
2. **Long-term predictions drift** — after several autoregressive steps, the scene degrades. The ball might vanish, colors might shift. This is error accumulation
3. **No control** — you can't influence what happens. The ball moves on its own, the paddles move on their own. There's no way to say "move paddle up"
4. **This is GPT for video** — the architecture and training are nearly identical to language modeling. The only difference is the "words" are image patches
